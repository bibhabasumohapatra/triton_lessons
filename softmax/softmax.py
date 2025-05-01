import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
print("Using device:", DEVICE)

torch.manual_seed(0)



def naive_softmax(x, dim=-1):
    """
    Naive implementation of softmax
    """
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    output_custom = x_exp/x_exp_sum
    return output_custom


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)

NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

@triton.jit
def softmax_kernel(input_ptr, output_ptr, 
                   input_stride, 
                   output_stride,
                   n_rows, 
                   n_columns,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr,
                   ):
    """
    Kernel to compute softmax
    """
    # Compute the index of the current thread
    PID = tl.program_id(axis=0)
    ## total number of programs
    num_programs = tl.num_programs(0)

    for row_idx in tl.range(PID, n_rows, num_programs,num_stages=num_stages):
        # Compute the index of the current thread
        row_start_ptr = row_idx * input_stride + input_ptr
        col_offset = tl.arange(0, BLOCK_SIZE)
        index = col_offset + row_start_ptr

        mask = col_offset < n_columns # do not go out of bounds to grab the places in next row
        # Compute the softmax
        row = tl.load(index, mask=mask,other=float('-inf'))

        x_exp = row - tl.max(row, axis=0)
        exps = tl.exp(x_exp) #Shape Block Size
        softmax = exps / tl.sum(exps, axis=0) #Shape Block Size
        output_row_start_ptr = output_ptr + row_idx*output_stride
        tl.store(output_row_start_ptr + col_offset, softmax,mask=mask)
# fetching a dictionary full of the GPU's specifications
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
# each Streaming Multi-processor (SM) is like a mini-processor that can run multiple programs
NUM_SM = properties["multiprocessor_count"] 
# registers are the fastest memory on the GPU
NUM_REGS = properties["max_num_regs"] 
    # each SM has a limited number of registers; 
    # programs share these registers, so using too many per program limits parallelism
# each SM has a dedicated pool of SRAM that it can access
# since there can be multiple programs per SM, those programs share the same SRAM
    # ^that will be very useful information later in the matmul tutorial
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
# a warp is a group of threads that execute together
# a thread can be thought of as analagous to a single CPU core, but far more limited in the operations it can do
WARP_SIZE = properties["warpSize"]# usually 32 on nvidia GPUs and 64 on AMD

def triton_softmax(x):
    """
    Triton implementation of softmax
    """
    output = torch.empty_like(x)
    n_rows, n_columns = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_columns)

    ## We need to use AutoTune or understand GPU arch to get these values
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8 
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2 #(IDK this part)

    ## we need to warm up our kernel
    kernel = softmax_kernel.warmup(
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_rows,
        n_columns,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,))
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    print(f"n_regs_per_program: {n_regs_per_program}, sram_needed_per_program: {sram_needed_per_program}")
    ## num_registers = (n_regs_per_program * num_warps*WARP_SIZE)
    reg_occupancy = NUM_REGS // (n_regs_per_program * num_warps * WARP_SIZE)
    sram_occupancy = TOTAL_SRAM_PER_SM//sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM*programs_per_sm, n_rows)

    grid = (num_programs,1,1)

    kernel[grid](x, output, x.stride(0), output.stride(0), 
                 n_rows, n_columns)
    
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)



# Run the benchmark
if __name__ == "__main__":

    X = torch.randn(1823, 781, device=DEVICE)

    print(X)
    torch_out = naive_softmax(X)
    print("Torch result: \n", torch_out)

    triton_out = triton_softmax(X)
    print("Triton result: \n", triton_out)
    print("nn Softmax result: \n", torch.nn.functional.softmax(X, dim=-1))
    # assert torch.allclose(torch_out, triton_out), "Results do not match!"
    # benchmark.run(show_plots=True, print_data=True)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-3, rtol=1e-3)
    print("Results match!")
    

