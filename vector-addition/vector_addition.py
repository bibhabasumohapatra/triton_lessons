import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda:0")
print("Using device:", DEVICE)


@triton.jit
def addition_kernel(a_ptr, b_ptr, output_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    """
    Kernel to add two vectors
    """
    PID = tl.program_id(axis=0) 
    # Compute the index of the current thread
    index = PID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load the data from the input tensors
    a = tl.load(a_ptr + index)
    b = tl.load(b_ptr + index)

    # Perform the addition
    output = a + b

    # Store the result in the output tensor
    tl.store(output_ptr + index, output)
  
def triton_vector_addition(a,b):

    output = torch.empty_like(a)

    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    addition_kernel[grid](a, b, output, n_elements, 1024)

    return output

def test_kernel(size=1024):

    torch.manual_seed(144)
    a = torch.randn(size, device=DEVICE)
    b = torch.randn(size, device=DEVICE)

    z_torch = a + b

    z_triton = triton_vector_addition(a,b)

    print("Torch result: ", z_torch)
    print("Triton result: ", z_triton)
    assert torch.allclose(z_torch, z_triton), "Results do not match!"

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'], # designators of the different entries in the legend
        line_names=['Triton', 'Torch'], # names to visibly go in the legend
        styles=[('blue', '-'), ('green', '-')], # triton will be blue; pytorch will be green
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # we'll see how this is used in a later tutorial; need it even if it's empty
    )
)
def benchmark(size, provider):
    # creating our input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    # each benchmark runs multiple times and quantiles tells matplotlib what confidence intervals to plot
    quantiles = [0.5, 0.05, 0.95]
    # defining which function this benchmark instance runs
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_vector_addition(x, y), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # Test the kernel with a size of 1024
    test_kernel(1024)
    # Test the kernel with a size of 4096
    test_kernel(4096)
    # Test the kernel with a size of 8192
    test_kernel(8192)

    benchmark.run(save_path='.', print_data=False)