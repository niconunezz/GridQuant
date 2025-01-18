import torch
import triton
from main import grid_quant


# some approximate thing in torch
def torch_impl(A,B):
    
    m_a = torch.max(torch.abs(A))
    m_b = torch.max(torch.abs(B))

    A *= 1/m_a
    B *= 1/m_b

    out = torch.matmul(A, B.T)

    return out

@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[(2**i) for i in range(13)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names= ['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='perf',
            args = {}
    
        )
)

def benchmark(N, provider):
    torch.manual_seed(20)
    M = K = N 
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    
    
    A = torch.randn((M,K), dtype=dtype, device=device)
    B = torch.randn((N,K), dtype=dtype, device=device)
    B = B.T.contiguous()
    C = torch.empty((M,N), dtype=dtype, device=device)

    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_impl(A,B))


    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: grid_quant(A, B ,C))
    

    return ms


if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True)