import torch
from quant import quantization
from gemm import tma_mm


def grid_quant(A, B, C):
    BLOCK_SZE = 256
    GROUP_SZE = 32

    A_float8, scale_a, B_float8, scale_b = quantization(A, B, BLOCK_SZE, GROUP_SZE)
    C = tma_mm(A_float8, B_float8, C)

    return C



if __name__ == '__main__':
    M, K, N = 4096, 4096, 4096
    
    dtype = torch.bfloat16
    device = torch.device('cuda')
    A = torch.randn((M,K), dtype=dtype, device=device)
    B = torch.randn((N,K), dtype=dtype, device=device)
    C = torch.empty((M,N), dtype=dtype, device=device)
    
    c = grid_quant(A,B,C)
    
