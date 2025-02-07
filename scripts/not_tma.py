import torch
import triton
import triton.language as tl
from triton.runtime import driver
import numpy as np

@triton.jit
def block_quantization(A, 
                       A_row_stride,
                       A_float_8,
                       scale_matrix_ptr,
                       M:tl.constexpr,
                       N:tl.constexpr,
                       NUM_MSBLOCKS:tl.constexpr,
                       NUM_SBLOCKS:tl.constexpr,
                       GROUP_SZE:tl.constexpr,
                       BLOCK_SZE:tl.constexpr):
    
    pid = tl.program_id(0)

    n_blocks = tl.cdiv(N, BLOCK_SZE)
    pid_m = pid//n_blocks
    pid_n = pid%n_blocks

    block_off_m = pid_m*BLOCK_SZE*A_row_stride
    block_off_n = pid_n*BLOCK_SZE
    block_off = block_off_m + block_off_n
    A += block_off
    A_float_8 += block_off
    max_vals = tl.zeros([GROUP_SZE, GROUP_SZE], dtype = tl.bfloat16)

    for i in range(0, NUM_SBLOCKS):
        
        # position in subgroup
        curr_m = i // NUM_MSBLOCKS
        curr_n = i % NUM_MSBLOCKS

        off = tl.arange(0, GROUP_SZE)
        off_m = (off[:, None] + curr_m*GROUP_SZE)*A_row_stride
        off_n = off[None, :] + curr_n*GROUP_SZE
        offset = off_m + off_n
        mask =  (((off_m + block_off_m) < M*A_row_stride) & ((off_n+block_off_n) < N))
        block = tl.load(A + offset, mask=mask, other=0.0, eviction_policy='evict_last')
        max_vals = tl.maximum(max_vals,tl.abs(block)).to(tl.bfloat16)
    
    sf = tl.max(max_vals).to(tl.bfloat16)

    for i in range(0, NUM_SBLOCKS):
        
        curr_m = i // NUM_MSBLOCKS
        curr_n = i % NUM_MSBLOCKS

        off = tl.arange(0, GROUP_SZE)
        off_m = (off[:, None] + curr_m*GROUP_SZE)*A_row_stride
        off_n = off[None, :] + curr_n*GROUP_SZE
        offset = off_m + off_n
        mask =  (((off_m + block_off_m) < M*A_row_stride) & ((off_n+block_off_n) < N))
        block = tl.load(A + offset, mask=mask, other=0.0, eviction_policy='evict_last')

        scaled_matrix = block * 1/sf
        fp8 = tl.cast(scaled_matrix, tl.float8e5, fp_downcast_rounding='rtne')
        tl.store(A_float_8 + offset, fp8 , mask=mask)
    
    tl.store(scale_matrix_ptr + pid_m*n_blocks + pid_n, sf)


def quantization(A, B, BLOCK_SZE, GROUP_SZE):

    device = 'cuda'
    M, N = A.shape
    
    M_BLOCKS = triton.cdiv(M,BLOCK_SZE) # blocks on each axis
    NUM_SMBLOCKS = triton.cdiv(BLOCK_SZE,GROUP_SZE) # subblocks on each axis
    NUM_SBLOCKS = NUM_SMBLOCKS**2 # total subblocks
    
    A_float8 = torch.empty((M,N), dtype=torch.float8_e5m2, device = device)
    scale_a = torch.empty((M_BLOCKS,M_BLOCKS), dtype=torch.bfloat16, device=device)
    B_float8 = torch.empty((M,N), dtype=torch.float8_e5m2, device = device)
    scale_b = torch.empty((M_BLOCKS,M_BLOCKS), dtype=torch.bfloat16, device=device)

    block_quantization[((triton.cdiv(M, BLOCK_SZE) * triton.cdiv(N, BLOCK_SZE)), )](A, A.stride(0),
                                                                                    A_float8, scale_a, M, N, NUM_SMBLOCKS,
                                                                                    NUM_SBLOCKS, GROUP_SZE, BLOCK_SZE)
    

    block_quantization[((triton.cdiv(M, BLOCK_SZE) * triton.cdiv(N, BLOCK_SZE)), )](B, B.stride(0),
                                                                                    B_float8, scale_b, M, N, NUM_SMBLOCKS,
                                                                                    NUM_SBLOCKS, GROUP_SZE, BLOCK_SZE)
    

    return A_float8, scale_a, B_float8, scale_b


@triton.jit
def block_dequantization(c_ptr, out_c,
                   M,N,
                   scale_ab_ptr,
                   cm_stride,
                   BLOCK_SZE:tl.constexpr):
    
    pid = tl.program_id(0)

    n_pid_n = tl.cdiv(N, BLOCK_SZE)
    pid_m = pid // n_pid_n
    pid_n = pid % n_pid_n

    scale = tl.load(scale_ab_ptr + pid)


    start_m = pid_m*BLOCK_SZE
    start_n = pid_n*BLOCK_SZE

    off_m = (start_m + tl.arange(0, BLOCK_SZE)) 
    off_n = (start_n + tl.arange(0, BLOCK_SZE))
    mask = ((off_m < M)[:, None]) & ((off_n < N)[None, :])

    c_ptrs = c_ptr + off_m[:,None]* cm_stride + off_n[None, :]
    c = tl.load(c_ptrs, mask, other= 0.0)

    c_out = c*scale
    out_c_ptrs =out_c + off_m[:,None]* cm_stride + off_n[None, :]
    tl.store(out_c_ptrs, c_out, mask)


def dequantization(C, scale, BLOCK_SZE):
    new_C = torch.empty_like(C)
    M,N = C.shape
    block_dequantization[((triton.cdiv(M, BLOCK_SZE) * triton.cdiv(N, BLOCK_SZE)), )](C, new_C, M, N, scale, C.stride(0), BLOCK_SZE)

    return new_C


@triton.jit
def grouping(A, B, C,
                am_stride, ak_stride,
                bk_stride, bn_stride,
                cm_stride, cn_stride,
                M, N, K,
                Br:tl.constexpr, Bc:tl.constexpr, Bk:tl.constexpr,
                GROUP_SZE_M:tl.constexpr,
                ):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, Br)
    num_pid_n = tl.cdiv(N, Bc)
    num_pid_in_group = GROUP_SZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * Br + tl.arange(0, Br)) % M
    offs_bn = (pid_n * Bc + tl.arange(0, Bc)) % N

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, Br), Br)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, Bc), Bc)

    offs_k = tl.arange(0, Bk)
    a_ptrs = A + (offs_am[:, None] * am_stride + offs_k[None, :] * ak_stride)
    b_ptrs = B + (offs_k[:, None] * bn_stride  + offs_bn[None, :] * bk_stride)

    accumulator = tl.zeros((Br, Bc), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, Bk)):

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * Bk, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * Bk, other=0.0)

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += Bk * ak_stride
        b_ptrs += Bk * bn_stride 

    
    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * Br + tl.arange(0, Br)
    offs_cn = pid_n * Bc + tl.arange(0, Bc)
    c_ptrs = C + cm_stride * offs_cm[:, None] + cn_stride * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def tma_mm(A,B,C):

    M, K = A.shape
    K, N = B.shape

    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    
    
    Br = 16
    Bc = 16
    Bk = 16
    GROUP_SZE = 8
    TMA_SIZE = 256
    num_warps = 4
    num_stages = 4



    grouping[(triton.cdiv(M,Br) * triton.cdiv(N, Bc), )](A, B, C, A.stride(0),A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                                                                        M, N, K, 
                                                                        Br, Bc, Bk,
                                                                        GROUP_SZE, num_warps = num_warps,
                                                                        num_stages = num_stages)
    
    return C



def grid_quant(A, B, C):
    BLOCK_SZE = 16
    GROUP_SZE = 4

    A_float8, scale_a, B_float8, scale_b = quantization(A, B, BLOCK_SZE, GROUP_SZE)
    C = tma_mm(A_float8, B_float8, C)

    scale_ab = scale_a*scale_b

    C = dequantization(C, scale_ab, BLOCK_SZE)

    return C, scale_a.mean(), scale_b.mean()




if __name__ == '__main__':
    M, K, N = 4096, 4096, 4096
    
    dtype = torch.bfloat16
    device = torch.device('cuda')
    
    A = torch.randn((M,K), dtype=dtype, device=device)
    B = torch.randn((N,K), dtype=dtype, device=device)
    C = torch.empty((M,N), dtype=dtype, device=device)
    
    c = grid_quant(A,B,C)
    
  
