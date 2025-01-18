import triton
import triton.language as tl
import torch
from triton.runtime import driver
import numpy as np

DEVICE = torch.device('cuda:0')
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]


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
def persistent(a_desc, b_desc, c_desc,
               M, N, K,
               NUM_SM: tl.constexpr,             
               dtype:tl.constexpr,
               Br: tl.constexpr, Bc: tl.constexpr,Bk: tl.constexpr,
               GROUP_SZE: tl.constexpr):
    
    tile_id = tl.program_id(0)
    M_BLOCKS = tl.cdiv(M, Br)
    N_BLOCKS = tl.cdiv(N, Bc)
    K_BLOCKS = tl.cdiv(K, Bk)
    ki = -1
    pid_per_group = N_BLOCKS*GROUP_SZE

    TOTAL_TILES = M_BLOCKS * N_BLOCKS
    TILES_PER_SM = tl.cdiv(TOTAL_TILES,NUM_SM)
    accumulator = tl.zeros([Br, Bc], dtype = tl.float32)

    pid_m, pid_n = 0, 0
    off_am = 0
    off_bn = 0

    for i in range(K_BLOCKS * TILES_PER_SM):
        ki = tl.where(ki == K_BLOCKS-1, 0, ki+1)
        
        if ki == 0:
            group_id = tile_id//pid_per_group
            start_m = group_id*GROUP_SZE

            group_size = min(GROUP_SZE, M_BLOCKS-start_m)

            pid_m = start_m + (tile_id%pid_per_group)%group_size
            pid_n = (tile_id%pid_per_group)//group_size
            tile_id += NUM_SM
        
            off_am = pid_m*Br
            off_bn = pid_n*Bc

        off_k = ki*Bk
        
        a = tl._experimental_descriptor_load(a_desc, [off_am, off_k], [Br, Bk], dtype=tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc, [off_bn, off_k], [Bc, Bk], dtype=tl.float8e4nv)

        accumulator = tl.dot(a, b.T, accumulator)

        if ki == K_BLOCKS-1:
            c = accumulator.to(tl.bfloat16)

            tl._experimental_descriptor_store(c_desc, c, [off_am, off_bn])

            accumulator = tl.zeros([Br, Bc], dtype = tl.float32)


def tma_mm(A,B,C):

    M, K = A.shape
    K, N = B.shape

    dtype = tl.float8e4nv
    
    Br = 64
    Bc = 64
    Bk = 256
    GROUP_SZE = 8
    TMA_SIZE = 256
    num_warps = 4
    num_stages = 4

    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)

    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(A.data_ptr(), 
                                                                           M, K, 
                                                                           Br, Bk, 
                                                                           A.element_size(), desc_a)
    
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(B.data_ptr(),
                                                                            N, K,
                                                                            Bc, Bk,
                                                                            B.element_size(), desc_b)
    
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(C.data_ptr(),
                                                                            M, N,
                                                                            Br, Bc,
                                                                            C.element_size(), desc_c)
    desc_a = torch.tensor(desc_a, device='cuda')
    desc_b = torch.tensor(desc_b, device='cuda')
    desc_c = torch.tensor(desc_c, device='cuda')
    

    persistent[(min(NUM_SM, triton.cdiv(M,Br) * triton.cdiv(N, Bc)), )](desc_a, desc_b, desc_c,
                                                                        NUM_SM, 
                                                                        M, N, K, 
                                                                        dtype,
                                                                        Br, Bc, Bk,
                                                                        GROUP_SZE, num_warps = num_warps,
                                                                       num_stages = num_stages)
    
    return C


def grid_quant(A,B,C):
    
    A_float8, scale_a, B_float8, scale_b = quantization(A,B, 64, 32)


    C = tma_mm(A_float8, B_float8, C)

    return C



def torch_impl(A,B):
    assert A.dtype == torch.bfloat16
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
    
    
    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]

    dtype = torch.bfloat16
    device = torch.device('cuda')
    
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
    M, K, N = 1024, 4096, 4096
    
    dtype = torch.bfloat16
    device = torch.device('cuda')
    torch.cuda.empty_cache()    
    A = torch.randn((M,K), dtype=dtype, device=device)
    B = torch.randn((N,K), dtype=dtype, device=device)
    C = torch.empty((M,N), dtype=dtype, device=device)
    
    c = grid_quant(A,B,C)
    ref = torch.matmul(A,B.T)
    mdist = torch.max(torch.abs((c.to(torch.float16) - ref)))
    dist = torch.dist(c.to(torch.float16), ref)
    print(mdist)
    print(dist)

    print(c[0])
    print(ref[0])

