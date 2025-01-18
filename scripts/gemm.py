import triton
import triton.language as tl
import torch
from triton.runtime import driver
import numpy as np

@triton.jit
def persistent(a_desc, b_desc, c_desc,
               M, N, K,
               NUM_SM: tl.constexpr,             
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

    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]
    
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
                                                                        Br, Bc, Bk,
                                                                        GROUP_SZE, num_warps = num_warps,
                                                                        num_stages = num_stages)
    
    return C