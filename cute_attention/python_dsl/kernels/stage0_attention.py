#!/usr/bin/env python3
"""
Stage 0: Naive Attention Kernel (CuTe DSL)
基于用户提供的 tiled attention kernel
"""

import torch
import math

try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False
    print("WARNING: CuTe DSL not available, using PyTorch baseline")


# Hyperparameters
HEAD_DIM = 128
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64


if HAS_CUTE:
    @cute.kernel
    def naive_attention_cute_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        seq_len_q: int, seq_len_k: int, head_dim: int, scale: float,
        bM: int, bN: int, bK: int
    ):
        # 1. 线程与 Block 索引
        tid = cute.threadIdx.x
        m_block = cute.blockIdx.x
        
        # 2. 构建 Global Memory Layout 和 Tensor
        # 行主序: (SeqLen, HeadDim) -> Stride: (HeadDim, 1)
        g_layout = cute.make_layout(
            cute.make_shape(seq_len_q, head_dim),
            cute.make_stride(head_dim, 1)
        )
        
        gQ = cute.make_tensor(Q_ptr, g_layout)
        gK = cute.make_tensor(K_ptr, g_layout)
        gV = cute.make_tensor(V_ptr, g_layout)
        gO = cute.make_tensor(O_ptr, g_layout)
        
        # 3. 切片：获取当前 Block 负责的 Tile
        gQ_blk = cute.local_tile(gQ, cute.make_shape(bM, bK), cute.make_coord(m_block, 0))
        gO_blk = cute.local_tile(gO, cute.make_shape(bM, bK), cute.make_coord(m_block, 0))
        
        # 4. Shared Memory 分配 (假设 FP16, 2 bytes/element)
        sQ = cute.make_tensor(
            cute.make_smem_ptr(0),
            cute.make_layout(cute.make_shape(bM, bK))
        )
        sK = cute.make_tensor(
            cute.make_smem_ptr(bM * bK * 2),
            cute.make_layout(cute.make_shape(bN, bK))
        )
        sV = cute.make_tensor(
            cute.make_smem_ptr((bM + bN) * bK * 2),
            cute.make_layout(cute.make_shape(bN, bK))
        )
        
        # 5. 初始化 TiledMMA 和 TiledCopy
        tiled_copy = cute.make_tiled_copy_default()
        tiled_mma = cute.make_tiled_mma_default()
        
        thr_copy = tiled_copy.get_thread_slice(tid)
        thr_mma = tiled_mma.get_thread_slice(tid)
        
        # 分配寄存器 Fragment (累加器)
        tSrS = cute.partition_fragment_C(tiled_mma, cute.make_shape(bM, bN))
        tOrO = cute.partition_fragment_C(tiled_mma, cute.make_shape(bM, bK))
        cute.clear(tOrO)
        
        # 6. 从 Global Memory 加载 Q 到 Shared Memory
        cute.copy(tiled_copy, thr_copy.partition_S(gQ_blk), thr_copy.partition_D(sQ))
        cute.cp_async_wait(0)
        cute.syncthreads()
        
        # 7. 遍历 Key 和 Value 的内层循环
        num_n_blocks = (seq_len_k + bN - 1) // bN
        for n_block in range(num_n_blocks):
            gK_blk = cute.local_tile(gK, cute.make_shape(bN, bK), cute.make_coord(n_block, 0))
            gV_blk = cute.local_tile(gV, cute.make_shape(bN, bK), cute.make_coord(n_block, 0))
            
            # 拷贝当前 Tile 的 K 和 V 到 SMEM
            cute.copy(tiled_copy, thr_copy.partition_S(gK_blk), thr_copy.partition_D(sK))
            cute.copy(tiled_copy, thr_copy.partition_S(gV_blk), thr_copy.partition_D(sV))
            cute.cp_async_wait(0)
            cute.syncthreads()
            
            # [第一次 GEMM]: S = Q * K^T
            cute.clear(tSrS)
            tSrQ = thr_mma.partition_A(sQ)
            tSrK = thr_mma.partition_B(sK)
            cute.gemm(tiled_mma, tSrS, tSrQ, tSrK, tSrS)
            
            # [Softmax]: 在寄存器级别进行放缩、求最值、求指数、归一化
            max_val = float('-inf')
            for i in range(cute.size(tSrS)):
                tSrS[i] = tSrS[i] * scale
                max_val = max(max_val, tSrS[i])
            
            # Block reduce max (简化版本)
            smem_reduce = cute.SharedMemory(float, shape=(128,))
            smem_reduce[tid] = max_val
            cute.syncthreads()
            
            # Tree reduction for max
            stride = 128 // 2
            while stride > 0:
                if tid < stride:
                    smem_reduce[tid] = max(smem_reduce[tid], smem_reduce[tid + stride])
                cute.syncthreads()
                stride //= 2
            
            global_max = smem_reduce[0]
            cute.syncthreads()
            
            # Compute exp and sum
            sum_val = float(0.0)
            for i in range(cute.size(tSrS)):
                tSrS[i] = math.exp(tSrS[i] - global_max)
                sum_val += tSrS[i]
            
            smem_reduce[tid] = sum_val
            cute.syncthreads()
            
            stride = 128 // 2
            while stride > 0:
                if tid < stride:
                    smem_reduce[tid] += smem_reduce[tid + stride]
                cute.syncthreads()
                stride //= 2
            
            global_sum = smem_reduce[0]
            cute.syncthreads()
            
            # Normalize
            for i in range(cute.size(tSrS)):
                tSrS[i] = tSrS[i] / global_sum
            
            # [第二次 GEMM]: O = P * V
            tOrP = tSrS  # P 就是 softmax 后的 S
            tOrV = thr_mma.partition_B(sV)
            cute.gemm(tiled_mma, tOrO, tOrP, tOrV, tOrO)
            cute.syncthreads()
        
        # 8. 将 O 的结果从寄存器写回到 Global Memory
        cute.copy(tiled_copy, thr_copy.partition_S(tOrO), thr_copy.partition_D(gO_blk))


def attention_forward(Q, K, V, scale=None):
    """
    Attention forward
    
    Args:
        Q, K, V: (B, H, N, d) tensors
        scale: softmax scale
    
    Returns:
        O: (B, H, N, d) output tensor
    """
    B, H, N, d = Q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    
    # PyTorch baseline (when CuTe not available)
    if not HAS_CUTE:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
    
    # CuTe kernel path
    # Reshape to (B*H, N, d)
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.zeros_like(Q_flat)
    
    BH = B * H
    
    # Launch kernel for each batch-head
    for bh in range(BH):
        Q_ptr = Q_flat[bh].data_ptr()
        K_ptr = K_flat[bh].data_ptr()
        V_ptr = V_flat[bh].data_ptr()
        O_ptr = O_flat[bh].data_ptr()
        
        # Grid and block
        num_m_blocks = (N + BLOCK_M - 1) // BLOCK_M
        grid = (num_m_blocks, 1, 1)
        block = (128, 1, 1)
        
        # Launch
        naive_attention_cute_kernel[grid, block](
            Q_ptr, K_ptr, V_ptr, O_ptr,
            N, N, d, scale,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
    
    return O_flat.reshape(B, H, N, d)


# Performance Metrics
def compute_tflops(Q, time_ms):
    """Compute achieved TFLOPs"""
    B, H, N, d = Q.shape
    flops = 2 * B * H * N * N * d
    tflops = flops / time_ms / 1e9
    return tflops


def compute_tc_utilization(tflops, peak=2250):
    """Compute TC utilization percentage"""
    return tflops / peak * 100
