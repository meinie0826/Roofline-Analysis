#!/usr/bin/env python3
"""
Stage 0: Naive Attention Kernel (CuTe DSL)
每个 CTA 处理一个 (batch, head, query_row) 的 attention

基于用户提供的 naive attention 思路实现：
1. 加载整个 query row 到 registers
2. 遍历所有 KV，计算 dot product
3. 在 shared memory 做 softmax reduction
4. 累加 output
"""

import torch
import math

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass._mlir import ir
    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False
    print("WARNING: CuTe DSL not available, using PyTorch baseline")


# Hyperparameters
HEAD_DIM = 128
BLOCK_SIZE = 128
NUM_THREADS = 128


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
        print("INFO: Using PyTorch baseline (CuTe not available)")
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
    
    print("INFO: Using CuTe kernel")
    
    # Ensure MLIR context exists
    if ir.Context.current is None:
        ctx = ir.Context()
        ctx.__enter__()
    
    # Reshape to (B*H, N, d)
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.zeros_like(Q_flat)
    
    BH = B * H
    
    # Launch kernel for each batch-head
    for bh in range(BH):
        # Convert to CuTe tensors
        Q_bh = from_dlpack(Q_flat[bh])  # (N, d)
        K_bh = from_dlpack(K_flat[bh])  # (N, d)
        V_bh = from_dlpack(V_flat[bh])  # (N, d)
        O_bh = from_dlpack(O_flat[bh])  # (N, d)
        
        # Grid: one block per query row
        grid = (N, 1, 1)
        block = (NUM_THREADS, 1, 1)
        
        # Launch
        try:
            naive_attention_kernel(
                Q_bh, K_bh, V_bh, O_bh,
                N, scale
            ).launch(
                grid=grid,
                block=block
            )
        except Exception as e:
            print(f"ERROR: Kernel launch failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return O_flat.reshape(B, H, N, d)


if HAS_CUTE:
    @cute.kernel
    def naive_attention_kernel(
        Q: cute.Tensor,  # (N, d)
        K: cute.Tensor,  # (N, d)
        V: cute.Tensor,  # (N, d)
        O: cute.Tensor,  # (N, d)
        seq_len: int,
        scale: float
    ):
        """
        Naive attention kernel
        
        每个 CTA 处理一个 query row:
        1. 加载 q_vec 到寄存器
        2. 遍历所有 KV，计算 scores = Q @ K^T
        3. Softmax (需要 shared memory reduction)
        4. O = scores @ V
        """
        
        # 1. 线程和Block索引
        tid, _, _ = cute.arch.thread_idx()
        query_idx, _, _ = cute.arch.block_idx()
        
        # 2. 加载当前 query row 到寄存器
        q_vec = Q[query_idx]  # (d,) 自动加载到寄存器
        
        # 3. Shared memory 分配
        smem = cutlass.utils.SmemAllocator()
        # 用于存储 scores
        scores_ptr = smem.allocate(cutlass.Float32, seq_len)
        scores = cute.make_tensor(scores_ptr, cute.make_layout((seq_len,)))
        # 用于存储 reduction 结果
        reduce_ptr = smem.allocate(cutlass.Float32, NUM_THREADS)
        reduce_buf = cute.make_tensor(reduce_ptr, cute.make_layout((NUM_THREADS,)))
        
        # 4. Step 1: 计算 scores = q_vec @ K^T
        # 每个线程处理部分 KV positions
        local_max = float('-inf')
        for kv_idx in range(seq_len):
            # 简单的 thread partition
            if kv_idx % NUM_THREADS == tid:
                k_vec = K[kv_idx]  # (d,)
                # Compute dot product
                score = float(0.0)
                for dd in range(HEAD_DIM):
                    score += q_vec[dd] * k_vec[dd]
                scores[kv_idx] = score * scale
                local_max = max(local_max, score * scale)
        
        cute.syncthreads()
        
        # 5. Step 2: Softmax
        # 5a. Find global max
        reduce_buf[tid] = local_max
        cute.syncthreads()
        
        # Tree reduction
        stride = NUM_THREADS // 2
        while stride > 0:
            if tid < stride:
                reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + stride])
            cute.syncthreads()
            stride //= 2
        
        global_max = reduce_buf[0]
        cute.syncthreads()
        
        # 5b. Compute exp and sum
        local_sum = float(0.0)
        for kv_idx in range(seq_len):
            if kv_idx % NUM_THREADS == tid:
                val = math.exp(scores[kv_idx] - global_max)
                scores[kv_idx] = val
                local_sum += val
        
        reduce_buf[tid] = local_sum
        cute.syncthreads()
        
        stride = NUM_THREADS // 2
        while stride > 0:
            if tid < stride:
                reduce_buf[tid] += reduce_buf[tid + stride]
            cute.syncthreads()
            stride //= 2
        
        global_sum = reduce_buf[0]
        cute.syncthreads()
        
        # 5c. Normalize
        for kv_idx in range(seq_len):
            if kv_idx % NUM_THREADS == tid:
                scores[kv_idx] /= global_sum
        
        cute.syncthreads()
        
        # 6. Step 3: Compute output = scores @ V
        # 每个线程负责部分维度
        for d_idx in range(HEAD_DIM):
            if d_idx % NUM_THREADS == tid:
                acc = float(0.0)
                for kv_idx in range(seq_len):
                    acc += scores[kv_idx] * V[kv_idx][d_idx]
                O[query_idx][d_idx] = acc


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
