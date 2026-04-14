#!/usr/bin/env python3
"""
Stage 0: Naive Attention Kernel (CuTe DSL)
每个 CTA 处理一行 query
"""

import torch

# CuTe DSL imports
from cutlass.cute.runtime import *
from cutlass import cute
import cutlass
import numpy as np
import torch


HEAD_DIM = 128
BLOCK_N = 64
NUM_THREADS = 128


@cute.kernel
def naive_attention_kernel(
    Q: cute.Tensor,      # (BH, N, d)
    K: cute.Tensor,      # (BH, N, d)
    V: cute.Tensor,      # (BH, N, d)
    O: cute.Tensor,      # (BH, N, d)
    scale: float,
    seq_len: int
):
    """每个 CTA 处理一行 query"""
    
    # Grid: (N, BH) - 每个 CTA 负责 (query_row, batch_head)
    query_idx = cute.blockIdx.x
    bh_idx = cute.blockIdx.y
    tid = cute.threadIdx.x
    tid = cute.threadIdx.x
    
    # 取出当前 batch-head 的 Q, K, V slice
    Q_bh = Q[bh_idx]      # (N, d)
    K_bh = K[bh_idx]      # (N, d)
    V_bh = V[bh_idx]      # (N, d)
    O_bh = O[bh_idx]      # (N, d)
    
    q_vec = Q_bh[query_idx]  # (d,) 当前 query 行
    
    # Shared memory
    scores = cute.SharedMemory(float, shape=(seq_len,))
    output = cute.SharedMemory(float, shape=(HEAD_DIM,))
    smem_reduce = cute.SharedMemory(float, shape=(NUM_THREADS,))
    
    # ----------------------------------------------------------
    # Step 1: Compute scores = Q @ K^T
    # 每个线程处理部分 kv positions
    # ----------------------------------------------------------
    # 初始化 output
    for d_idx in cute.thread_partition(range(HEAD_DIM), tid, NUM_THREADS):
        output[d_idx] = 0.0
    
    # Compute attention scores
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        k_vec = K_bh[j]  # (d,)
        dot = float(0.0)
        for dd in range(HEAD_DIM):
            dot += q_vec[dd] * k_vec[dd]
        scores[j] = dot * scale
    
    cute.syncthreads()
    
    # ----------------------------------------------------------
    # Step 2: Softmax over scores
    # ----------------------------------------------------------
    # 2a: Find max (reduction)
    local_max = float('-inf')
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        local_max = max(local_max, scores[j])
    
    smem_reduce[tid] = local_max
    cute.syncthreads()
    
    # Tree reduction for max
    stride = NUM_THREADS // 2
    while stride > 0:
        if tid < stride:
            smem_reduce[tid] = max(smem_reduce[tid], smem_reduce[tid + stride])
        cute.syncthreads()
        stride //= 2
    
    global_max = smem_reduce[0]
    cute.syncthreads()
    
    # 2b: Compute exp and sum
    local_sum = float(0.0)
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        val = cute.exp(scores[j] - global_max)
        scores[j] = val
        local_sum += val
    
    smem_reduce[tid] = local_sum
    cute.syncthreads()
    
    stride = NUM_THREADS // 2
    while stride > 0:
        if tid < stride:
            smem_reduce[tid] += smem_reduce[tid + stride]
        cute.syncthreads()
        stride //= 2
    
    global_sum = smem_reduce[0]
    cute.syncthreads()
    
    # 2c: Normalize
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        scores[j] /= global_sum
    
    cute.syncthreads()
    
    # ----------------------------------------------------------
    # Step 3: Compute output = scores @ V
    # 每个线程负责部分维度
    # ----------------------------------------------------------
    for d_idx in cute.thread_partition(range(HEAD_DIM), tid, NUM_THREADS):
        acc = float(0.0)
        for j in range(seq_len):
            acc += scores[j] * V_bh[j][d_idx]
        O_bh[query_idx][d_idx] = acc


# ============================================================
# Launch Interface
# ============================================================
def attention_forward(Q, K, V, scale=None):
    B, H, N, d = Q.shape
    assert d == HEAD_DIM, f"Expected HEAD_DIM={HEAD_DIM}, got {d}"
    
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    
    # Reshape to (B*H, N, d)
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.zeros_like(Q_flat)
    
    BH = B * H
    
    # CuTe tensor wrappers
    Q_cute = from_dlpack(Q_flat)
    K_cute = from_dlpack(K_flat)
    V_cute = from_dlpack(V_flat)
    O_cute = from_dlpack(O_flat)
    
    # Launch kernel
    grid = (N, BH)
    block = (NUM_THREADS,)
    
    # Compile kernel first
    compiled_kernel = cute.compile(
        naive_attention_kernel,
        Q_cute, K_cute, V_cute, O_cute,
        scale, N
    )
    
    # Then launch
    compiled_kernel.launch(
        grid=grid, block=block,
        args=(Q_cute, K_cute, V_cute, O_cute, scale, N)
    )
    
    return O_flat.reshape(B, H, N, d)


# ============================================================
# Performance Metrics
# ============================================================
def compute_tflops(Q, time_ms):
    """Compute achieved TFLOPs"""
    B, H, N, d = Q.shape
    flops = 2 * B * H * N * N * d
    tflops = flops / time_ms / 1e9
    return tflops


def compute_tc_utilization(tflops, peak=2250):
    """Compute TC utilization percentage"""
    return tflops / peak * 100
