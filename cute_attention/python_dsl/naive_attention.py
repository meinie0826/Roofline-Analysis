#!/usr/bin/env python3
"""
Naive Attention Kernel using CuTe DSL
每个 CTA 处理一行 query
"""

import torch
import numpy as np

# CuTe DSL imports
try:
    import cuda.cooperative.experimental as cudax
    from cutlass.cute.runtime import from_dlpack
    from cutlass import cute
    HAS_CUTE = True
except ImportError as e:
    HAS_CUTE = False
    CUTE_ERROR = str(e)
    print(f"Warning: CuTe not available: {e}")


# ============================================================
# 超参数
# ============================================================
HEAD_DIM = 64
BLOCK_N = 64
NUM_THREADS = 128


# ============================================================
# Naive Attention Kernel (简化版，直接用 PyTorch)
# ============================================================
def naive_attention_pytorch(Q, K, V):
    """
    Reference implementation using PyTorch
    Q, K, V: (B, H, N, d)
    """
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    
    # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax
    weights = torch.softmax(scores, dim=-1)
    
    # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
    output = torch.matmul(weights, V)
    
    return output


# ============================================================
# Naive Attention Kernel (Python 实现，模拟 CuTe)
# ============================================================
def naive_attention_kernel_sim(
    Q,  # (BH, N, d)
    K,  # (BH, N, d)
    V,  # (BH, N, d)
    O,  # (BH, N, d)
    scale: float,
    seq_len: int
):
    """
    Simulated kernel: 每个 CTA 处理一个 query row
    """
    BH = Q.shape[0]
    
    for bh_idx in range(BH):
        for query_idx in range(seq_len):
            # 取出当前 query vector
            q_vec = Q[bh_idx, query_idx]  # (d,)
            
            # Step 1: Compute scores = Q @ K^T
            scores = torch.zeros(seq_len, dtype=torch.float32, device=Q.device)
            for j in range(seq_len):
                k_vec = K[bh_idx, j]  # (d,)
                scores[j] = torch.dot(q_vec, k_vec) * scale
            
            # Step 2: Softmax
            # 2a: Find max
            max_val = scores.max()
            
            # 2b: Exp and sum
            exp_scores = torch.exp(scores - max_val)
            sum_exp = exp_scores.sum()
            
            # 2c: Normalize
            probs = exp_scores / sum_exp
            
            # Step 3: Output = probs @ V
            out_vec = torch.zeros(HEAD_DIM, dtype=torch.float32, device=Q.device)
            for j in range(seq_len):
                v_vec = V[bh_idx, j]  # (d,)
                out_vec += probs[j] * v_vec
            
            O[bh_idx, query_idx] = out_vec


def naive_attention(Q, K, V):
    """
    Q, K, V: torch.Tensor of shape (B, H, N, d), float32, on CUDA
    Returns: O of same shape
    """
    B, H, N, d = Q.shape
    assert d == HEAD_DIM
    
    # Reshape to (B*H, N, d)
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.zeros_like(Q_flat)
    
    scale = 1.0 / (d ** 0.5)
    
    # Run simulated kernel
    naive_attention_kernel_sim(Q_flat, K_flat, V_flat, O_flat, scale, N)
    
    return O_flat.reshape(B, H, N, d)


# ============================================================
# CuTe Kernel (if available)
# ============================================================
if HAS_CUTE:
    @cute.kernel
    def naive_attention_kernel_cute(
        Q: cute.Tensor,      # (BH, N, d)
        K: cute.Tensor,      # (BH, N, d)
        V: cute.Tensor,      # (BH, N, d)
        O: cute.Tensor,      # (BH, N, d)
        scale: float,
        seq_len: int
    ):
        # 每个 CTA 负责一个 (bh, query_row)
        query_idx = cute.blockIdx.x
        bh_idx = cute.blockIdx.y
        tid = cute.threadIdx.x
        
        # 取出当前 batch-head 的 slice
        Q_bh = Q[bh_idx]      # (N, d)
        K_bh = K[bh_idx]
        V_bh = V[bh_idx]
        O_bh = O[bh_idx]
        
        q_vec = Q_bh[query_idx]  # (d,)
        
        # Shared memory for scores
        scores = cute.SharedMemory(float, shape=(seq_len,))
        smem_reduce = cute.SharedMemory(float, shape=(NUM_THREADS,))
        
        # Step 1: Compute Q @ K^T
        # 每个线程处理部分 kv positions
        for j in range(tid, seq_len, NUM_THREADS):
            k_vec = K_bh[j]  # (d,)
            dot = 0.0
            for dd in range(HEAD_DIM):
                dot += q_vec[dd] * k_vec[dd]
            scores[j] = dot * scale
        
        cute.syncthreads()
        
        # Step 2: Softmax
        # 2a: Find max
        local_max = float('-inf')
        for j in range(tid, seq_len, NUM_THREADS):
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
        
        # 2b: Exp and sum
        local_sum = 0.0
        for j in range(tid, seq_len, NUM_THREADS):
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
        for j in range(tid, seq_len, NUM_THREADS):
            scores[j] /= global_sum
        
        cute.syncthreads()
        
        # Step 3: Compute output = scores @ V
        # 每个线程负责部分维度
        for d_idx in range(tid, HEAD_DIM, NUM_THREADS):
            acc = 0.0
            for j in range(seq_len):
                acc += scores[j] * V_bh[j][d_idx]
            O_bh[query_idx][d_idx] = acc
    
    def naive_attention_cute(Q, K, V):
        """Launch CuTe kernel"""
        B, H, N, d = Q.shape
        assert d == HEAD_DIM
        
        Q_flat = Q.reshape(B * H, N, d).contiguous()
        K_flat = K.reshape(B * H, N, d).contiguous()
        V_flat = V.reshape(B * H, N, d).contiguous()
        O_flat = torch.zeros_like(Q_flat)
        
        scale = 1.0 / (d ** 0.5)
        BH = B * H
        
        Q_cute = from_dlpack(Q_flat)
        K_cute = from_dlpack(K_flat)
        V_cute = from_dlpack(V_flat)
        O_cute = from_dlpack(O_flat)
        
        grid = (N, BH)
        block = (NUM_THREADS,)
        
        naive_attention_kernel_cute[grid, block](
            Q_cute, K_cute, V_cute, O_cute,
            scale, N
        )
        
        return O_flat.reshape(B, H, N, d)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Testing Naive Attention Kernel")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
    
    torch.manual_seed(42)
    
    # Small test first
    B, H, N, d = 1, 1, 128, HEAD_DIM
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    
    print(f"\nTest config: B={B}, H={H}, N={N}, d={d}")
    
    # Reference
    print("Running PyTorch reference...")
    O_ref = naive_attention_pytorch(Q, K, V)
    
    # Our kernel
    print("Running our naive kernel...")
    O_ours = naive_attention(Q, K, V)
    
    # Compare
    max_diff = (O_ours - O_ref).abs().max().item()
    mean_diff = (O_ours - O_ref).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("  ✓ PASSED!")
    else:
        print("  ✗ FAILED!")
        
    # Performance test
    print("\n" + "="*60)
    print("Performance Test")
    print("="*60)
    
    import time
    
    N = 1024
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = naive_attention_pytorch(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(100):
        _ = naive_attention_pytorch(Q, K, V)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000
    
    # Benchmark our kernel
    start = time.perf_counter()
    for _ in range(100):
        _ = naive_attention(Q, K, V)
    torch.cuda.synchronize()
    our_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"\nN={N}:")
    print(f"  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Ours:    {our_time:.3f} ms")
    print(f"  Ratio:   {our_time/pytorch_time:.2f}x")
