import cuda.cooperative.experimental as cudax
from cutlass.cute.runtime import *
from cutlass.cute import *
import cutlass
import numpy as np
import torch

# ============================================================
# Naive Attention: 每个 CTA 处理一行 query
# Q, K, V: [B*H, N, d]
# O:       [B*H, N, d]
# ============================================================

# 超参数
HEAD_DIM = 64
BLOCK_N = 64      # 每次处理的 KV tile 的 seq 长度
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
    # ----------------------------------------------------------
    # 索引: 每个 CTA 负责一个 (bh, query_row)
    # grid = (N, BH)
    # ----------------------------------------------------------
    query_idx = cute.blockIdx.x
    bh_idx    = cute.blockIdx.y
    tid       = cute.threadIdx.x

    # ----------------------------------------------------------
    # 取出当前 batch-head 的 Q, K, V slice
    # Q_bh: (N, d),  q_vec: (d,)
    # ----------------------------------------------------------
    Q_bh = Q[bh_idx]           # (N, d)
    K_bh = K[bh_idx]           # (N, d)
    V_bh = V[bh_idx]           # (N, d)
    O_bh = O[bh_idx]           # (N, d)

    q_vec = Q_bh[query_idx]    # (d,)  -- 当前 query 行

    # ----------------------------------------------------------
    # Step 1: 计算 scores = Q @ K^T, 即 q_vec dot 每个 k_vec
    # 每个线程负责部分 kv positions
    # ----------------------------------------------------------
    # Shared memory for scores and output accumulator
    scores = cute.SharedMemory(float, shape=(seq_len,))
    output = cute.SharedMemory(float, shape=(HEAD_DIM,))

    # 初始化 output
    for d in cute.thread_partition(range(HEAD_DIM), tid, NUM_THREADS):
        output[d] = 0.0

    # 计算 attention scores: 每个线程处理若干个 kv positions
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        k_vec = K_bh[j]       # (d,)
        dot = float(0.0)
        for dd in range(HEAD_DIM):
            dot += q_vec[dd] * k_vec[dd]
        scores[j] = dot * scale

    cute.syncthreads()

    # ----------------------------------------------------------
    # Step 2: Softmax over scores
    # ----------------------------------------------------------
    # 2a: find max (reduction)
    local_max = float('-inf')
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        local_max = max(local_max, scores[j])

    # Block-wide max reduction (简单用 shared memory)
    smem_reduce = cute.SharedMemory(float, shape=(NUM_THREADS,))
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

    # 2b: compute exp and sum
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

    # 2c: normalize
    for j in cute.thread_partition(range(seq_len), tid, NUM_THREADS):
        scores[j] /= global_sum

    cute.syncthreads()

    # ----------------------------------------------------------
    # Step 3: 计算 output = scores @ V
    # 每个线程负责 output 的部分维度
    # ----------------------------------------------------------
    for d in cute.thread_partition(range(HEAD_DIM), tid, NUM_THREADS):
        acc = float(0.0)
        for j in range(seq_len):
            acc += scores[j] * V_bh[j][d]
        O_bh[query_idx][d] = acc


# ============================================================
# Host 端启动
# ============================================================
def naive_attention(Q, K, V):
    """
    Q, K, V: torch.Tensor of shape (B, H, N, d), float32, on CUDA
    Returns: O of same shape
    """
    B, H, N, d = Q.shape
    assert d == HEAD_DIM

    # reshape to (B*H, N, d)
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.zeros_like(Q_flat)

    scale = 1.0 / (d ** 0.5)
    BH = B * H

    # CuTe tensor wrappers
    Q_cute = cute.from_dlpack(Q_flat)
    K_cute = cute.from_dlpack(K_flat)
    V_cute = cute.from_dlpack(V_flat)
    O_cute = cute.from_dlpack(O_flat)

    # Launch
    grid = (N, BH)
    block = (NUM_THREADS,)

    naive_attention_kernel[grid, block](
        Q_cute, K_cute, V_cute, O_cute,
        scale, N
    )

    return O_flat.reshape(B, H, N, d)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 128, HEAD_DIM
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)

    # our kernel
    O_ours = naive_attention(Q, K, V)

    # reference
    attn_weights = torch.softmax(
        Q @ K.transpose(-2, -1) / (d ** 0.5), dim=-1
    )
    O_ref = attn_weights @ V

    # compare
    max_diff = (O_ours - O_ref).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Too large diff: {max_diff}"
    print("PASSED!")
