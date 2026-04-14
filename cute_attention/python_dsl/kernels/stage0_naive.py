from __future__ import annotations

import math

from .common import (
    AttentionConfig,
    HAS_CUTE,
    cutlass,
    cute,
    from_dlpack,
    require_torch,
    validate_qkv,
)


MAX_SEQ_LEN_FOR_STAGE0_CUTE = 2048
_STAGE0_COMPILED_CACHE = {}


if HAS_CUTE:
    @cute.kernel
    def naive_causal_attention_kernel(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        softmax_scale: cutlass.Float32,
        log2e: float,
        num_threads: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_idx, bh_idx, _ = cute.arch.block_idx()
        seq_len = cute.size(q.shape, mode=[1])
        head_dim = cute.size(q.shape, mode=[2])

        smem = cutlass.utils.SmemAllocator()
        scores_ptr = smem.allocate(cutlass.Float32, seq_len)
        reduce_ptr = smem.allocate(cutlass.Float32, num_threads)
        scores = cute.make_tensor(scores_ptr, cute.make_layout((seq_len,)))
        reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

        local_max = -cutlass.Float32.inf
        for kv_idx in range(tidx, seq_len, num_threads):
            if kv_idx > query_idx:
                scores[kv_idx] = -cutlass.Float32.inf
                continue

            score = 0.0
            for d_idx in range(head_dim):
                score += q[bh_idx, query_idx, d_idx] * k[bh_idx, kv_idx, d_idx]
            score *= softmax_scale
            scores[kv_idx] = score
            local_max = score if local_max < score else local_max

        reduce[tidx] = local_max
        cute.arch.barrier()

        stride = num_threads // 2
        while stride > 0:
            if tidx < stride:
                rhs = reduce[tidx + stride]
                lhs = reduce[tidx]
                reduce[tidx] = rhs if lhs < rhs else lhs
            cute.arch.barrier()
            stride //= 2

        row_max = reduce[0]
        local_sum = 0.0
        for kv_idx in range(tidx, seq_len, num_threads):
            prob = 0.0
            if kv_idx <= query_idx:
                prob = cute.math.exp2((scores[kv_idx] - row_max) * log2e, fastmath=True)
            scores[kv_idx] = prob
            local_sum += prob

        reduce[tidx] = local_sum
        cute.arch.barrier()

        stride = num_threads // 2
        while stride > 0:
            if tidx < stride:
                reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
            cute.arch.barrier()
            stride //= 2

        row_sum = reduce[0]
        inv_sum = cute.arch.rcp_approx(row_sum if row_sum != 0.0 else 1.0)

        for d_idx in range(tidx, head_dim, num_threads):
            acc = 0.0
            for kv_idx in range(query_idx + 1):
                acc += scores[kv_idx] * inv_sum * v[bh_idx, kv_idx, d_idx]
            o[bh_idx, query_idx, d_idx] = acc


    @cute.jit
    def stage0_forward_host(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        softmax_scale: cutlass.Float32,
        num_threads: cutlass.Constexpr[int],
    ):
        seq_len = cute.size(q.shape, mode=[1])
        batch_heads = cute.size(q.shape, mode=[0])
        naive_causal_attention_kernel(
            q,
            k,
            v,
            o,
            softmax_scale,
            math.log2(math.e),
            num_threads,
        ).launch(
            grid=(seq_len, batch_heads, 1),
            block=(num_threads, 1, 1),
        )

def stage0_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage0 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage0 requires cutlass.cute. No PyTorch fallback is available.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE0_CUTE:
        raise ValueError(
            f"stage0 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE0_CUTE}, got {seq_len}."
        )

    scale = config.resolve_scale(head_dim)
    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = q_flat.new_zeros(q_flat.shape)

    q_cute = from_dlpack(q_flat, assumed_align=16).mark_layout_dynamic()
    k_cute = from_dlpack(k_flat, assumed_align=16).mark_layout_dynamic()
    v_cute = from_dlpack(v_flat, assumed_align=16).mark_layout_dynamic()
    o_cute = from_dlpack(o_flat, assumed_align=16).mark_layout_dynamic()

    cache_key = (tuple(q_flat.shape), str(q_flat.dtype), config.num_threads)
    compiled = _stage0_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, config.num_threads)
    compiled(q_cute, k_cute, v_cute, o_cute, scale, config.num_threads)

    return o_flat.reshape(batch, heads, seq_len, head_dim)


def _stage0_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, num_threads):
    compiled = _STAGE0_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            stage0_forward_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            scale,
            num_threads,
        )
        _STAGE0_COMPILED_CACHE[cache_key] = compiled
    return compiled
