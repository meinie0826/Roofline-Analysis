from __future__ import annotations

from .common import (
    AttentionConfig,
    HAS_CUTE,
    cutlass,
    cute,
    from_dlpack,
    require_torch,
    torch,
    validate_qkv,
)


MAX_SEQ_LEN_FOR_STAGE1_CUTE = 4096
_STAGE1_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_stage1_host(seq_len: int, head_dim: int, block_n: int, num_threads: int):
        @cute.kernel
        def fa2_causal_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            query_idx, bh_idx, _ = cute.arch.block_idx()

            smem = cutlass.utils.SmemAllocator()
            score_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_n)
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)
            score_blk = cute.make_tensor(score_ptr, cute.make_layout((block_n,)))
            reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))
            acc_vec = cute.make_tensor(acc_ptr, cute.make_layout((head_dim,)))

            for d_idx in cutlass.range_constexpr(head_dim):
                if d_idx % num_threads == tidx:
                    acc_vec[d_idx] = 0.0

            row_m = -1e20
            row_l = 0.0

            for blk_start in cutlass.range_constexpr(0, seq_len, block_n):
                if blk_start <= query_idx:
                    local_max = -cutlass.Float32.inf

                    for blk_j in cutlass.range_constexpr(block_n):
                        if blk_j % num_threads == tidx:
                            kv_idx = blk_start + blk_j
                            score = -cutlass.Float32.inf
                            if kv_idx <= query_idx and kv_idx < seq_len:
                                score = 0.0
                                for d_idx in cutlass.range_constexpr(head_dim):
                                    score += q[bh_idx, query_idx, d_idx] * k[bh_idx, kv_idx, d_idx]
                                score *= softmax_scale
                                local_max = score if local_max < score else local_max
                            score_blk[blk_j] = score

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

                    blk_max = reduce[0]
                    row_m_new = row_m if row_m > blk_max else blk_max
                    alpha = cute.math.exp(row_m - row_m_new)

                    local_sum = 0.0
                    for blk_j in cutlass.range_constexpr(block_n):
                        if blk_j % num_threads == tidx:
                            kv_idx = blk_start + blk_j
                            prob = 0.0
                            if kv_idx <= query_idx and kv_idx < seq_len:
                                prob = cute.math.exp(score_blk[blk_j] - row_m_new)
                            score_blk[blk_j] = prob
                            local_sum += prob

                    reduce[tidx] = local_sum
                    cute.arch.barrier()

                    stride = num_threads // 2
                    while stride > 0:
                        if tidx < stride:
                            reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                        cute.arch.barrier()
                        stride //= 2

                    blk_sum = reduce[0]
                    row_l_new = row_l * alpha + blk_sum

                    for d_idx in cutlass.range_constexpr(head_dim):
                        if d_idx % num_threads == tidx:
                            acc = acc_vec[d_idx] * alpha
                            for blk_j in cutlass.range_constexpr(block_n):
                                kv_idx = blk_start + blk_j
                                if kv_idx <= query_idx and kv_idx < seq_len:
                                    acc += score_blk[blk_j] * v[bh_idx, kv_idx, d_idx]
                            acc_vec[d_idx] = acc

                    row_m = row_m_new
                    row_l = row_l_new

            for d_idx in cutlass.range_constexpr(head_dim):
                if d_idx % num_threads == tidx:
                    o[bh_idx, query_idx, d_idx] = (acc_vec[d_idx] / row_l).to(o.element_type)

        @cute.jit
        def stage1_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            fa2_causal_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=(seq_len, batch_heads, 1),
                block=(num_threads, 1, 1),
            )

        return stage1_forward_host


def stage1_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage1 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage1 requires cutlass.cute.")
    if q.dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(f"stage1 currently only supports fp16/bf16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE1_CUTE:
        raise ValueError(
            f"stage1 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE1_CUTE}, got {seq_len}."
        )
    if head_dim > 256:
        raise ValueError(f"stage1 currently supports head_dim <= 256, got {head_dim}.")
    if config.block_n <= 0:
        raise ValueError("block_n must be > 0.")
    if config.num_threads <= 0 or config.num_threads % 32 != 0:
        raise ValueError("num_threads must be > 0 and divisible by 32.")

    softmax_scale = config.resolve_scale(head_dim)
    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = q_flat.new_zeros(q_flat.shape)

    q_cute = from_dlpack(q_flat, assumed_align=16).mark_layout_dynamic()
    k_cute = from_dlpack(k_flat, assumed_align=16).mark_layout_dynamic()
    v_cute = from_dlpack(v_flat, assumed_align=16).mark_layout_dynamic()
    o_cute = from_dlpack(o_flat, assumed_align=16).mark_layout_dynamic()

    cache_key = (
        tuple(q_flat.shape),
        str(q_flat.dtype),
        config.block_n,
        config.num_threads,
    )
    compiled = _stage1_compile(
        cache_key,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        softmax_scale,
        config.block_n,
        config.num_threads,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, softmax_scale)
    return o_flat.reshape(batch, heads, seq_len, head_dim)


def _stage1_compile(cache_key, q_cute, k_cute, v_cute, o_cute, softmax_scale, block_n, num_threads):
    compiled = _STAGE1_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage1_host = _make_stage1_host(
            seq_len=seq_len,
            head_dim=head_dim,
            block_n=block_n,
            num_threads=num_threads,
        )
        compiled = cute.compile(
            stage1_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            softmax_scale,
        )
        _STAGE1_COMPILED_CACHE[cache_key] = compiled
    return compiled
