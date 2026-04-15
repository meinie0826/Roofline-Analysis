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


MAX_SEQ_LEN_FOR_STAGE2_CUTE = 4096
_STAGE2_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_stage2_host(seq_len: int, head_dim: int, block_m: int, num_threads: int):
        """
        stage2: blocked along the head_dim (column) dimension.

        Each CTA handles one query token (same grid as stage0/stage3).
        head_dim is split into tiles of size block_m.  For each kv position
        the dot product Q·K is accumulated across block_m tiles; a
        num_threads-wide tree reduction sums the per-thread partial dot
        products within every tile.  After all kv scores are computed the
        kernel runs the usual online softmax and accumulates V with the same
        head_dim tiling.
        """
        @cute.kernel
        def col_blocked_causal_attention_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            query_idx, bh_idx, _ = cute.arch.block_idx()

            smem = cutlass.utils.SmemAllocator()
            # scores for all kv positions
            scores_ptr = smem.allocate_array(cutlass.Float32, num_elems=seq_len)
            # thread-reduction scratch buffer
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            # output accumulator
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)

            scores  = cute.make_tensor(scores_ptr, cute.make_layout((seq_len,)))
            reduce  = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))
            acc_vec = cute.make_tensor(acc_ptr,    cute.make_layout((head_dim,)))

            # ---------- initialise acc_vec ----------
            for d_idx in range(head_dim):
                if d_idx % num_threads == tidx:
                    acc_vec[d_idx] = 0.0

            # ---------- compute Q·K scores ----------
            # For each kv position each thread accumulates its share of the
            # head_dim dot product, then a tree-reduce sums across threads.
            for kv_idx in range(seq_len):
                if kv_idx <= query_idx:
                    # Each thread computes a partial dot product over the
                    # head_dim elements it owns (every num_threads-th element).
                    local_dot = 0.0
                    for d_idx in range(tidx, head_dim, num_threads):
                        local_dot += q[bh_idx, query_idx, d_idx] * k[bh_idx, kv_idx, d_idx]

                    # Reduce partial dot products across threads.
                    reduce[tidx] = local_dot
                    cute.arch.barrier()

                    stride = num_threads // 2
                    while stride > 0:
                        if tidx < stride:
                            reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                        cute.arch.barrier()
                        stride //= 2

                    if tidx == 0:
                        scores[kv_idx] = reduce[0] * softmax_scale
                    cute.arch.barrier()
                else:
                    if tidx == 0:
                        scores[kv_idx] = -cutlass.Float32.inf
                    cute.arch.barrier()

            # ---------- row max ----------
            local_max = -cutlass.Float32.inf
            for kv_idx in range(seq_len):
                if kv_idx % num_threads == tidx:
                    s = scores[kv_idx]
                    local_max = s if local_max < s else local_max

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

            # ---------- softmax normaliser ----------
            local_sum = 0.0
            for kv_idx in range(seq_len):
                if kv_idx % num_threads == tidx:
                    prob = 0.0
                    if kv_idx <= query_idx:
                        prob = cute.math.exp(scores[kv_idx] - row_max)
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
            inv_sum = 1.0 / row_sum

            # ---------- weighted V accumulation (col-blocked) ----------
            # Outer loop over head_dim tiles of size block_m; inner loop
            # over kv positions.  Each thread owns the d indices that
            # satisfy  d_start <= d_idx < d_start+block_m  AND
            #          d_idx % num_threads == tidx  (within the tile).
            for d_start in range(0, head_dim, block_m):
                for d_idx in range(d_start, d_start + block_m):
                    if d_idx % num_threads == tidx and d_idx < head_dim:
                        acc = 0.0
                        for kv_idx in range(seq_len):
                            if kv_idx <= query_idx:
                                acc += scores[kv_idx] * v[bh_idx, kv_idx, d_idx]
                        acc_vec[d_idx] = acc

            # ---------- write output ----------
            for d_idx in range(head_dim):
                if d_idx % num_threads == tidx:
                    o[bh_idx, query_idx, d_idx] = (acc_vec[d_idx] * inv_sum).to(o.element_type)

        @cute.jit
        def stage2_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            col_blocked_causal_attention_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=(seq_len, batch_heads, 1),
                block=(num_threads, 1, 1),
            )

        return stage2_forward_host


def stage2_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage2 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage2 requires cutlass.cute.")
    if q.dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(f"stage2 currently only supports fp16/bf16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE2_CUTE:
        raise ValueError(
            f"stage2 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE2_CUTE}, got {seq_len}."
        )
    if head_dim > 256:
        raise ValueError(f"stage2 currently supports head_dim <= 256, got {head_dim}.")
    if config.block_m <= 0:
        raise ValueError("block_m must be > 0.")
    if head_dim % config.block_m != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by block_m ({config.block_m}).")
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
        config.block_m,
        config.num_threads,
    )
    compiled = _stage2_compile(
        cache_key,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        softmax_scale,
        config.block_m,
        config.num_threads,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, softmax_scale)
    return o_flat.reshape(batch, heads, seq_len, head_dim)


def _stage2_compile(cache_key, q_cute, k_cute, v_cute, o_cute, softmax_scale, block_m, num_threads):
    compiled = _STAGE2_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len  = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage2_host = _make_stage2_host(
            seq_len=seq_len,
            head_dim=head_dim,
            block_m=block_m,
            num_threads=num_threads,
        )
        compiled = cute.compile(
            stage2_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            softmax_scale,
        )
        _STAGE2_COMPILED_CACHE[cache_key] = compiled
    return compiled
