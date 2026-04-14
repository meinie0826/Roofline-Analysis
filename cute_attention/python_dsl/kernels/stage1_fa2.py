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
    def _make_stage1_host(seq_len: int, head_dim: int, block_m: int, block_n: int, num_threads: int):
        @cute.kernel
        def fa2_causal_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            m_block, bh_idx, _ = cute.arch.block_idx()
            m_start = m_block * block_m

            smem = cutlass.utils.SmemAllocator()
            q_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * head_dim)
            score_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * block_n)
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * head_dim)
            row_m_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)
            row_l_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)
            q_tile = cute.make_tensor(q_ptr, cute.make_layout((block_m * head_dim,)))
            score_tile = cute.make_tensor(score_ptr, cute.make_layout((block_m * block_n,)))
            acc_tile = cute.make_tensor(acc_ptr, cute.make_layout((block_m * head_dim,)))
            row_m = cute.make_tensor(row_m_ptr, cute.make_layout((block_m,)))
            row_l = cute.make_tensor(row_l_ptr, cute.make_layout((block_m,)))

            for lin in range(tidx, block_m * head_dim, num_threads):
                row = lin // head_dim
                d_idx = lin - row * head_dim
                q_idx = m_start + row
                q_val = cutlass.Float32(0.0)
                if q_idx < seq_len:
                    q_val = q[bh_idx, q_idx, d_idx].to(cutlass.Float32)
                q_tile[lin] = q_val
                acc_tile[lin] = 0.0

            for row in range(tidx, block_m, num_threads):
                q_idx = m_start + row
                if q_idx < seq_len:
                    row_m[row] = -1e20
                    row_l[row] = 0.0
                else:
                    row_m[row] = 0.0
                    row_l[row] = 1.0
            cute.arch.barrier()

            for n_start in range(0, seq_len, block_n):
                for lin in range(tidx, block_m * block_n, num_threads):
                    row = lin // block_n
                    blk_j = lin - row * block_n
                    q_idx = m_start + row
                    kv_idx = n_start + blk_j
                    score = -cutlass.Float32.inf
                    if q_idx < seq_len and kv_idx < seq_len and kv_idx <= q_idx:
                        dot = cutlass.Float32(0.0)
                        q_base = row * head_dim
                        for d_idx in range(head_dim):
                            dot += q_tile[q_base + d_idx] * k[bh_idx, kv_idx, d_idx].to(cutlass.Float32)
                        score = dot * softmax_scale
                    score_tile[lin] = score
                cute.arch.barrier()

                for row in range(tidx, block_m, num_threads):
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        row_base = row * block_n
                        block_max = -cutlass.Float32.inf
                        for blk_j in range(block_n):
                            s = score_tile[row_base + blk_j]
                            block_max = s if block_max < s else block_max

                        m_prev = row_m[row]
                        m_new = m_prev if m_prev > block_max else block_max
                        alpha = cute.math.exp(m_prev - m_new)

                        block_sum = 0.0
                        for blk_j in range(block_n):
                            kv_idx = n_start + blk_j
                            p = 0.0
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                p = cute.math.exp(score_tile[row_base + blk_j] - m_new)
                            score_tile[row_base + blk_j] = p
                            block_sum += p

                        row_m[row] = m_new
                        row_l[row] = row_l[row] * alpha + block_sum

                        acc_base = row * head_dim
                        for d_idx in range(head_dim):
                            acc_tile[acc_base + d_idx] = acc_tile[acc_base + d_idx] * alpha
                cute.arch.barrier()

                for lin in range(tidx, block_m * head_dim, num_threads):
                    row = lin // head_dim
                    d_idx = lin - row * head_dim
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        acc = acc_tile[lin]
                        row_base = row * block_n
                        for blk_j in range(block_n):
                            kv_idx = n_start + blk_j
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                acc += score_tile[row_base + blk_j] * v[bh_idx, kv_idx, d_idx].to(cutlass.Float32)
                        acc_tile[lin] = acc
                cute.arch.barrier()

            for lin in range(tidx, block_m * head_dim, num_threads):
                row = lin // head_dim
                d_idx = lin - row * head_dim
                q_idx = m_start + row
                if q_idx < seq_len:
                    o[bh_idx, q_idx, d_idx] = (acc_tile[lin] / row_l[row]).to(o.element_type)

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
                grid=((seq_len + block_m - 1) // block_m, batch_heads, 1),
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
    if config.block_m <= 0:
        raise ValueError("block_m must be > 0.")
    if config.block_n <= 0:
        raise ValueError("block_n must be > 0.")
    if config.num_threads <= 0 or config.num_threads % 32 != 0:
        raise ValueError("num_threads must be > 0 and divisible by 32.")
    smem_bytes = 4 * (
        config.block_m * head_dim * 2
        + config.block_m * config.block_n
        + 2 * config.block_m
    )
    if smem_bytes > 96 * 1024:
        raise ValueError(
            f"stage1 shared memory footprint too large ({smem_bytes} bytes). "
            "Reduce block_m/block_n/head_dim."
        )

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
        config.block_m,
        config.block_n,
        config.num_threads,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, softmax_scale)
    return o_flat.reshape(batch, heads, seq_len, head_dim)


def _stage1_compile(cache_key, q_cute, k_cute, v_cute, o_cute, softmax_scale, block_m, block_n, num_threads):
    compiled = _STAGE1_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage1_host = _make_stage1_host(
            seq_len=seq_len,
            head_dim=head_dim,
            block_m=block_m,
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
