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


MAX_SEQ_LEN_FOR_STAGE8_CUTE = 4096
_STAGE8_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_stage8_host(seq_len: int, head_dim: int, block_m: int, block_n: int, num_threads: int):
        @cute.kernel
        def stage8_no_score_tile_kernel(
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
            q_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_m * head_dim)
            k_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_n * head_dim)
            v_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_n * head_dim)
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * head_dim)
            row_m_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)
            row_l_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)

            q_tile = cute.make_tensor(q_ptr, cute.make_layout((block_m * head_dim,)))
            k_tile = cute.make_tensor(k_ptr, cute.make_layout((block_n * head_dim,)))
            v_tile = cute.make_tensor(v_ptr, cute.make_layout((block_n * head_dim,)))
            acc_tile = cute.make_tensor(acc_ptr, cute.make_layout((block_m * head_dim,)))
            row_m = cute.make_tensor(row_m_ptr, cute.make_layout((block_m,)))
            row_l = cute.make_tensor(row_l_ptr, cute.make_layout((block_m,)))

            for lin in range(tidx, block_m * head_dim, num_threads):
                row = lin // head_dim
                d_idx = lin - row * head_dim
                q_idx = m_start + row
                q_val = cutlass.Float16(0.0)
                if q_idx < seq_len:
                    q_val = q[bh_idx, q_idx, d_idx]
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
                for lin in range(tidx, block_n * head_dim, num_threads):
                    blk_j = lin // head_dim
                    d_idx = lin - blk_j * head_dim
                    kv_idx = n_start + blk_j
                    k_val = cutlass.Float16(0.0)
                    v_val = cutlass.Float16(0.0)
                    if kv_idx < seq_len:
                        k_val = k[bh_idx, kv_idx, d_idx]
                        v_val = v[bh_idx, kv_idx, d_idx]
                    k_tile[lin] = k_val
                    v_tile[lin] = v_val
                cute.arch.barrier()

                for row in range(tidx, block_m, num_threads):
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        block_max = -cutlass.Float32.inf
                        q_base = row * head_dim
                        for blk_j in range(block_n):
                            kv_idx = n_start + blk_j
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                k_base = blk_j * head_dim
                                dot = cutlass.Float32(0.0)
                                for d_idx in range(head_dim):
                                    dot += q_tile[q_base + d_idx].to(cutlass.Float32) * k_tile[k_base + d_idx].to(cutlass.Float32)
                                score = dot * softmax_scale
                                block_max = score if block_max < score else block_max

                        m_prev = row_m[row]
                        m_new = m_prev if m_prev > block_max else block_max
                        alpha = cute.math.exp(m_prev - m_new)
                        block_sum = 0.0

                        acc_base = row * head_dim
                        for d_idx in range(head_dim):
                            acc_tile[acc_base + d_idx] = acc_tile[acc_base + d_idx] * alpha

                        for blk_j in range(block_n):
                            kv_idx = n_start + blk_j
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                k_base = blk_j * head_dim
                                dot = cutlass.Float32(0.0)
                                for d_idx in range(head_dim):
                                    dot += q_tile[q_base + d_idx].to(cutlass.Float32) * k_tile[k_base + d_idx].to(cutlass.Float32)
                                score = dot * softmax_scale
                                p = cute.math.exp(score - m_new)
                                block_sum += p
                                v_base = blk_j * head_dim
                                for d_idx in range(head_dim):
                                    acc_tile[acc_base + d_idx] = acc_tile[acc_base + d_idx] + p * v_tile[v_base + d_idx].to(cutlass.Float32)

                        row_m[row] = m_new
                        row_l[row] = row_l[row] * alpha + block_sum
                cute.arch.barrier()

            for lin in range(tidx, block_m * head_dim, num_threads):
                row = lin // head_dim
                d_idx = lin - row * head_dim
                q_idx = m_start + row
                if q_idx < seq_len:
                    o[bh_idx, q_idx, d_idx] = (acc_tile[lin] / row_l[row]).to(o.element_type)

        @cute.jit
        def stage8_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            stage8_no_score_tile_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=((seq_len + block_m - 1) // block_m, batch_heads, 1),
                block=(num_threads, 1, 1),
            )

        return stage8_forward_host


def stage8_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage8 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage8 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage8 currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE8_CUTE:
        raise ValueError(
            f"stage8 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE8_CUTE}, got {seq_len}."
        )
    if head_dim > 256:
        raise ValueError(f"stage8 currently supports head_dim <= 256, got {head_dim}.")
    if config.block_m <= 0:
        raise ValueError("block_m must be > 0.")
    if config.block_n <= 0:
        raise ValueError("block_n must be > 0.")
    if config.num_threads <= 0 or config.num_threads % 32 != 0:
        raise ValueError("num_threads must be > 0 and divisible by 32.")

    smem_bytes = (
        2 * (config.block_m * head_dim + 2 * config.block_n * head_dim)
        + 4 * (config.block_m * head_dim + 2 * config.block_m)
    )
    if smem_bytes > 96 * 1024:
        raise ValueError(
            f"stage8 shared memory footprint too large ({smem_bytes} bytes). "
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
    compiled = _stage8_compile(
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


def _stage8_compile(cache_key, q_cute, k_cute, v_cute, o_cute, softmax_scale, block_m, block_n, num_threads):
    compiled = _STAGE8_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage8_host = _make_stage8_host(
            seq_len=seq_len,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            num_threads=num_threads,
        )
        compiled = cute.compile(
            stage8_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            softmax_scale,
        )
        _STAGE8_COMPILED_CACHE[cache_key] = compiled
    return compiled
