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


MAX_SEQ_LEN_FOR_STAGE9_CUTE = 4096
_STAGE9_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_stage9_host(seq_len: int, head_dim: int, block_m: int, block_n: int, num_threads: int):
        threads_per_row = num_threads // block_m

        @cute.kernel
        def stage9_threadgroup_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            m_block, bh_idx, _ = cute.arch.block_idx()
            m_start = m_block * block_m

            row = tidx // threads_per_row
            lane = tidx - row * threads_per_row

            smem = cutlass.utils.SmemAllocator()
            q_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_m * head_dim)
            k_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_n * head_dim)
            v_ptr = smem.allocate_array(cutlass.Float16, num_elems=block_n * head_dim)
            score_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * block_n)
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m * head_dim)
            row_m_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)
            row_l_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            alpha_ptr = smem.allocate_array(cutlass.Float32, num_elems=block_m)

            q_tile = cute.make_tensor(q_ptr, cute.make_layout((block_m * head_dim,)))
            k_tile = cute.make_tensor(k_ptr, cute.make_layout((block_n * head_dim,)))
            v_tile = cute.make_tensor(v_ptr, cute.make_layout((block_n * head_dim,)))
            score_tile = cute.make_tensor(score_ptr, cute.make_layout((block_m * block_n,)))
            acc_tile = cute.make_tensor(acc_ptr, cute.make_layout((block_m * head_dim,)))
            row_m = cute.make_tensor(row_m_ptr, cute.make_layout((block_m,)))
            row_l = cute.make_tensor(row_l_ptr, cute.make_layout((block_m,)))
            reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))
            alpha = cute.make_tensor(alpha_ptr, cute.make_layout((block_m,)))

            for lin in range(tidx, block_m * head_dim, num_threads):
                r = lin // head_dim
                d_idx = lin - r * head_dim
                q_idx = m_start + r
                q_val = cutlass.Float16(0.0)
                if q_idx < seq_len:
                    q_val = q[bh_idx, q_idx, d_idx]
                q_tile[lin] = q_val
                acc_tile[lin] = 0.0

            for r in range(tidx, block_m, num_threads):
                q_idx = m_start + r
                if q_idx < seq_len:
                    row_m[r] = -1e20
                    row_l[r] = 0.0
                else:
                    row_m[r] = 0.0
                    row_l[r] = 1.0
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

                if row < block_m:
                    q_idx = m_start + row
                    row_base = row * block_n
                    q_base = row * head_dim
                    local_max = -cutlass.Float32.inf

                    if q_idx < seq_len and n_start <= q_idx:
                        for blk_j in range(lane, block_n, threads_per_row):
                            kv_idx = n_start + blk_j
                            score = -cutlass.Float32.inf
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                k_base = blk_j * head_dim
                                dot = cutlass.Float32(0.0)
                                for d_idx in range(head_dim):
                                    dot += q_tile[q_base + d_idx].to(cutlass.Float32) * k_tile[k_base + d_idx].to(cutlass.Float32)
                                score = dot * softmax_scale
                                local_max = score if local_max < score else local_max
                            score_tile[row_base + blk_j] = score

                    reduce[tidx] = local_max
                cute.arch.barrier()

                stride = threads_per_row // 2
                while stride > 0:
                    if row < block_m and lane < stride:
                        rhs = reduce[tidx + stride]
                        lhs = reduce[tidx]
                        reduce[tidx] = rhs if lhs < rhs else lhs
                    cute.arch.barrier()
                    stride //= 2

                if row < block_m and lane == 0:
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        block_max = reduce[tidx]
                        m_prev = row_m[row]
                        m_new = m_prev if m_prev > block_max else block_max
                        row_m[row] = m_new
                        alpha[row] = cute.math.exp(m_prev - m_new)
                    else:
                        alpha[row] = 1.0
                cute.arch.barrier()

                if row < block_m:
                    q_idx = m_start + row
                    row_base = row * block_n
                    local_sum = 0.0
                    m_new = row_m[row]
                    if q_idx < seq_len and n_start <= q_idx:
                        for blk_j in range(lane, block_n, threads_per_row):
                            kv_idx = n_start + blk_j
                            p = 0.0
                            if kv_idx < seq_len and kv_idx <= q_idx:
                                p = cute.math.exp(score_tile[row_base + blk_j] - m_new)
                            score_tile[row_base + blk_j] = p
                            local_sum += p
                    reduce[tidx] = local_sum
                cute.arch.barrier()

                stride = threads_per_row // 2
                while stride > 0:
                    if row < block_m and lane < stride:
                        reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                    cute.arch.barrier()
                    stride //= 2

                if row < block_m and lane == 0:
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        row_l[row] = row_l[row] * alpha[row] + reduce[tidx]
                cute.arch.barrier()

                if row < block_m:
                    q_idx = m_start + row
                    if q_idx < seq_len and n_start <= q_idx:
                        q_base = row * head_dim
                        row_base = row * block_n
                        a = alpha[row]
                        for d_idx in range(lane, head_dim, threads_per_row):
                            acc = acc_tile[q_base + d_idx] * a
                            for blk_j in range(block_n):
                                kv_idx = n_start + blk_j
                                if kv_idx < seq_len and kv_idx <= q_idx:
                                    v_base = blk_j * head_dim
                                    acc += score_tile[row_base + blk_j] * v_tile[v_base + d_idx].to(cutlass.Float32)
                            acc_tile[q_base + d_idx] = acc
                cute.arch.barrier()

            for lin in range(tidx, block_m * head_dim, num_threads):
                r = lin // head_dim
                d_idx = lin - r * head_dim
                q_idx = m_start + r
                if q_idx < seq_len:
                    o[bh_idx, q_idx, d_idx] = (acc_tile[lin] / row_l[r]).to(o.element_type)

        @cute.jit
        def stage9_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            stage9_threadgroup_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=((seq_len + block_m - 1) // block_m, batch_heads, 1),
                block=(num_threads, 1, 1),
            )

        return stage9_forward_host


def stage9_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage9 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage9 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage9 currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE9_CUTE:
        raise ValueError(
            f"stage9 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE9_CUTE}, got {seq_len}."
        )
    if head_dim > 256:
        raise ValueError(f"stage9 currently supports head_dim <= 256, got {head_dim}.")
    if config.block_m <= 0:
        raise ValueError("block_m must be > 0.")
    if config.block_n <= 0:
        raise ValueError("block_n must be > 0.")
    if config.num_threads <= 0 or config.num_threads % 32 != 0:
        raise ValueError("num_threads must be > 0 and divisible by 32.")
    if config.num_threads % config.block_m != 0:
        raise ValueError("stage9 requires num_threads % block_m == 0.")

    threads_per_row = config.num_threads // config.block_m
    if threads_per_row < 1 or (threads_per_row & (threads_per_row - 1)) != 0:
        raise ValueError("stage9 requires threads_per_row=num_threads/block_m to be a power of two.")

    smem_bytes = (
        2 * (config.block_m * head_dim + 2 * config.block_n * head_dim)
        + 4 * (config.block_m * config.block_n + config.block_m * head_dim + 2 * config.block_m + config.num_threads)
    )
    if smem_bytes > 96 * 1024:
        raise ValueError(
            f"stage9 shared memory footprint too large ({smem_bytes} bytes). "
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
    compiled = _stage9_compile(
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


def _stage9_compile(cache_key, q_cute, k_cute, v_cute, o_cute, softmax_scale, block_m, block_n, num_threads):
    compiled = _STAGE9_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage9_host = _make_stage9_host(
            seq_len=seq_len,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            num_threads=num_threads,
        )
        compiled = cute.compile(
            stage9_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            softmax_scale,
        )
        _STAGE9_COMPILED_CACHE[cache_key] = compiled
    return compiled
