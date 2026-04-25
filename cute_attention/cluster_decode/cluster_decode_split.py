from __future__ import annotations

from .common import ClusterDecodeConfig, HAS_CUTE, cutlass, cute, from_dlpack, require_torch, validate_decode_qkv


_CLUSTER_DECODE_SPLIT_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_cluster_decode_split_host(seq_len: int, head_dim: int, num_threads: int, cluster_size: int):
        kv_per_cta = (seq_len + cluster_size - 1) // cluster_size
        cluster_shape = (cluster_size, 1, 1)

        @cute.kernel
        def decode_split_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            bh_idx = bidx // cluster_size

            kv_start = cta_rank * kv_per_cta
            kv_stop = kv_start + kv_per_cta
            if kv_stop > seq_len:
                kv_stop = seq_len

            smem = cutlass.utils.SmemAllocator()
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

            # First milestone: each CTA owns a disjoint KV slice and computes
            # local metadata. The next iteration will exchange these values
            # across the cluster and reduce O to the leader CTA.
            local_max = -cutlass.Float32.inf
            for kv_idx in range(kv_start, kv_stop):
                if (kv_idx - kv_start) % num_threads == tidx:
                    score = 0.0
                    for d_idx in range(head_dim):
                        score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                    score *= softmax_scale
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

            slice_max = reduce[0]
            local_sum = 0.0
            for kv_idx in range(kv_start, kv_stop):
                if (kv_idx - kv_start) % num_threads == tidx:
                    score = 0.0
                    for d_idx in range(head_dim):
                        score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                    local_sum += cute.math.exp(score * softmax_scale - slice_max)

            reduce[tidx] = local_sum
            cute.arch.barrier()

            stride = num_threads // 2
            while stride > 0:
                if tidx < stride:
                    reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                cute.arch.barrier()
                stride //= 2

            # Keep the split skeleton correctness-preserving until the cluster
            # reduce lands: only the leader CTA computes the full decode row.
            if cta_rank == 0:
                full_max = -cutlass.Float32.inf
                for kv_idx in range(seq_len):
                    if kv_idx % num_threads == tidx:
                        score = 0.0
                        for d_idx in range(head_dim):
                            score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                        score *= softmax_scale
                        full_max = score if full_max < score else full_max

                reduce[tidx] = full_max
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
                full_sum = 0.0
                for kv_idx in range(seq_len):
                    if kv_idx % num_threads == tidx:
                        score = 0.0
                        for d_idx in range(head_dim):
                            score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                        full_sum += cute.math.exp(score * softmax_scale - row_max)

                reduce[tidx] = full_sum
                cute.arch.barrier()

                stride = num_threads // 2
                while stride > 0:
                    if tidx < stride:
                        reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                    cute.arch.barrier()
                    stride //= 2

                row_sum = reduce[0]
                inv_sum = 1.0 / row_sum

                for d_idx in range(head_dim):
                    if d_idx % num_threads == tidx:
                        acc = 0.0
                        for kv_idx in range(seq_len):
                            score = 0.0
                            for qk_d_idx in range(head_dim):
                                score += q[bh_idx, 0, qk_d_idx] * k[bh_idx, kv_idx, qk_d_idx]
                            prob = cute.math.exp(score * softmax_scale - row_max) * inv_sum
                            acc += prob * v[bh_idx, kv_idx, d_idx]
                        o[bh_idx, 0, d_idx] = acc.to(o.element_type)

        @cute.jit
        def cluster_decode_split_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            decode_split_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=(batch_heads * cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return cluster_decode_split_forward_host


def cluster_decode_split_forward(q, k, v, config: ClusterDecodeConfig | None = None):
    require_torch()
    config = config or ClusterDecodeConfig()
    validate_decode_qkv(q, k, v, config)

    if not HAS_CUTE:
        raise RuntimeError("cluster_decode_split requires cutlass.cute.")

    batch, heads, _, head_dim = q.shape
    seq_len = k.shape[2]
    scale = config.resolve_scale(head_dim)

    q_flat = q.reshape(batch * heads, 1, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = q_flat.new_empty(q_flat.shape)

    q_cute = from_dlpack(q_flat, assumed_align=16).mark_layout_dynamic()
    k_cute = from_dlpack(k_flat, assumed_align=16).mark_layout_dynamic()
    v_cute = from_dlpack(v_flat, assumed_align=16).mark_layout_dynamic()
    o_cute = from_dlpack(o_flat, assumed_align=16).mark_layout_dynamic()

    cache_key = (
        tuple(q_flat.shape),
        tuple(k_flat.shape),
        str(q_flat.dtype),
        config.num_threads,
        config.cluster_size,
    )
    compiled = _cluster_decode_split_compile(
        cache_key,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        scale,
        config.num_threads,
        config.cluster_size,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, scale)

    return o_flat.reshape(batch, heads, 1, head_dim)


def _cluster_decode_split_compile(
    cache_key,
    q_cute,
    k_cute,
    v_cute,
    o_cute,
    scale,
    num_threads,
    cluster_size,
):
    compiled = _CLUSTER_DECODE_SPLIT_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[1][1]
        head_dim = cache_key[0][2]
        host = _make_cluster_decode_split_host(
            seq_len=seq_len,
            head_dim=head_dim,
            num_threads=num_threads,
            cluster_size=cluster_size,
        )
        compiled = cute.compile(
            host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            scale,
        )
        _CLUSTER_DECODE_SPLIT_COMPILED_CACHE[cache_key] = compiled
    return compiled
