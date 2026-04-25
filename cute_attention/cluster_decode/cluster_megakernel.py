"""ClusterFusion-style decode megakernel – v2 (no DSMEM primitives).

Avoids ALL cute.arch.mapa() and cluster-scope atomic operations which trigger
MLIR/PTX ICE errors. Uses global scratch buffers for all cross-CTA communication
with cute.arch.cluster_arrive/cluster_wait as the only cluster primitive.

This baseline must pass correctness first; DSMEM st.async will be added later.
"""

from __future__ import annotations

from .common import (
    HAS_CUTE,
    MegakernelConfig,
    cutlass,
    cute,
    from_dlpack,
    require_torch,
    validate_megakernel_inputs,
)

_MEGAKERNEL_COMPILED_CACHE: dict = {}


if HAS_CUTE:

    def _make_cluster_megakernel_host(
        seq_len: int,
        config: MegakernelConfig,
    ):
        hidden_dim   = config.hidden_dim
        num_heads    = config.num_heads
        head_dim     = config.head_dim
        cluster_size = config.cluster_size
        num_threads  = config.num_threads
        dim_per_block = config.dim_per_block
        tile_attn    = config.tma_load_once // 2
        kv_per_cta   = ((seq_len + cluster_size - 1) // cluster_size + tile_attn - 1) & ~(tile_attn - 1)

        cluster_shape = (cluster_size, 1, 1)

        @cute.kernel
        def _megakernel(
            hidden:      cute.Tensor,
            w_qkv:       cute.Tensor,
            w_o:         cute.Tensor,
            k_cache:     cute.Tensor,
            v_cache:     cute.Tensor,
            rms_weight:  cute.Tensor,
            cos_rope:    cute.Tensor,
            sin_rope:    cute.Tensor,
            output:      cute.Tensor,
            k_out:       cute.Tensor,
            v_out:       cute.Tensor,
            # Global scratch buffers for cross-CTA communication
            scratch_l2:  cute.Tensor,   # (num_heads, cluster_size)     f32 – RMSNorm partial l2
            scratch_max: cute.Tensor,   # (num_heads, cluster_size)     f32 – attn partial max
            scratch_sum: cute.Tensor,   # (num_heads, cluster_size)     f32 – attn partial sum
            scratch_out: cute.Tensor,   # (num_heads, cluster_size, head_dim) f16 – attn partial O
            scratch_wo:  cute.Tensor,   # (num_heads, hidden_dim)        f16 – W_o per-head output
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank   = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            head_id    = bidx // cluster_size
            cta_block_start = cta_rank * dim_per_block

            eps = cutlass.Float32(1e-6)

            smem = cutlass.utils.SmemAllocator()

            # Intra-CTA reduction scratch
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce     = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

            # local_qkv: [Q | K | V] in fp16
            local_qkv_ptr = smem.allocate_array(cutlass.Float16, num_elems=3 * head_dim)
            local_qkv     = cute.make_tensor(local_qkv_ptr, cute.make_layout((3 * head_dim,)))

            # ============================================================ #
            # Stage 0 – RMSNorm                                            #
            # ============================================================ #
            local_l2 = cutlass.Float32(0.0)
            d = tidx
            while d < dim_per_block:
                val = hidden[0, cta_block_start + d].to(cutlass.Float32)
                local_l2 = local_l2 + val * val
                d = d + num_threads

            # Intra-CTA tree reduce
            reduce[tidx] = local_l2
            cute.arch.barrier()
            stride = num_threads // 2
            while stride > 0:
                if tidx < stride:
                    reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                cute.arch.barrier()
                stride = stride // 2
            local_l2 = reduce[0]

            # Write partial l2 to global scratch, then cluster sync
            if tidx == 0:
                scratch_l2[head_id, cta_rank] = local_l2

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # Leader CTA reads all partials and computes RMSNorm
            if cta_rank == 0:
                cluster_l2 = cutlass.Float32(0.0)
                for c in range(cluster_size):
                    cluster_l2 = cluster_l2 + scratch_l2[head_id, c].to(cutlass.Float32)
                rms_rcp = cute.math.rsqrt(cluster_l2 / cutlass.Float32(hidden_dim) + eps)

                # Broadcast rms_rcp to all CTAs via global scratch
                # (reuse scratch_l2[head_id, 0] as rms_rcp storage since we already read it)
                # Actually, simpler: write rms_rcp to a dedicated position
                # We'll just have every CTA compute it independently – they all have
                # the same partial_l2 values after cluster_wait, so they can independently
                # sum. But that requires each CTA to read all scratch_l2 values.
                # For now: every CTA independently sums scratch_l2 after cluster_wait.
                # This works because cluster_arrive/cluster_wait ensures all writes are visible.
                if tidx == 0:
                    scratch_l2[head_id, 0] = cluster_l2  # reuse slot 0 for total

            # All CTAs read the total l2
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            cluster_l2 = scratch_l2[head_id, 0].to(cutlass.Float32)
            rms_rcp    = cute.math.rsqrt(cluster_l2 / cutlass.Float32(hidden_dim) + eps)

            # ============================================================ #
            # Stage 1 – W_qkv GEMM (independent, no cluster reduce)       #
            # Each CTA reads full hidden_dim input and computes Q/K/V.     #
            # ============================================================ #
            for proj in range(3):
                if tidx < head_dim:
                    out_d      = tidx
                    global_row = head_id * head_dim + out_d + proj * hidden_dim
                    acc        = cutlass.Float32(0.0)
                    for col in range(hidden_dim):
                        x_val  = hidden[0, col].to(cutlass.Float32)
                        x_norm = x_val * rms_rcp * rms_weight[col].to(cutlass.Float32)
                        w_val  = w_qkv[global_row, col].to(cutlass.Float32)
                        acc    = acc + x_norm * w_val
                    local_qkv[proj * head_dim + out_d] = acc.to(cutlass.Float16)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 2 – RoPE                                               #
            # ============================================================ #
            if tidx < head_dim:
                q_val = local_qkv[tidx].to(cutlass.Float32)
                k_val = local_qkv[head_dim + tidx].to(cutlass.Float32)
                c     = cos_rope[tidx]
                s     = sin_rope[tidx]
                if tidx % 2 == 0:
                    q1 = local_qkv[tidx + 1].to(cutlass.Float32)
                    k1 = local_qkv[head_dim + tidx + 1].to(cutlass.Float32)
                    local_qkv[tidx]            = (q_val * c - q1 * s).to(cutlass.Float16)
                    local_qkv[head_dim + tidx] = (k_val * c - k1 * s).to(cutlass.Float16)
                else:
                    q1 = local_qkv[tidx - 1].to(cutlass.Float32)
                    k1 = local_qkv[head_dim + tidx - 1].to(cutlass.Float32)
                    local_qkv[tidx]            = (q_val * c + q1 * s).to(cutlass.Float16)
                    local_qkv[head_dim + tidx] = (k_val * c + k1 * s).to(cutlass.Float16)

            # Write K/V outputs (leader only)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if cta_rank == 0:
                if tidx < head_dim:
                    k_out[0, head_id, tidx] = local_qkv[head_dim + tidx]
                    v_out[0, head_id, tidx] = local_qkv[2 * head_dim + tidx]

            # ============================================================ #
            # Stage 3 – Flash-decoding attention (per-CTA KV slice)       #
            # ============================================================ #
            kv_start = cta_rank * kv_per_cta
            kv_stop  = kv_start + kv_per_cta
            if kv_stop > seq_len:
                kv_stop = seq_len

            local_max = -cutlass.Float32.inf
            local_sum =  cutlass.Float32(0.0)

            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)
            acc_o   = cute.make_tensor(acc_ptr, cute.make_layout((head_dim,)))
            if tidx < head_dim:
                acc_o[tidx] = cutlass.Float32(0.0)
            cute.arch.barrier()

            for kv_idx in range(kv_start, kv_stop):
                if tidx < head_dim:
                    qk = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        qk = qk + local_qkv[d].to(cutlass.Float32) * k_cache[kv_idx, head_id, d].to(cutlass.Float32)
                    qk = qk * softmax_scale

                    prev_max = local_max
                    local_max = qk if qk > local_max else local_max
                    scale_old = cute.math.exp(prev_max - local_max)
                    local_sum = local_sum * scale_old + cute.math.exp(qk - local_max)

                    prob = cute.math.exp(qk - local_max)
                    acc_o[tidx] = acc_o[tidx] * scale_old + prob * v_cache[kv_idx, head_id, tidx].to(cutlass.Float32)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 4 – Cross-CTA softmax reduce via global scratch        #
            # ============================================================ #
            # Write partials to global scratch
            if tidx < head_dim:
                scratch_out[head_id, cta_rank, tidx] = acc_o[tidx].to(cutlass.Float16)
            if tidx == 0:
                scratch_max[head_id, cta_rank] = local_max
                scratch_sum[head_id, cta_rank] = local_sum

            # Ensure all writes visible across cluster
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # Leader CTA reads all partials and reduces
            if cta_rank == 0:
                # Global max
                global_max = -cutlass.Float32.inf
                for c in range(cluster_size):
                    c_max = scratch_max[head_id, c].to(cutlass.Float32)
                    global_max = global_max if global_max > c_max else c_max

                # Global sum with rescaling
                global_sum = cutlass.Float32(0.0)
                for c in range(cluster_size):
                    c_max = scratch_max[head_id, c].to(cutlass.Float32)
                    c_sum = scratch_sum[head_id, c].to(cutlass.Float32)
                    rescale = cute.math.exp(c_max - global_max)
                    global_sum = global_sum + c_sum * rescale

                # Accumulate partial output vectors
                if tidx < head_dim:
                    final_o = cutlass.Float32(0.0)
                    for c in range(cluster_size):
                        c_max = scratch_max[head_id, c].to(cutlass.Float32)
                        rescale = cute.math.exp(c_max - global_max)
                        c_out_val = scratch_out[head_id, c, tidx].to(cutlass.Float32)
                        final_o = final_o + c_out_val * rescale

                    inv_sum = cutlass.Float32(1.0) / global_sum
                    final_o = final_o * inv_sum
                    local_qkv[2 * head_dim + tidx] = final_o.to(cutlass.Float16)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 5 – W_o GEMM (leader CTA only)                        #
            # ============================================================ #
            if cta_rank == 0:
                if tidx < head_dim:
                    for out_col in range(hidden_dim):
                        if out_col % num_threads == tidx:
                            partial = cutlass.Float32(0.0)
                            for d in range(head_dim):
                                a_val = local_qkv[2 * head_dim + d].to(cutlass.Float32)
                                w_val = w_o[head_id * head_dim + d, out_col].to(cutlass.Float32)
                                partial = partial + a_val * w_val
                            scratch_wo[head_id, out_col] = partial.to(cutlass.Float16)

        @cute.jit
        def _megakernel_host(
            hidden, w_qkv, w_o, k_cache, v_cache,
            rms_weight, cos_rope, sin_rope,
            output, k_out, v_out,
            scratch_l2, scratch_max, scratch_sum, scratch_out, scratch_wo,
            softmax_scale,
        ):
            _megakernel(
                hidden, w_qkv, w_o, k_cache, v_cache,
                rms_weight, cos_rope, sin_rope,
                output, k_out, v_out,
                scratch_l2, scratch_max, scratch_sum, scratch_out, scratch_wo,
                softmax_scale,
            ).launch(
                grid=(num_heads * cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return _megakernel_host


def cluster_megakernel_forward(
    hidden_states, w_qkv, w_o, k_cache, v_cache,
    rms_weight, cos_rope, sin_rope,
    config: MegakernelConfig | None = None,
):
    """Run the ClusterFusion-style decode megakernel (v2 baseline)."""
    require_torch()
    if not HAS_CUTE:
        raise RuntimeError("cluster_megakernel requires cutlass.cute.")

    config = config or MegakernelConfig()
    validate_megakernel_inputs(hidden_states, w_qkv, w_o, k_cache, v_cache, rms_weight, config)

    import torch

    seq_len      = k_cache.shape[0]
    hidden_dim   = config.hidden_dim
    num_heads    = config.num_heads
    head_dim     = config.head_dim
    cluster_size = config.cluster_size
    scale        = config.resolve_scale()

    output = torch.zeros((1, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    k_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    v_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

    # Global scratch buffers
    scratch_l2  = torch.zeros((num_heads, cluster_size), device=hidden_states.device, dtype=torch.float32)
    scratch_max = torch.zeros((num_heads, cluster_size), device=hidden_states.device, dtype=torch.float32)
    scratch_sum = torch.zeros((num_heads, cluster_size), device=hidden_states.device, dtype=torch.float32)
    scratch_out = torch.zeros((num_heads, cluster_size, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    scratch_wo  = torch.zeros((num_heads, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)

    def _wrap(t):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic()

    h_cute   = _wrap(hidden_states)
    wqkv_c   = _wrap(w_qkv)
    wo_c     = _wrap(w_o)
    kc_c     = _wrap(k_cache)
    vc_c     = _wrap(v_cache)
    rms_c    = _wrap(rms_weight)
    cos_c    = _wrap(cos_rope)
    sin_c    = _wrap(sin_rope)
    out_c    = _wrap(output)
    knew_c   = _wrap(k_new)
    vnew_c   = _wrap(v_new)
    sl2_c    = _wrap(scratch_l2)
    smax_c   = _wrap(scratch_max)
    ssum_c   = _wrap(scratch_sum)
    sout_c   = _wrap(scratch_out)
    swo_c    = _wrap(scratch_wo)

    cache_key = (seq_len, config)
    compiled  = _MEGAKERNEL_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host     = _make_cluster_megakernel_host(seq_len=seq_len, config=config)
        compiled = cute.compile(
            host,
            h_cute, wqkv_c, wo_c, kc_c, vc_c,
            rms_c, cos_c, sin_c,
            out_c, knew_c, vnew_c,
            sl2_c, smax_c, ssum_c, sout_c, swo_c,
            scale,
        )
        _MEGAKERNEL_COMPILED_CACHE[cache_key] = compiled

    compiled(
        h_cute, wqkv_c, wo_c, kc_c, vc_c,
        rms_c, cos_c, sin_c,
        out_c, knew_c, vnew_c,
        sl2_c, smax_c, ssum_c, sout_c, swo_c,
        scale,
    )
    # Reduce W_o across heads: output = sum of all heads' partial projections
    output = scratch_wo.sum(dim=0).unsqueeze(0).to(hidden_states.dtype)

    return output, k_new, v_new