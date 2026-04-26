"""ClusterFusion-style decode megakernel – v5 (CTA-cluster correctness path).

One CTA cluster per attention head, matching ClusterFusion's launch topology.
Stages are split across CTA-owned hidden/KV/output slices and use DSM
reductions where cross-CTA communication is required.

Pipeline per cluster (= per attention head):
  Stage 0 – RMSNorm (hidden slice per CTA + DSM scalar reduce)
  Stage 1 – W_qkv GEMM (hidden slice per CTA + DSM vector reduce)
  Stage 2 – RoPE (GPT-J style)
  Stage 3 – Flash-decode attention (previous KV slice per CTA + current KV)
  Stage 4 – W_o GEMM (output slice per CTA + global head reduction)
"""

from __future__ import annotations

import operator

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
    from .cluster_primitives import (
        cluster_reduce_scalar_max_mbarrier,
        cluster_reduce_scalar_sum_mbarrier,
        cluster_reduce_vector_sum_mbarrier,
    )

    @cute.jit
    def _block_sum_f32(local_val, reduce, tidx, num_threads: int):
        """Reduce one fp32 value across a CTA with warp shuffles.

        Compared with the old shared-memory tree reduction, this keeps the
        per-token QK dot path to one CTA barrier instead of log2(block) barriers.
        """
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        warps_per_block = num_threads // 32

        warp_val = cute.arch.warp_reduction(local_val, operator.add)
        if lane_idx == 0:
            reduce[warp_idx] = warp_val
        cute.arch.barrier()

        block_val = cutlass.Float32(0.0)
        if lane_idx < warps_per_block:
            block_val = reduce[lane_idx]
        return cute.arch.warp_reduction(block_val, operator.add)

    def _make_cluster_megakernel_host(
        seq_len: int,
        config: MegakernelConfig,
    ):
        hidden_dim   = config.hidden_dim
        num_heads    = config.num_heads
        head_dim     = config.head_dim
        num_threads  = config.num_threads
        dim_per_block = config.dim_per_block
        cluster_size = config.cluster_size
        cluster_shape = (cluster_size, 1, 1)
        qkv_elems = 3 * head_dim
        kv_per_cta = (seq_len + cluster_size - 1) // cluster_size

        @cute.kernel
        def _megakernel(
            hidden:      cute.Tensor,   # (1, hidden_dim)         fp16
            w_qkv:       cute.Tensor,   # (3*hidden_dim, hidden_dim) fp16
            w_o:         cute.Tensor,   # (hidden_dim, hidden_dim) fp16
            k_cache:     cute.Tensor,   # (seq_len, num_heads, head_dim) fp16
            v_cache:     cute.Tensor,   # (seq_len, num_heads, head_dim) fp16
            rms_weight:  cute.Tensor,   # (hidden_dim,)            fp16
            cos_rope:    cute.Tensor,   # (head_dim,)              fp32
            sin_rope:    cute.Tensor,   # (head_dim,)              fp32
            k_out:       cute.Tensor,   # (1, num_heads, head_dim) fp16
            v_out:       cute.Tensor,   # (1, num_heads, head_dim) fp16
            output_acc:  cute.Tensor,   # (1, hidden_dim)          fp32
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            head_id = bidx // cluster_size  # 1 CTA cluster per head

            eps = cutlass.Float32(1e-6)

            smem = cutlass.utils.SmemAllocator()

            # Intra-CTA reduction scratch
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce     = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))
            cluster_l2_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size)
            cluster_l2_mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

            # local_qkv: [Q | K | V] in fp32 to match the PyTorch reference's
            # internal precision. Public K/V outputs are cast to fp16 below.
            local_qkv_ptr = smem.allocate_array(cutlass.Float32, num_elems=3 * head_dim)
            local_qkv     = cute.make_tensor(local_qkv_ptr, cute.make_layout((3 * head_dim,)))
            qkv_recv_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size * qkv_elems)
            qkv_mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)
            attn_max_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size)
            attn_max_mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)
            attn_sum_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size)
            attn_sum_mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)
            attn_recv_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size * head_dim)
            attn_mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

            # ============================================================ #
            # Stage 0 – RMSNorm                                            #
            # Each CTA owns one DIM_PER_BLOCK hidden slice, then DSM-reduces
            # the per-CTA partial L2 sum across the cluster.
            # ============================================================ #
            local_l2 = cutlass.Float32(0.0)
            slice_start = cta_rank * dim_per_block
            slice_stop = slice_start + dim_per_block
            col = slice_start + tidx
            while col < slice_stop:
                val = hidden[0, col].to(cutlass.Float32)
                local_l2 = local_l2 + val * val
                col = col + num_threads

            # Intra-CTA tree reduce to one partial L2 per CTA.
            reduce[tidx] = local_l2
            cute.arch.barrier()
            stride = num_threads // 2
            while stride > 0:
                if tidx < stride:
                    reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                cute.arch.barrier()
                stride = stride // 2

            total_l2 = cluster_reduce_scalar_sum_mbarrier(
                cluster_l2_ptr,
                cluster_l2_mbar_ptr,
                reduce[0],
                tidx,
                cluster_size,
            )

            # RMSNorm: rms_rcp = rsqrt(mean(x^2) + eps)
            mean_l2 = total_l2 / cutlass.Float32(hidden_dim)
            rms_rcp = cute.math.rsqrt(mean_l2 + eps)

            # ============================================================ #
            # Stage 1 – W_qkv GEMM                                        #
            # h_norm @ W_qkv^T                                             #
            # Each CTA accumulates its DIM_PER_BLOCK hidden slice, then DSM #
            # reduces the 3*head_dim partial vector across the cluster.     #
            # w_qkv layout: (3*hidden_dim, hidden_dim) row-major           #
            # output element j = sum_i (h_norm[i] * w_qkv[j, i])          #
            # ============================================================ #
            for proj in range(3):   # 0=Q, 1=K, 2=V
                if tidx < head_dim:
                    out_d      = tidx
                    global_row = head_id * head_dim + out_d + proj * hidden_dim
                    acc        = cutlass.Float32(0.0)
                    for i in range(slice_start, slice_stop):
                        x_val  = hidden[0, i].to(cutlass.Float32)
                        x_norm = x_val * rms_rcp * rms_weight[i].to(cutlass.Float32)
                        x_norm = x_norm.to(cutlass.Float16).to(cutlass.Float32)
                        w_val  = w_qkv[global_row, i].to(cutlass.Float32)
                        acc    = acc + x_norm * w_val
                    local_qkv[proj * head_dim + out_d] = acc

            cute.arch.barrier()

            cluster_reduce_vector_sum_mbarrier(
                local_qkv_ptr,
                qkv_recv_ptr,
                qkv_mbar_ptr,
                qkv_elems,
                tidx,
                cluster_size,
                num_threads,
            )

            # SGLang Llama exposes activation-dtype QKV projection outputs to
            # RoPE/attention. Keep values in fp32 storage for the scalar code,
            # but quantize at the same semantic boundary.
            qkv_idx = tidx
            while qkv_idx < qkv_elems:
                local_qkv[qkv_idx] = local_qkv[qkv_idx].to(cutlass.Float16).to(cutlass.Float32)
                qkv_idx = qkv_idx + num_threads
            cute.arch.barrier()

            # ============================================================ #
            # Stage 2 – RoPE (GPT-J style, interleaved pairs)             #
            # ============================================================ #
            if tidx < head_dim and tidx % 2 == 0:
                q0 = local_qkv[tidx].to(cutlass.Float32)
                q1 = local_qkv[tidx + 1].to(cutlass.Float32)
                k0 = local_qkv[head_dim + tidx].to(cutlass.Float32)
                k1 = local_qkv[head_dim + tidx + 1].to(cutlass.Float32)
                c0 = cos_rope[tidx].to(cutlass.Float16).to(cutlass.Float32)
                s0 = sin_rope[tidx].to(cutlass.Float16).to(cutlass.Float32)
                c1 = cos_rope[tidx + 1].to(cutlass.Float16).to(cutlass.Float32)
                s1 = sin_rope[tidx + 1].to(cutlass.Float16).to(cutlass.Float32)
                local_qkv[tidx]                = (q0 * c0 - q1 * s0).to(cutlass.Float16).to(cutlass.Float32)
                local_qkv[tidx + 1]            = (q1 * c1 + q0 * s1).to(cutlass.Float16).to(cutlass.Float32)
                local_qkv[head_dim + tidx]     = (k0 * c0 - k1 * s0).to(cutlass.Float16).to(cutlass.Float32)
                local_qkv[head_dim + tidx + 1] = (k1 * c1 + k0 * s1).to(cutlass.Float16).to(cutlass.Float32)

            # Write K/V outputs
            cute.arch.barrier()
            if cta_rank == 0 and tidx < head_dim:
                k_out[0, head_id, tidx] = local_qkv[head_dim + tidx].to(cutlass.Float16)
                v_out[0, head_id, tidx] = local_qkv[2 * head_dim + tidx].to(cutlass.Float16)

            # ============================================================ #
            # Stage 3 – Flash-decoding attention                           #
            # Each CTA owns a previous-KV slice. CTA rank 0 additionally    #
            # folds in the current token K/V before the cluster reductions. #
            # ============================================================ #
            local_max = -cutlass.Float32.inf
            local_sum =  cutlass.Float32(0.0)

            # acc_o: head_dim float32 accumulator in SMEM
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)
            acc_o   = cute.make_tensor(acc_ptr, cute.make_layout((head_dim,)))
            if tidx < head_dim:
                acc_o[tidx] = cutlass.Float32(0.0)
            cute.arch.barrier()

            kv_start = cta_rank * kv_per_cta
            kv_stop = kv_start + kv_per_cta
            if kv_stop > seq_len:
                kv_stop = seq_len

            for kv_idx in range(kv_start, kv_stop):
                qk_part = cutlass.Float32(0.0)
                if tidx < head_dim:
                    qk_part = local_qkv[tidx].to(cutlass.Float32) * k_cache[kv_idx, head_id, tidx].to(cutlass.Float32)

                qk = _block_sum_f32(qk_part, reduce, tidx, num_threads) * softmax_scale
                if tidx < head_dim:
                    # Online softmax update
                    prev_max = local_max
                    local_max = qk if qk > local_max else local_max
                    scale_old = cute.math.exp(prev_max - local_max)
                    local_sum = local_sum * scale_old + cute.math.exp(qk - local_max)

                    prob = cute.math.exp(qk - local_max)
                    acc_o[tidx] = acc_o[tidx] * scale_old + prob * v_cache[kv_idx, head_id, tidx].to(cutlass.Float32)

            if cta_rank == 0:
                qk_new_part = cutlass.Float32(0.0)
                if tidx < head_dim:
                    k_new_val = local_qkv[head_dim + tidx].to(cutlass.Float16).to(cutlass.Float32)
                    qk_new_part = local_qkv[tidx].to(cutlass.Float32) * k_new_val

                qk_new = _block_sum_f32(qk_new_part, reduce, tidx, num_threads) * softmax_scale
                if tidx < head_dim:
                    prev_max_new = local_max
                    local_max = qk_new if qk_new > local_max else local_max
                    scale_old_new = cute.math.exp(prev_max_new - local_max)
                    local_sum = local_sum * scale_old_new + cute.math.exp(qk_new - local_max)

                    prob_new = cute.math.exp(qk_new - local_max)
                    v_new_val = local_qkv[2 * head_dim + tidx].to(cutlass.Float16).to(cutlass.Float32)
                    acc_o[tidx] = acc_o[tidx] * scale_old_new + prob_new * v_new_val

            global_max = cluster_reduce_scalar_max_mbarrier(
                attn_max_ptr,
                attn_max_mbar_ptr,
                local_max,
                tidx,
                cluster_size,
            )

            if tidx < head_dim:
                local_scale = cute.math.exp(local_max - global_max)
                local_sum = local_sum * local_scale
                acc_o[tidx] = acc_o[tidx] * local_scale

            global_sum = cluster_reduce_scalar_sum_mbarrier(
                attn_sum_ptr,
                attn_sum_mbar_ptr,
                local_sum,
                tidx,
                cluster_size,
            )

            cluster_reduce_vector_sum_mbarrier(
                acc_ptr,
                attn_recv_ptr,
                attn_mbar_ptr,
                head_dim,
                tidx,
                cluster_size,
                num_threads,
            )

            # Normalize
            cute.arch.barrier()
            inv_sum = cutlass.Float32(1.0) / global_sum
            if tidx < head_dim:
                acc_o[tidx] = acc_o[tidx] * inv_sum
                # Store attn output to V slot of local_qkv (reuse)
                local_qkv[2 * head_dim + tidx] = acc_o[tidx].to(cutlass.Float16).to(cutlass.Float32)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 4 – W_o GEMM (per head, output slice per CTA)           #
            # Reference computes output = attn_vec @ w_o.T, so each head
            # contributes sum_d attn_out[d] * w_o[col, head_id*head_dim+d].
            # Accumulate all heads directly into the final fp32 output.      #
            # ============================================================ #
            for out_col in range(slice_start, slice_stop):
                if (out_col - slice_start) % num_threads == tidx:
                    partial = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        a_val = local_qkv[2 * head_dim + d].to(cutlass.Float32)
                        w_val = w_o[out_col, head_id * head_dim + d].to(cutlass.Float32)
                        partial = partial + a_val * w_val
                    out_ptr = output_acc.iterator + cute.crd2idx(
                        (0, out_col), output_acc.layout
                    )
                    cute.arch.atomic_add(
                        out_ptr.llvm_ptr,
                        partial,
                        sem="relaxed",
                        scope="gpu",
                    )

        @cute.jit
        def _megakernel_host(
            hidden, w_qkv, w_o, k_cache, v_cache,
            rms_weight, cos_rope, sin_rope,
            k_out, v_out, output_acc,
            softmax_scale,
        ):
            _megakernel(
                hidden, w_qkv, w_o, k_cache, v_cache,
                rms_weight, cos_rope, sin_rope,
                k_out, v_out, output_acc,
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
    use_tensor_core: bool = False,
):
    """Run the ClusterFusion-style decode megakernel.

    Set ``use_tensor_core=True`` to run the tensor-core implementation path for
    QKV/WO.  The default remains the original CuTeDSL CTA-cluster correctness
    kernel so existing tests and benchmarks keep their historical behavior.
    """
    require_torch()
    if use_tensor_core:
        from .cluster_megakernel_tc import cluster_megakernel_tc_forward

        return cluster_megakernel_tc_forward(
            hidden_states,
            w_qkv,
            w_o,
            k_cache,
            v_cache,
            rms_weight,
            cos_rope,
            sin_rope,
            config=config,
        )

    if not HAS_CUTE:
        raise RuntimeError("cluster_megakernel requires cutlass.cute.")

    config = config or MegakernelConfig()
    validate_megakernel_inputs(hidden_states, w_qkv, w_o, k_cache, v_cache, rms_weight, config)

    import torch

    seq_len     = k_cache.shape[0]
    hidden_dim  = config.hidden_dim
    num_heads   = config.num_heads
    head_dim    = config.head_dim
    scale       = config.resolve_scale()

    k_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    v_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    output_acc = torch.zeros((1, hidden_dim), device=hidden_states.device, dtype=torch.float32)

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
    knew_c   = _wrap(k_new)
    vnew_c   = _wrap(v_new)
    out_c    = _wrap(output_acc)

    cache_key = (seq_len, config)
    compiled  = _MEGAKERNEL_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host     = _make_cluster_megakernel_host(seq_len=seq_len, config=config)
        compiled = cute.compile(
            host,
            h_cute, wqkv_c, wo_c, kc_c, vc_c,
            rms_c, cos_c, sin_c,
            knew_c, vnew_c, out_c,
            scale,
        )
        _MEGAKERNEL_COMPILED_CACHE[cache_key] = compiled

    compiled(
        h_cute, wqkv_c, wo_c, kc_c, vc_c,
        rms_c, cos_c, sin_c,
        knew_c, vnew_c, out_c,
        scale,
    )
    torch.cuda.synchronize()

    output = output_acc.to(hidden_states.dtype)
    return output, k_new, v_new
