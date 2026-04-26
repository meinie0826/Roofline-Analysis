"""ClusterFusion-style decode megakernel with in-kernel tensor-core projections.

One CTA cluster per attention head, matching ClusterFusion's launch topology.
Stages are split across CTA-owned hidden/KV/output slices and use DSM
reductions where cross-CTA communication is required.

Pipeline per cluster (= per attention head):
  Stage 0 – RMSNorm (hidden slice per CTA + DSM scalar reduce)
  Stage 1 – W_qkv GEMV via warp MMA tiles (hidden slice per CTA + DSM vector reduce)
  Stage 2 – RoPE (GPT-J style)
  Stage 3 – Flash-decode attention (previous KV slice per CTA + current KV)
  Stage 4 – W_o GEMV via warp MMA tiles (output slice per CTA + global head reduction)
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

_MEGAKERNEL_TC_COMPILED_CACHE: dict = {}


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

    def _make_cluster_megakernel_tc_host(
        seq_len: int,
        config: MegakernelConfig,
    ):
        hidden_dim   = config.hidden_dim
        num_heads    = config.num_heads
        head_dim     = config.head_dim
        # The tensor-core path has many independent 16x16 projection tiles and
        # decode-attention KV lanes.  Use at least 8 warps per CTA even when the
        # scalar config uses 4 warps, while keeping the public config unchanged.
        num_threads  = max(config.num_threads, 256)
        dim_per_block = config.dim_per_block
        cluster_size = config.cluster_size
        cluster_shape = (cluster_size, 1, 1)
        qkv_elems = 3 * head_dim
        kv_per_cta = (seq_len + cluster_size - 1) // cluster_size
        num_mma_warps = num_threads // 32
        tc_m = 16
        tc_n = 8
        tc_k = 16
        qkv_tc_tiles = qkv_elems // tc_m
        qkv_k_tiles = dim_per_block // tc_k
        wo_tc_tiles = dim_per_block // tc_m
        wo_k_tiles = head_dim // tc_k

        @cute.kernel
        def _megakernel(
            qkv_tiled_mma: cute.TiledMma,
            qkv_sA_layout: cute.Layout,
            qkv_sB_layout: cute.Layout,
            qkv_sC_layout: cute.Layout,
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
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
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
            qkv_sA = smem.allocate_tensor(cutlass.Float16, qkv_sA_layout, 16)
            qkv_sB = smem.allocate_tensor(cutlass.Float16, qkv_sB_layout, 16)
            qkv_sC = smem.allocate_tensor(cutlass.Float32, qkv_sC_layout, 16)

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
            # Stage 1 – W_qkv tensor-core projection                       #
            # Recast each CTA's GEMV slice as tiled GEMM:
            #   A = W_qkv rows        (16 x 16)
            #   B = normalized token  (8 x 16, only column-lane 0 is live)
            #   C = partial QKV rows  (16 x 8, read C[:, 0])
            # This keeps QKV inside the same fused launch while using MMA.  #
            # ============================================================ #
            qkv_thr_mma = qkv_tiled_mma.get_slice(lane_idx)
            qkv_sA_warp = qkv_sA[warp_idx, None, None]
            qkv_sB_warp = qkv_sB[warp_idx, None, None]
            qkv_sC_warp = qkv_sC[warp_idx, None, None]
            qkv_tCsA = qkv_thr_mma.partition_A(qkv_sA_warp)
            qkv_tCsB = qkv_thr_mma.partition_B(qkv_sB_warp)
            qkv_tCsC = qkv_thr_mma.partition_C(qkv_sC_warp)
            qkv_tCrA = qkv_tiled_mma.make_fragment_A(qkv_tCsA)
            qkv_tCrB = qkv_tiled_mma.make_fragment_B(qkv_tCsB)
            qkv_tCrC = qkv_tiled_mma.make_fragment_C(qkv_tCsC)

            qkv_tile = warp_idx
            while qkv_tile < qkv_tc_tiles:
                qkv_tCrC.fill(0.0)

                for k_tile in range(qkv_k_tiles):
                    a_idx = lane_idx
                    while a_idx < tc_m * tc_k:
                        m = a_idx // tc_k
                        k = a_idx - m * tc_k
                        qkv_idx = qkv_tile * tc_m + m
                        proj = qkv_idx // head_dim
                        out_d = qkv_idx - proj * head_dim
                        global_row = head_id * head_dim + out_d + proj * hidden_dim
                        global_col = slice_start + k_tile * tc_k + k
                        qkv_sA_warp[m, k] = w_qkv[global_row, global_col]
                        a_idx = a_idx + 32

                    b_idx = lane_idx
                    while b_idx < tc_n * tc_k:
                        n = b_idx // tc_k
                        k = b_idx - n * tc_k
                        if n == 0:
                            global_col = slice_start + k_tile * tc_k + k
                            x_val = hidden[0, global_col].to(cutlass.Float32)
                            x_norm = x_val * rms_rcp * rms_weight[global_col].to(cutlass.Float32)
                            qkv_sB_warp[n, k] = x_norm.to(cutlass.Float16)
                        else:
                            qkv_sB_warp[n, k] = cutlass.Float16(0.0)
                        b_idx = b_idx + 32

                    cute.arch.sync_warp()
                    cute.autovec_copy(qkv_tCsA, qkv_tCrA)
                    cute.autovec_copy(qkv_tCsB, qkv_tCrB)
                    cute.gemm(
                        qkv_tiled_mma,
                        qkv_tCrC,
                        qkv_tCrA,
                        qkv_tCrB,
                        qkv_tCrC,
                    )
                    cute.arch.sync_warp()

                cute.autovec_copy(qkv_tCrC, qkv_tCsC)
                cute.arch.sync_warp()

                if lane_idx < tc_m:
                    local_qkv[qkv_tile * tc_m + lane_idx] = qkv_sC_warp[lane_idx, 0]
                qkv_tile = qkv_tile + num_mma_warps

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

            # One warp owns one interleaved KV subsequence.  We combine the
            # four online-softmax states inside the CTA before DSM reduction.
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_mma_warps * head_dim)
            acc_o = cute.make_tensor(acc_ptr, cute.make_layout((num_mma_warps, head_dim)))
            warp_max_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_mma_warps)
            warp_max = cute.make_tensor(warp_max_ptr, cute.make_layout((num_mma_warps,)))
            warp_sum_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_mma_warps)
            warp_sum = cute.make_tensor(warp_sum_ptr, cute.make_layout((num_mma_warps,)))
            cta_acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)
            cta_acc = cute.make_tensor(cta_acc_ptr, cute.make_layout((head_dim,)))

            acc_init = tidx
            while acc_init < num_mma_warps * head_dim:
                acc_w = acc_init // head_dim
                acc_d = acc_init - acc_w * head_dim
                acc_o[acc_w, acc_d] = cutlass.Float32(0.0)
                acc_init = acc_init + num_threads
            cute.arch.barrier()

            kv_start = cta_rank * kv_per_cta
            kv_stop = kv_start + kv_per_cta
            if kv_stop > seq_len:
                kv_stop = seq_len

            kv_idx = kv_start + warp_idx
            while kv_idx < kv_stop:
                qk_part = cutlass.Float32(0.0)
                for d_iter in range(head_dim // 32):
                    d = lane_idx + d_iter * 32
                    qk_part = qk_part + local_qkv[d].to(cutlass.Float32) * k_cache[kv_idx, head_id, d].to(cutlass.Float32)

                qk = cute.arch.warp_reduction(qk_part, operator.add) * softmax_scale
                prev_max = local_max
                local_max = qk if qk > local_max else local_max
                scale_old = cute.math.exp(prev_max - local_max)
                local_sum = local_sum * scale_old + cute.math.exp(qk - local_max)

                prob = cute.math.exp(qk - local_max)
                for d_iter in range(head_dim // 32):
                    d = lane_idx + d_iter * 32
                    acc_o[warp_idx, d] = acc_o[warp_idx, d] * scale_old + prob * v_cache[kv_idx, head_id, d].to(cutlass.Float32)
                kv_idx = kv_idx + num_mma_warps

            if cta_rank == 0 and warp_idx == 0:
                qk_new_part = cutlass.Float32(0.0)
                for d_iter in range(head_dim // 32):
                    d = lane_idx + d_iter * 32
                    k_new_val = local_qkv[head_dim + d].to(cutlass.Float16).to(cutlass.Float32)
                    qk_new_part = qk_new_part + local_qkv[d].to(cutlass.Float32) * k_new_val

                qk_new = cute.arch.warp_reduction(qk_new_part, operator.add) * softmax_scale
                prev_max_new = local_max
                local_max = qk_new if qk_new > local_max else local_max
                scale_old_new = cute.math.exp(prev_max_new - local_max)
                local_sum = local_sum * scale_old_new + cute.math.exp(qk_new - local_max)

                prob_new = cute.math.exp(qk_new - local_max)
                for d_iter in range(head_dim // 32):
                    d = lane_idx + d_iter * 32
                    v_new_val = local_qkv[2 * head_dim + d].to(cutlass.Float16).to(cutlass.Float32)
                    acc_o[warp_idx, d] = acc_o[warp_idx, d] * scale_old_new + prob_new * v_new_val

            if lane_idx == 0:
                warp_max[warp_idx] = local_max
                warp_sum[warp_idx] = local_sum
            cute.arch.barrier()

            if tidx == 0:
                cta_max = -cutlass.Float32.inf
                for w in range(num_mma_warps):
                    cta_max = warp_max[w] if warp_max[w] > cta_max else cta_max
                reduce[0] = cta_max
            cute.arch.barrier()

            cta_max = reduce[0]
            local_scale = cute.math.exp(local_max - cta_max)
            local_sum = local_sum * local_scale
            for d_iter in range(head_dim // 32):
                d = lane_idx + d_iter * 32
                acc_o[warp_idx, d] = acc_o[warp_idx, d] * local_scale
            if lane_idx == 0:
                warp_sum[warp_idx] = local_sum
            cute.arch.barrier()

            if tidx == 0:
                cta_sum = cutlass.Float32(0.0)
                for w in range(num_mma_warps):
                    cta_sum = cta_sum + warp_sum[w]
                reduce[0] = cta_sum
            if tidx < head_dim:
                acc_sum = cutlass.Float32(0.0)
                for w in range(num_mma_warps):
                    acc_sum = acc_sum + acc_o[w, tidx]
                cta_acc[tidx] = acc_sum
            cute.arch.barrier()

            cta_sum = reduce[0]

            global_max = cluster_reduce_scalar_max_mbarrier(
                attn_max_ptr,
                attn_max_mbar_ptr,
                cta_max,
                tidx,
                cluster_size,
            )

            if tidx < head_dim:
                cluster_scale = cute.math.exp(cta_max - global_max)
                cta_acc[tidx] = cta_acc[tidx] * cluster_scale
            cta_sum = cta_sum * cute.math.exp(cta_max - global_max)

            global_sum = cluster_reduce_scalar_sum_mbarrier(
                attn_sum_ptr,
                attn_sum_mbar_ptr,
                cta_sum,
                tidx,
                cluster_size,
            )

            cluster_reduce_vector_sum_mbarrier(
                cta_acc_ptr,
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
                cta_acc[tidx] = cta_acc[tidx] * inv_sum
                # Store attn output to V slot of local_qkv (reuse)
                local_qkv[2 * head_dim + tidx] = cta_acc[tidx].to(cutlass.Float16).to(cutlass.Float32)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 4 – W_o tensor-core projection                          #
            # Reference computes output = attn_vec @ w_o.T, so each head
            # contributes sum_d attn_out[d] * w_o[col, head_id*head_dim+d].
            # Accumulate all heads directly into the final fp32 output.      #
            # ============================================================ #
            wo_tile = warp_idx
            while wo_tile < wo_tc_tiles:
                qkv_tCrC.fill(0.0)

                for k_tile in range(wo_k_tiles):
                    a_idx = lane_idx
                    while a_idx < tc_m * tc_k:
                        m = a_idx // tc_k
                        k = a_idx - m * tc_k
                        out_col = slice_start + wo_tile * tc_m + m
                        in_col = head_id * head_dim + k_tile * tc_k + k
                        qkv_sA_warp[m, k] = w_o[out_col, in_col]
                        a_idx = a_idx + 32

                    b_idx = lane_idx
                    while b_idx < tc_n * tc_k:
                        n = b_idx // tc_k
                        k = b_idx - n * tc_k
                        if n == 0:
                            d = k_tile * tc_k + k
                            qkv_sB_warp[n, k] = local_qkv[2 * head_dim + d].to(cutlass.Float16)
                        else:
                            qkv_sB_warp[n, k] = cutlass.Float16(0.0)
                        b_idx = b_idx + 32

                    cute.arch.sync_warp()
                    cute.autovec_copy(qkv_tCsA, qkv_tCrA)
                    cute.autovec_copy(qkv_tCsB, qkv_tCrB)
                    cute.gemm(
                        qkv_tiled_mma,
                        qkv_tCrC,
                        qkv_tCrA,
                        qkv_tCrB,
                        qkv_tCrC,
                    )
                    cute.arch.sync_warp()

                cute.autovec_copy(qkv_tCrC, qkv_tCsC)
                cute.arch.sync_warp()

                if lane_idx < tc_m:
                    out_col = slice_start + wo_tile * tc_m + lane_idx
                    out_ptr = output_acc.iterator + cute.crd2idx(
                        (0, out_col), output_acc.layout
                    )
                    cute.arch.atomic_add(
                        out_ptr.llvm_ptr,
                        qkv_sC_warp[lane_idx, 0],
                        sem="relaxed",
                        scope="gpu",
                    )
                wo_tile = wo_tile + num_mma_warps

        @cute.jit
        def _megakernel_host(
            hidden, w_qkv, w_o, k_cache, v_cache,
            rms_weight, cos_rope, sin_rope,
            k_out, v_out, output_acc,
            softmax_scale,
        ):
            qkv_op = cute.nvgpu.warp.MmaF16BF16Op(
                cutlass.Float16,
                cutlass.Float32,
                (tc_m, tc_n, tc_k),
            )
            qkv_tiled_mma = cute.make_tiled_mma(qkv_op)
            qkv_sA_layout = cute.make_layout(
                (num_mma_warps, tc_m, tc_k),
                stride=(tc_m * tc_k, tc_k, 1),
            )
            qkv_sB_layout = cute.make_layout(
                (num_mma_warps, tc_n, tc_k),
                stride=(tc_n * tc_k, tc_k, 1),
            )
            qkv_sC_layout = cute.make_layout(
                (num_mma_warps, tc_m, tc_n),
                stride=(tc_m * tc_n, tc_n, 1),
            )
            _megakernel(
                qkv_tiled_mma,
                qkv_sA_layout,
                qkv_sB_layout,
                qkv_sC_layout,
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


def cluster_megakernel_tc_forward(
    hidden_states, w_qkv, w_o, k_cache, v_cache,
    rms_weight, cos_rope, sin_rope,
    config: MegakernelConfig | None = None,
):
    """Run the fused CuTeDSL tensor-core megakernel path.

    QKV and WO projection are performed by warp-level MMA tiles inside the
    same CTA cluster kernel as RMSNorm, RoPE, and attention.
    """
    require_torch()
    if not HAS_CUTE:
        raise RuntimeError("cluster_megakernel_tc requires cutlass.cute.")

    config = config or MegakernelConfig()
    validate_megakernel_inputs(hidden_states, w_qkv, w_o, k_cache, v_cache, rms_weight, config)
    if config.head_dim % 16 != 0:
        raise ValueError("tensor-core QKV path requires head_dim to be divisible by 16.")
    if config.dim_per_block % 16 != 0:
        raise ValueError("tensor-core QKV path requires dim_per_block to be divisible by 16.")

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
    compiled  = _MEGAKERNEL_TC_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host     = _make_cluster_megakernel_tc_host(seq_len=seq_len, config=config)
        compiled = cute.compile(
            host,
            h_cute, wqkv_c, wo_c, kc_c, vc_c,
            rms_c, cos_c, sin_c,
            knew_c, vnew_c, out_c,
            scale,
        )
        _MEGAKERNEL_TC_COMPILED_CACHE[cache_key] = compiled

    compiled(
        h_cute, wqkv_c, wo_c, kc_c, vc_c,
        rms_c, cos_c, sin_c,
        knew_c, vnew_c, out_c,
        scale,
    )
    torch.cuda.synchronize()

    output = output_acc.to(hidden_states.dtype)
    return output, k_new, v_new
