"""ClusterFusion-style decode megakernel implemented in CuTeDSL.

Full pipeline per cluster (= per attention head):

  Stage 0 – RMSNorm
  Stage 1 – W_qkv GEMM (cluster_reduce LINEAR)
  Stage 2 – RoPE
  Stage 3 – Flash-decoding attention (per-CTA KV slice)
  Stage 4 – Cross-CTA softmax reduce (cluster_reduce ATTN)
  Stage 5 – W_o GEMM (atomic add to global output)

Uses official CuTeDSL cluster communication primitives from
  cutlass/examples/python/CuTeDSL/blackwell/reduce.py
(st.async.shared::cluster + mbarrier pattern) instead of raw atomic_add.
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

# ------------------------------------------------------------------ #
# Cluster communication helpers (copied from official reduce.py)       #
# ------------------------------------------------------------------ #

if HAS_CUTE:
    import operator
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op, Int32, Float32

    @dsl_user_op
    def _set_block_rank(smem_ptr, peer_cta_rank_in_cluster, *, loc=None, ip=None):
        """mapa.shared::cluster – map local SMEM ptr to peer CTA address."""
        smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return Int32(
            llvm.inline_asm(
                cutlass.Int32.mlir_type,
                [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_shared_remote_f32(val, smem_ptr_i32, mbar_ptr_i32, *, loc=None, ip=None):
        """st.async.shared::cluster.mbarrier::complete_tx::bytes.f32"""
        llvm.inline_asm(
            None,
            [smem_ptr_i32, val.ir_value(loc=loc, ip=ip), mbar_ptr_i32],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
            "r,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _store_shared_remote_f16(val, smem_ptr_i32, mbar_ptr_i32, *, loc=None, ip=None):
        """st.async.shared::cluster.mbarrier::complete_tx::bytes.f16"""
        llvm.inline_asm(
            None,
            [smem_ptr_i32, val.ir_value(loc=loc, ip=ip), mbar_ptr_i32],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f16 [$0], $1, [$2];",
            "r,h,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


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
        num_warps     = num_threads // 32

        @cute.kernel
        def _megakernel(
            hidden:      cute.Tensor,   # (1, hidden_dim)        fp16
            w_qkv:       cute.Tensor,   # (3*hidden_dim, hidden_dim) fp16
            w_o:         cute.Tensor,   # (hidden_dim, hidden_dim) fp16
            k_cache:     cute.Tensor,   # (seq_len, num_heads, head_dim) fp16
            v_cache:     cute.Tensor,   # (seq_len, num_heads, head_dim) fp16
            rms_weight:  cute.Tensor,   # (hidden_dim,)          fp16
            cos_rope:    cute.Tensor,   # (head_dim,)            fp32
            sin_rope:    cute.Tensor,   # (head_dim,)            fp32
            output:      cute.Tensor,   # (1, hidden_dim)        fp16
            k_out:       cute.Tensor,   # (1, num_heads, head_dim) fp16
            v_out:       cute.Tensor,   # (1, num_heads, head_dim) fp16
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank   = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            head_id    = bidx // cluster_size
            cta_block_start = cta_rank * dim_per_block
            lane_id    = tidx % 32
            warp_id    = tidx // 32

            eps = cutlass.Float32(1e-6)

            smem = cutlass.utils.SmemAllocator()

            # Intra-CTA reduction scratch
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce     = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

            # local_qkv: [Q | K | V] in fp16
            local_qkv_ptr = smem.allocate_array(cutlass.Float16, num_elems=3 * head_dim)
            local_qkv     = cute.make_tensor(local_qkv_ptr, cute.make_layout((3 * head_dim,)))

            # mbarrier for cluster st.async
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

            # Scalar cluster-shared values
            cls_sum_ptr = smem.allocate_array(cutlass.Float32, num_elems=1)
            cls_max_ptr = smem.allocate_array(cutlass.Float32, num_elems=1)

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

            # Cross-CTA cluster reduce (scalar sum)
            # Pattern: tid==0 does atomicAdd to peer's __shared__ scalar
            cls_sum_ptr[0] = local_l2
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                # Use cute.arch.mapa on the underlying pointer
                remote_sum_ptr = cute.arch.mapa(cls_sum_ptr, peer)
                if tidx == 0:
                    cute.arch.atomic_add(remote_sum_ptr, local_l2, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            cluster_l2 = cls_sum_ptr[0]
            rms_rcp    = cute.math.rsqrt(cluster_l2 / cutlass.Float32(hidden_dim) + eps)

            # ============================================================ #
            # Stage 1 – W_qkv GEMM                                         #
            # ============================================================ #
            for proj in range(3):
                if tidx < head_dim:
                    out_d      = tidx
                    global_row = head_id * head_dim + out_d + proj * hidden_dim
                    acc        = cutlass.Float32(0.0)
                    for col in range(dim_per_block):
                        x_val  = hidden[0, cta_block_start + col].to(cutlass.Float32)
                        x_norm = x_val * rms_rcp * rms_weight[cta_block_start + col].to(cutlass.Float32)
                        w_val  = w_qkv[global_row, cta_block_start + col].to(cutlass.Float32)
                        acc    = acc + x_norm * w_val
                    local_qkv[proj * head_dim + out_d] = acc.to(cutlass.Float16)

            cute.arch.barrier()

            # Cluster reduce LINEAR: each thread sends its partial fp16
            # values to all peer CTAs via st.async, then locally accumulates.
            #
            # Simplified approach: use cluster_arrive/cluster_wait to sync,
            # then each thread reads from the peer's SMEM buffer (via mapa)
            # using cute.arch.atomic_add on the mapped fp16 pointer.
            #
            # For fp16 atomic_add, we need element-level pointer mapping.
            # Since mapa returns an LLVM ptr and we need indexed access,
            # we use a per-element approach: each thread reads its local
            # value, maps a single-element pointer, and atomic-adds to the peer.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                # Each thread processes its strided slice of 3*head_dim elements
                j = tidx
                while j < 3 * head_dim:
                    # Get a Pointer to local_qkv[j] and map it to peer
                    local_elem_ptr = local_qkv_ptr + j
                    remote_elem_ptr = cute.arch.mapa(local_elem_ptr, peer)
                    local_val = local_qkv[j].to(cutlass.Float32)
                    cute.arch.atomic_add(remote_elem_ptr, local_val.to(cutlass.Float16), scope="cluster")
                    j = j + num_threads
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            # ============================================================ #
            # Stage 2 – RoPE (GPT-J style, interleaved pairs)             #
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

            # Write current-token K/V (leader CTA only)
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

            # acc_o: head_dim float32 accumulator in SMEM
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
            # Stage 4 – Cross-CTA softmax reduce                          #
            # ============================================================ #
            # 4a: cluster-wide max (scalar, atomic_max pattern)
            cls_max_ptr[0] = local_max
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                remote = cute.arch.mapa(cls_max_ptr, peer)
                if tidx == 0:
                    cute.arch.atomic_max(remote, local_max, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            global_max = cls_max_ptr[0]

            rescale = cute.math.exp(local_max - global_max)
            local_sum = local_sum * rescale
            if tidx < head_dim:
                acc_o[tidx] = acc_o[tidx] * rescale

            # 4b: cluster-wide sum (scalar, atomic_add pattern)
            cls_sum_ptr[0] = local_sum
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                remote = cute.arch.mapa(cls_sum_ptr, peer)
                if tidx == 0:
                    cute.arch.atomic_add(remote, local_sum, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            global_sum = cls_sum_ptr[0]

            # 4c: cluster-wide output vector reduce (element-wise atomic_add)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                if tidx < head_dim:
                    local_elem_ptr  = acc_ptr + tidx
                    remote_elem_ptr = cute.arch.mapa(local_elem_ptr, peer)
                    cute.arch.atomic_add(remote_elem_ptr, acc_o[tidx], scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            # Normalise
            inv_sum = cutlass.Float32(1.0) / global_sum
            if tidx < head_dim:
                acc_o[tidx] = acc_o[tidx] * inv_sum
                local_qkv[2 * head_dim + tidx] = acc_o[tidx].to(cutlass.Float16)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 5 – W_o GEMM                                           #
            # ============================================================ #
            for out_col_local in range(dim_per_block):
                if out_col_local % num_threads == tidx:
                    out_col = cta_block_start + out_col_local
                    partial = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        a_val = local_qkv[2 * head_dim + d].to(cutlass.Float32)
                        w_val = w_o[head_id * head_dim + d, out_col].to(cutlass.Float32)
                        partial = partial + a_val * w_val
                    # Use tensor indexing for atomic_add (not pointer arithmetic)
                    cute.arch.atomic_add(output[0, out_col], partial.to(cutlass.Float16), scope="gpu")

        # ------------------------------------------------------------------ #
        # JIT host                                                            #
        # ------------------------------------------------------------------ #

        @cute.jit
        def _megakernel_host(
            hidden, w_qkv, w_o, k_cache, v_cache,
            rms_weight, cos_rope, sin_rope,
            output, k_out, v_out, softmax_scale,
        ):
            _megakernel(
                hidden, w_qkv, w_o, k_cache, v_cache,
                rms_weight, cos_rope, sin_rope,
                output, k_out, v_out, softmax_scale,
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
    """Run the full ClusterFusion-style decode megakernel."""
    require_torch()
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

    output = torch.zeros((1, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    k_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    v_new  = torch.zeros((1, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

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

    cache_key = (seq_len, config)
    compiled  = _MEGAKERNEL_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host     = _make_cluster_megakernel_host(seq_len=seq_len, config=config)
        compiled = cute.compile(
            host,
            h_cute, wqkv_c, wo_c, kc_c, vc_c,
            rms_c, cos_c, sin_c,
            out_c, knew_c, vnew_c,
            scale,
        )
        _MEGAKERNEL_COMPILED_CACHE[cache_key] = compiled

    compiled(
        h_cute, wqkv_c, wo_c, kc_c, vc_c,
        rms_c, cos_c, sin_c,
        out_c, knew_c, vnew_c,
        scale,
    )
    return output, k_new, v_new