"""ClusterFusion-style decode megakernel implemented in CuTeDSL.

Full pipeline per cluster (= per attention head):

  Stage 0 – RMSNorm
    Each CTA normalises its DIM_PER_BLOCK slice of the hidden vector.
    Partial l2-sum is reduced across the cluster to get the global norm.

  Stage 1 – W_qkv GEMM  (input @ W_q, W_k, W_v)
    Each CTA computes head_dim rows of Q, K and V using its own
    DIM_PER_BLOCK columns of the weight matrix.
    The partial dot-products are reduced across the cluster → full Q/K/V
    in SMEM of every CTA (cluster_reduce LINEAR).

  Stage 2 – RoPE
    Applied in-register on the assembled Q and K vectors.

  Stage 3 – Flash-decoding attention
    Each CTA processes KV_DIM_PER_BLOCK rows from the KV cache.
    Online softmax with running max / sum per CTA.

  Stage 4 – Cross-CTA softmax reduce  (cluster_reduce ATTN)
    A two-pass scalar cluster-reduce finds the global max and sum;
    each CTA rescales its partial output vector accordingly.

  Stage 5 – W_o GEMM  (attn_out @ W_o)
    Each CTA computes its DIM_PER_BLOCK slice of the output projection.
    Partial results are atomically added to the global output tensor.

This mirrors the design of
  3rd/clusterfusion/include/5090/llama/kernel.cuh
but expressed in CuTeDSL / Python for Blackwell (sm_100 / sm_103).
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
        """Build and return the @cute.jit host function for the megakernel.

        All compile-time constants are captured by closure so the kernel body
        contains only runtime-varying values.
        """
        hidden_dim   = config.hidden_dim
        num_heads    = config.num_heads
        head_dim     = config.head_dim
        cluster_size = config.cluster_size
        num_threads  = config.num_threads
        dim_per_block = config.dim_per_block          # hidden_dim // cluster_size
        tile_attn    = config.tma_load_once // 2      # KV rows per TMA tile (attn)
        kv_per_cta   = ((seq_len + cluster_size - 1) // cluster_size + tile_attn - 1) & ~(tile_attn - 1)

        cluster_shape = (cluster_size, 1, 1)

        # ------------------------------------------------------------------ #
        # Kernel                                                               #
        # ------------------------------------------------------------------ #

        @cute.kernel
        def _megakernel(
            # Inputs
            hidden: cute.Tensor,       # (1, hidden_dim)  fp16
            w_qkv:  cute.Tensor,       # (3*hidden_dim, hidden_dim) fp16
            w_o:    cute.Tensor,       # (hidden_dim, hidden_dim)   fp16
            k_cache: cute.Tensor,      # (seq_len, num_heads, head_dim) fp16
            v_cache: cute.Tensor,      # (seq_len, num_heads, head_dim) fp16
            rms_weight: cute.Tensor,   # (hidden_dim,) fp16
            cos_rope: cute.Tensor,     # (head_dim,) fp32
            sin_rope: cute.Tensor,     # (head_dim,) fp32
            # Outputs
            output:  cute.Tensor,      # (1, hidden_dim) fp16  – zero-initialised
            k_out:   cute.Tensor,      # (1, num_heads, head_dim) fp16
            v_out:   cute.Tensor,      # (1, num_heads, head_dim) fp16
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank   = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            # head_id = which attention head this cluster is responsible for
            head_id    = bidx // cluster_size
            # Each CTA owns a contiguous DIM_PER_BLOCK slice of the hidden dim
            cta_block_start = cta_rank * dim_per_block

            eps = cutlass.Float32(1e-6)

            # ---------------------------------------------------------- #
            # SMEM layout                                                 #
            # ---------------------------------------------------------- #
            smem = cutlass.utils.SmemAllocator()

            # Scratch for intra-CTA reductions (num_threads floats)
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce     = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

            # local_qkv: holds [Q(head_dim) | K(head_dim) | V(head_dim)] in fp16
            local_qkv_ptr = smem.allocate_array(cutlass.Float16, num_elems=3 * head_dim)
            local_qkv     = cute.make_tensor(local_qkv_ptr, cute.make_layout((3 * head_dim,)))

            # Scalar cluster-shared scalars (RMSNorm sum, flash-decode max/sum)
            cls_sum_ptr = smem.allocate_array(cutlass.Float32, num_elems=1)
            cls_max_ptr = smem.allocate_array(cutlass.Float32, num_elems=1)

            # ============================================================ #
            # Stage 0 – RMSNorm                                            #
            # ============================================================ #
            # Each CTA computes partial ||x||^2 over its DIM_PER_BLOCK slice.
            local_l2 = cutlass.Float32(0.0)
            d = tidx
            while d < dim_per_block:
                val = hidden[0, cta_block_start + d].to(cutlass.Float32)
                local_l2 = local_l2 + val * val
                d = d + num_threads

            reduce[tidx] = local_l2
            cute.arch.barrier()
            stride = num_threads // 2
            while stride > 0:
                if tidx < stride:
                    reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
                cute.arch.barrier()
                stride = stride // 2

            local_l2 = reduce[0]

            # Cross-CTA sum (cluster_reduce_scalar_sum pattern)
            cls_sum_ptr[0] = local_l2
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                remote = cute.arch.mapa(cls_sum_ptr, peer)
                if tidx == 0:
                    cute.arch.atomic_add(remote, local_l2, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            cluster_l2 = cls_sum_ptr[0]
            rms_rcp    = cute.math.rsqrt(cluster_l2 / cutlass.Float32(hidden_dim) + eps)

            # ============================================================ #
            # Stage 1 – W_qkv GEMM                                         #
            # (input @ W_q), (input @ W_k), (input @ W_v)                  #
            #                                                               #
            # Each thread computes one output element of Q (or K or V).    #
            # Thread tidx → output head_dim element tidx (if tidx<head_dim).#
            # Multiple warps share the work: warp w → rows [w*rows_per_warp,#
            # (w+1)*rows_per_warp) of the output head vector.              #
            #                                                               #
            # Simplified loop (not yet TMA-pipelined).                     #
            # ============================================================ #
            num_warps      = num_threads // 32
            rows_per_warp  = head_dim // num_warps  # = head_dim / 4 = 32

            for proj in range(3):   # 0=Q, 1=K, 2=V
                # Weight row for this output element: w_qkv layout is
                # (3*hidden_dim rows, hidden_dim cols)
                # Row for output element 'out_d' in projection 'proj':
                #   global_row = head_id * head_dim + out_d + proj * hidden_dim
                # Col: cta_block_start .. cta_block_start + dim_per_block - 1

                if tidx < head_dim:
                    out_d        = tidx
                    global_row   = head_id * head_dim + out_d + proj * hidden_dim
                    acc          = cutlass.Float32(0.0)
                    for col in range(dim_per_block):
                        x_val  = hidden[0, cta_block_start + col].to(cutlass.Float32)
                        x_norm = x_val * rms_rcp * rms_weight[cta_block_start + col].to(cutlass.Float32)
                        w_val  = w_qkv[global_row, cta_block_start + col].to(cutlass.Float32)
                        acc    = acc + x_norm * w_val
                    # Partial – needs cluster reduce (LINEAR stage)
                    local_qkv[proj * head_dim + out_d] = acc.to(cutlass.Float16)

            cute.arch.barrier()

            # ---------------------------------------------------------- #
            # Cluster reduce LINEAR: accumulate partial Q/K/V dots        #
            # across all CTAs.  Each element gets cluster_size-1          #
            # contributions added via atomic_add on mapped SMEM pointer.  #
            # ---------------------------------------------------------- #
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                j = tidx
                while j < 3 * head_dim:
                    elem_ptr    = local_qkv_ptr + j
                    remote_elem = cute.arch.mapa(elem_ptr, peer)
                    local_val   = local_qkv_ptr[j].to(cutlass.Float32)
                    cute.arch.atomic_add(remote_elem, local_val.to(cutlass.Float16), scope="cluster")
                    j = j + num_threads
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            # After the reduce, local_qkv on every CTA holds the full Q/K/V.

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

            # Write current-token K and V to KV-cache output (leader CTA only)
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

            # Online softmax state (thread-local)
            local_max = -cutlass.Float32.inf
            local_sum =  cutlass.Float32(0.0)

            # Output accumulator: head_dim floats, one per tid (strided)
            # We use a fixed-size SMEM tile to collect results before the
            # cross-CTA reduce.  Each thread owns (head_dim / num_threads) elements.
            # For head_dim=128, num_threads=128: 1 element per thread.
            acc_ptr = smem.allocate_array(cutlass.Float32, num_elems=head_dim)
            acc_o   = cute.make_tensor(acc_ptr, cute.make_layout((head_dim,)))
            if tidx < head_dim:
                acc_o[tidx] = cutlass.Float32(0.0)
            cute.arch.barrier()

            # Q vector reloaded into register (all threads need full Q)
            # thread tidx reads element (tidx % head_dim)
            q_elem = local_qkv[tidx % head_dim].to(cutlass.Float32)

            for kv_idx in range(kv_start, kv_stop):
                # Compute Q·K for this KV row (only head_dim threads participate)
                if tidx < head_dim:
                    qk = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        qk = qk + local_qkv[d].to(cutlass.Float32) * k_cache[kv_idx, head_id, d].to(cutlass.Float32)
                    qk = qk * softmax_scale

                    # Online softmax update
                    prev_max = local_max
                    local_max = qk if qk > local_max else local_max
                    scale_old = cute.math.exp(prev_max - local_max)
                    local_sum = local_sum * scale_old + cute.math.exp(qk - local_max)

                    # Accumulate V contribution
                    prob = cute.math.exp(qk - local_max)
                    for d_out in range(head_dim):
                        if d_out == tidx:
                            # This thread accumulates element d_out of the output
                            acc_o[tidx] = acc_o[tidx] * scale_old + prob * v_cache[kv_idx, head_id, tidx].to(cutlass.Float32)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 4 – Cross-CTA softmax reduce (ATTN stage)             #
            # ============================================================ #
            # Step 4a: cluster-wide max
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

            # Rescale local sum and acc_o to the global max
            rescale = cute.math.exp(local_max - global_max)
            local_sum = local_sum * rescale
            if tidx < head_dim:
                acc_o[tidx] = acc_o[tidx] * rescale

            # Step 4b: cluster-wide sum
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

            # Accumulate partial output vectors across CTAs (ATTN cluster reduce)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in range(1, cluster_size):
                peer = (cta_rank + i) % cluster_size
                if tidx < head_dim:
                    elem_ptr    = acc_ptr + tidx
                    remote_elem = cute.arch.mapa(elem_ptr, peer)
                    cute.arch.atomic_add(remote_elem, acc_o[tidx], scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            # Normalise
            inv_sum = cutlass.Float32(1.0) / global_sum
            if tidx < head_dim:
                acc_o[tidx] = acc_o[tidx] * inv_sum
                # Store normalised attn output to local_qkv[2*head_dim + d] (V slot reused)
                local_qkv[2 * head_dim + tidx] = acc_o[tidx].to(cutlass.Float16)

            cute.arch.barrier()

            # ============================================================ #
            # Stage 5 – W_o GEMM  (attn_out @ W_o)                        #
            # Each CTA computes its dim_per_block output slice and          #
            # atomic-adds into the global output tensor.                    #
            # ============================================================ #
            # output shape: (1, hidden_dim); this CTA writes slice
            #   [cta_block_start, cta_block_start + dim_per_block).
            # W_o layout: (hidden_dim rows, hidden_dim cols)  row-major
            #   output[0, out_col] += attn_out[d] * W_o[d + head_id*head_dim, out_col]

            for out_col_local in range(dim_per_block):
                if out_col_local % num_threads == tidx:
                    out_col = cta_block_start + out_col_local
                    partial = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        a_val = local_qkv[2 * head_dim + d].to(cutlass.Float32)
                        w_val = w_o[head_id * head_dim + d, out_col].to(cutlass.Float32)
                        partial = partial + a_val * w_val
                    # atomic add because multiple CTAs/heads write to the same output row
                    cute.arch.atomic_add(output + out_col, partial.to(cutlass.Float16), scope="gpu")

        # ------------------------------------------------------------------ #
        # JIT host function                                                    #
        # ------------------------------------------------------------------ #

        @cute.jit
        def _megakernel_host(
            hidden:     cute.Tensor,
            w_qkv:      cute.Tensor,
            w_o:        cute.Tensor,
            k_cache:    cute.Tensor,
            v_cache:    cute.Tensor,
            rms_weight: cute.Tensor,
            cos_rope:   cute.Tensor,
            sin_rope:   cute.Tensor,
            output:     cute.Tensor,
            k_out:      cute.Tensor,
            v_out:      cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            # Grid: num_heads * cluster_size blocks, cluster_size per cluster
            _megakernel(
                hidden, w_qkv, w_o, k_cache, v_cache,
                rms_weight, cos_rope, sin_rope,
                output, k_out, v_out,
                softmax_scale,
            ).launch(
                grid=(num_heads * cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return _megakernel_host


def cluster_megakernel_forward(
    hidden_states,
    w_qkv,
    w_o,
    k_cache,
    v_cache,
    rms_weight,
    cos_rope,
    sin_rope,
    config: MegakernelConfig | None = None,
):
    """Run the full ClusterFusion-style decode megakernel.

    Args:
        hidden_states: (1, hidden_dim) fp16 CUDA tensor – input to RMSNorm.
        w_qkv:         (3*hidden_dim, hidden_dim) fp16 – Q/K/V projection weights.
        w_o:           (hidden_dim, hidden_dim) fp16 – output projection weight.
        k_cache:       (seq_len, num_heads, head_dim) fp16 – KV cache keys.
        v_cache:       (seq_len, num_heads, head_dim) fp16 – KV cache values.
        rms_weight:    (hidden_dim,) fp16 – RMSNorm learnable scale.
        cos_rope:      (head_dim,) fp32 – RoPE cosine values.
        sin_rope:      (head_dim,) fp32 – RoPE sine values.
        config:        MegakernelConfig, defaults to Llama-2-7B settings.

    Returns:
        output:  (1, hidden_dim) fp16 – result of attn @ W_o (residual NOT added).
        k_new:   (1, num_heads, head_dim) fp16 – current-token K vector.
        v_new:   (1, num_heads, head_dim) fp16 – current-token V vector.
    """
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

    # Wrap tensors for CuTeDSL
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
