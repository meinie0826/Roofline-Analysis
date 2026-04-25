"""CuTeDSL cluster communication primitives.

Mirrors the role of dsm.cuh from ClusterFusion:
  - cluster_reduce_linear  ← Stage::LINEAR  (accumulate 3×head_dim halves across CTAs)
  - cluster_reduce_attn    ← Stage::ATTN    (accumulate 1×head_dim halves across CTAs)
  - cluster_reduce_max     ← scalar fmaxf across CTAs via __shared__ + atomicMax
  - cluster_reduce_sum     ← scalar DSM store/reduce across CTAs via __shared__

The underlying mechanism is:
  1. Each CTA writes its partial result to its own SMEM buffer.
  2. All CTAs do cluster_arrive / cluster_wait to synchronise.
  3. Each CTA uses inline PTX mapa + st.async.shared::cluster to send its
     local value/vector slots to every peer CTA's SMEM.
  4. Each CTA waits on a local mbarrier, then locally accumulates the received
     slots from all peers.

Note: the high-level `cute.arch.mapa() + cute.arch.atomic_add()` path currently
ICEs in NVVM on sm_100a.  Scalar reductions therefore use the same lower-level
pattern as CUTLASS' `blackwell/reduce.py`: inline PTX `mapa.shared::cluster`
plus `st.async.shared::cluster...mbarrier::complete_tx`, then a local sum over
the values received from peer CTAs.

For the ATTN and LINEAR stages the ClusterFusion CUDA code uses
  cp.async.bulk.shared::cluster + mbarrier
to move a tile of halfs, then does a half2 add-reduction.  The equivalent in
CuTeDSL correctness path uses:
  - inline PTX mapa to translate a local SMEM pointer to a peer CTA
  - st.async.shared::cluster to send float32 partials to every peer
  - a local element-wise sum over the received peer slots

This is not as efficient as a bulk copy, but it is a validated correctness path.
A future revision can switch to a true bulk-copy path for vector reductions.
"""

from __future__ import annotations

from .common import HAS_CUTE, cutlass, cute


if HAS_CUTE:
    from cutlass import Float32, Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def _mapa_shared_cluster(
        smem_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_shared_remote_f32(
        val: Float32,
        smem_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        remote_smem_i32 = _mapa_shared_cluster(
            smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        remote_mbar_i32 = _mapa_shared_cluster(
            mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        llvm.inline_asm(
            None,
            [remote_smem_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_i32],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
            "r,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    # ------------------------------------------------------------------ #
    # Scalar cluster reductions (for RMSNorm sum, flash-decode max/sum)   #
    # ------------------------------------------------------------------ #

    @cute.jit
    def cluster_reduce_scalar_sum_mbarrier(
        vals_ptr,       # shared float32 array with cluster_size elements
        mbar_ptr,       # shared int64 mbarrier storage with 1 element
        local_val,      # cutlass.Float32 contribution from this CTA
        tidx,
        cluster_size: int,
    ):
        """Reduce one float32 scalar across a CTA cluster by summing.

        This uses inline PTX remote shared stores plus mbarrier completion
        tracking.  It is the currently validated path for sm_100a.
        """
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        if tidx == 0:
            for i in range(cluster_size):
                vals_ptr[i] = cutlass.Float32(0.0)
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, cluster_size * 4)

        if tidx < cluster_size:
            dst_ptr = vals_ptr + cta_rank
            _store_shared_remote_f32(local_val, dst_ptr, mbar_ptr, tidx)

        cute.arch.mbarrier_wait(mbar_ptr, phase=0)

        if tidx == 0:
            total = cutlass.Float32(0.0)
            for i in range(cluster_size):
                total = total + vals_ptr[i]
            vals_ptr[0] = total

        cute.arch.barrier()
        return vals_ptr[0]

    @cute.jit
    def cluster_reduce_scalar_max_mbarrier(
        vals_ptr,       # shared float32 array with cluster_size elements
        mbar_ptr,       # shared int64 mbarrier storage with 1 element
        local_val,      # cutlass.Float32 contribution from this CTA
        tidx,
        cluster_size: int,
    ):
        """Reduce one float32 scalar across a CTA cluster by taking max."""
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        if tidx == 0:
            for i in range(cluster_size):
                vals_ptr[i] = -cutlass.Float32.inf
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, cluster_size * 4)

        if tidx < cluster_size:
            dst_ptr = vals_ptr + cta_rank
            _store_shared_remote_f32(local_val, dst_ptr, mbar_ptr, tidx)

        cute.arch.mbarrier_wait(mbar_ptr, phase=0)

        if tidx == 0:
            max_val = -cutlass.Float32.inf
            for i in range(cluster_size):
                val = vals_ptr[i]
                max_val = val if max_val < val else max_val
            vals_ptr[0] = max_val

        cute.arch.barrier()
        return vals_ptr[0]

    @cute.jit
    def cluster_reduce_scalar_sum_atomic_inplace(
        smem_scalar_ptr,  # cute Pointer to __shared__ float32  (1 element)
        local_val,        # cutlass.Float32 – this CTA's contribution
        tidx,             # thread index (only tid==0 does the atomic)
        cluster_size: int,
    ):
        """Experimental high-level mapa+atomic scalar sum.

        This path currently reproduces an NVVM ICE on sm_100a and should stay
        out of production megakernels until the compiler path is fixed.
        """
        # Write local contribution then synchronise the cluster.
        smem_scalar_ptr[0] = local_val
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        for i in range(1, cluster_size):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            dst_cta = (cta_rank + i) % cluster_size
            # Map our smem_scalar_ptr into the dst_cta's address space.
            remote_ptr = cute.arch.mapa(smem_scalar_ptr, dst_cta)
            if tidx == 0:
                cute.arch.atomic_add(remote_ptr, local_val, scope="cluster")
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def cluster_reduce_scalar_max_inplace(
        smem_scalar_ptr,  # cute Pointer to __shared__ float32
        local_val,        # cutlass.Float32
        tidx,
        cluster_size: int,
    ):
        """Reduce a scalar float32 across all CTAs in the cluster by taking max."""
        smem_scalar_ptr[0] = local_val
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        for i in range(1, cluster_size):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            dst_cta = (cta_rank + i) % cluster_size
            remote_ptr = cute.arch.mapa(smem_scalar_ptr, dst_cta)
            if tidx == 0:
                cute.arch.atomic_max(remote_ptr, local_val, scope="cluster")
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    # ------------------------------------------------------------------ #
    # Vector cluster reductions (LINEAR / ATTN stages)                    #
    # ------------------------------------------------------------------ #

    @cute.jit
    def cluster_reduce_vector_sum_mbarrier(
        src_ptr,        # shared float32 array with num_floats elements
        recv_ptr,       # shared float32 array with cluster_size*num_floats elements
        mbar_ptr,       # shared int64 mbarrier storage with 1 element
        num_floats: int,
        tidx,
        cluster_size: int,
        num_threads: int,
    ):
        """Element-wise float32 sum across CTAs in a cluster.

        The reduced vector is written back into `src_ptr` on every CTA.
        This is a correctness-first LINEAR/ATTN reduction path.  It sends each
        CTA's local vector to every peer CTA using inline-PTX DSM stores and
        then sums the `cluster_size` received slots locally.
        """
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        if tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                cluster_size * num_floats * 4,
            )

        j = tidx
        while j < num_floats:
            local_val = src_ptr[j].to(cutlass.Float32)
            for peer_cta in range(cluster_size):
                dst_ptr = recv_ptr + cta_rank * num_floats + j
                _store_shared_remote_f32(
                    local_val,
                    dst_ptr,
                    mbar_ptr,
                    cutlass.Int32(peer_cta),
                )
            j = j + num_threads

        cute.arch.mbarrier_wait(mbar_ptr, phase=0)

        j = tidx
        while j < num_floats:
            total = cutlass.Float32(0.0)
            for peer_cta in range(cluster_size):
                total = total + recv_ptr[peer_cta * num_floats + j]
            src_ptr[j] = total
            j = j + num_threads

        cute.arch.barrier()

    @cute.jit
    def cluster_reduce_vector_add_inplace(
        src_ptr,        # Pointer to local SMEM vector (num_halfs elements)
        num_halfs: int, # total number of float16 elements
        tidx,
        cluster_size: int,
    ):
        """Accumulate a vector of fp16 values from all peer CTAs into src_ptr.

        Each thread accumulates a strided subset of elements from each peer's
        SMEM, serialised per peer.  After all cluster_size-1 peers have been
        visited, src_ptr contains the element-wise sum across the cluster.

        Corresponds to Stage::ATTN in dsm.cuh (or Stage::LINEAR for 3×HEAD_DIM).

        Layout:
          src_ptr[0 .. num_halfs-1] – fp16 elements, stride-1 layout.
        """
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        for i in range(1, cluster_size):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            dst_cta = (cta_rank + i) % cluster_size

            # Map a single element of src_ptr into the peer's address space.
            # We accumulate element-by-element.  Each thread handles a strided
            # subset: thread tidx handles indices tidx, tidx+num_threads, ...
            # (num_threads is passed implicitly via the loop bound).
            #
            # NOTE: We cannot do a bulk memcpy here without inline PTX, so we
            # do element-wise atomic adds on the remote pointer.  This is
            # correct but slower than a bulk path; it will be replaced later.
            j = tidx
            while j < num_halfs:
                elem_ptr = src_ptr + j          # pointer arithmetic on smem ptr
                remote_elem = cute.arch.mapa(elem_ptr, dst_cta)
                local_val = elem_ptr[0]         # read our own copy
                cute.arch.atomic_add(remote_elem, local_val, scope="cluster")
                j = j + cute.arch.block_dim()[0]  # stride by blockDim.x

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
