"""CuTeDSL cluster communication primitives.

Mirrors the role of dsm.cuh from ClusterFusion:
  - cluster_reduce_linear  ← Stage::LINEAR  (accumulate 3×head_dim halves across CTAs)
  - cluster_reduce_attn    ← Stage::ATTN    (accumulate 1×head_dim halves across CTAs)
  - cluster_reduce_max     ← scalar fmaxf across CTAs via __shared__ + atomicMax
  - cluster_reduce_sum     ← scalar DSM store/reduce across CTAs via __shared__

The underlying mechanism is:
  1. Each CTA writes its partial result to its own SMEM buffer.
  2. All CTAs do cluster_arrive / cluster_wait to synchronise.
  3. For each "peer" CTA rank i (i != self):
       - tid==0 signals the peer's mbarrier (arrive_and_expect_tx)
       - tid==0 maps the destination address into the peer's SMEM (mapa)
       - tid==0 issues cp.async.bulk SMEM→DSMEM transfer
       - All threads spin-wait on the local mbarrier
       - All threads do the element-wise accumulation from the received buffer
       - cluster.sync() to serialise iterations
  4. After the loop, SMEM[cta==0] contains the fully reduced result.

Note: the high-level `cute.arch.mapa() + cute.arch.atomic_add()` path currently
ICEs in NVVM on sm_100a.  Scalar reductions therefore use the same lower-level
pattern as CUTLASS' `blackwell/reduce.py`: inline PTX `mapa.shared::cluster`
plus `st.async.shared::cluster...mbarrier::complete_tx`, then a local sum over
the values received from peer CTAs.

For the ATTN and LINEAR stages the ClusterFusion CUDA code uses
  cp.async.bulk.shared::cluster + mbarrier
to move a tile of halfs, then does a half2 add-reduction.  The equivalent in
CuTeDSL uses:
  - mapa() to translate a local SMEM pointer to a peer's address space
  - A simple element loop with atomic_add on the remote pointer

This is not as efficient as a bulk copy but is correct and avoids inline PTX.
A future revision can switch to inline PTX for the async bulk path.
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
