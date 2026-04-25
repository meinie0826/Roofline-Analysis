"""CuTeDSL cluster communication primitives.

Mirrors the role of dsm.cuh from ClusterFusion:
  - cluster_reduce_linear  ← Stage::LINEAR  (accumulate 3×head_dim halves across CTAs)
  - cluster_reduce_attn    ← Stage::ATTN    (accumulate 1×head_dim halves across CTAs)
  - cluster_reduce_max     ← scalar fmaxf across CTAs via __shared__ + atomicMax
  - cluster_reduce_sum     ← scalar atomicAdd across CTAs via __shared__

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

Note: CuTeDSL does not yet expose cp.async.bulk.shared::cluster directly as a
Python function.  We therefore fall back to an equivalent pattern that *is*
available: for each pair we use mapa to get the peer's shared-memory address
and perform the add-reduction via register loads (shuffle-based inter-warp
reduction is already inside the kernel; the inter-CTA part here uses
atomic_add on the mapped pointer).

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
    # ------------------------------------------------------------------ #
    # Scalar cluster reductions (for RMSNorm sum, flash-decode max/sum)   #
    # ------------------------------------------------------------------ #

    def cluster_reduce_scalar_sum_inplace(
        smem_scalar_ptr,  # cute Pointer to __shared__ float32  (1 element)
        local_val,        # cutlass.Float32 – this CTA's contribution
        tidx,             # thread index (only tid==0 does the atomic)
        cluster_size: int,
    ):
        """Reduce a scalar float32 across all CTAs in the cluster by summing.

        Mirrors the ClusterFusion pattern:
          cluster_local_sum = local_sum; cluster.sync();
          for i in 1..cluster_size-1:
              dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
              atomicAdd(dst_shmem, local_sum); cluster.sync();

        After the call, smem_scalar_ptr[0] contains the cluster-wide sum on
        every CTA.
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
