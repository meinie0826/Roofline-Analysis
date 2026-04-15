import cuda.bindings.driver as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp

from .common import cutlass, cute


class Stage22FlashAttentionRewrite:
    """Independent SM90-oriented producer/consumer state-machine backend for stage22.

    Current support is intentionally narrow and explicit:
      - 256 threads per CTA
      - 128 producer threads + 128 consumer threads
      - num_stages_kv in {2, 3}
    """

    def __init__(
        self,
        head_dim: int,
        m_block_size: int,
        n_block_size: int,
        num_threads: int,
        num_stages_kv: int,
        is_causal: bool,
    ):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._num_stages_kv = num_stages_kv
        self._is_causal = is_causal
        self._producer_threads = 128
        self._consumer_threads = 128
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

    def _steady_state_wait_groups(self) -> int:
        # Stage21 keeps stage18's tuned wait policy but exposes it through
        # explicit producer/consumer state helpers so we can evolve the
        # schedule without rewriting the whole mainloop again.
        # For a 2-stage pipeline we wait
        # more aggressively so consumers do not keep an extra async group alive.
        # For a 3-stage pipeline we retain one in-flight group in steady state.
        return 1 if self._num_stages_kv >= 3 else 0

    def _steady_state_step(self) -> int:
        return self._num_stages_kv

    def _prefetch_block_for_slot(self, first_pipeline_block, prefetch_slot):
        return first_pipeline_block - prefetch_slot

    def _compute_block_for_slot(self, n_block_max, n_tile, stage_slot):
        return n_block_max - n_tile - 1 - stage_slot

    def _next_block_after_compute(self, n_block):
        return n_block - self._steady_state_step()

    def _producer_prefetch_block(self, first_pipeline_block, prefetch_slot):
        return self._prefetch_block_for_slot(first_pipeline_block, prefetch_slot)

    def _consumer_compute_block(self, n_block_max, n_tile, stage_slot):
        return self._compute_block_for_slot(n_block_max, n_tile, stage_slot)

    def _consumer_wait_groups(self) -> int:
        return self._steady_state_wait_groups()

    @cute.jit
    def _producer_prefetch_slot(
        self,
        first_pipeline_block: cutlass.Int32,
        prefetch_slot: cutlass.Constexpr,
        is_producer,
        tKVcKV,
        tKVpKV,
        tKgK,
        tVgV,
        tKsK,
        tVsV,
        gmem_tiled_copy_KV,
        mK: cute.Tensor,
    ):
        prefetch_block = self._producer_prefetch_block(first_pipeline_block, prefetch_slot)
        if is_producer and prefetch_block >= 0:
            self._copy_kv_tile(
                prefetch_block,
                first_pipeline_block,
                tKVcKV,
                tKVpKV,
                tKgK,
                tVgV,
                tKsK,
                tVsV,
                gmem_tiled_copy_KV,
                mK,
            )

    @cute.jit
    def _producer_advance_state(
        self,
        is_producer,
        next_k_block: cutlass.Int32,
        n_block: cutlass.Int32,
        tKVcKV,
        tKVpKV,
        tKgK,
        tVgV,
        tKsK,
        tVsV,
        gmem_tiled_copy_KV,
        mK: cute.Tensor,
    ):
        if is_producer and next_k_block >= 0:
            self._copy_kv_tile(next_k_block, n_block, tKVcKV, tKVpKV, tKgK, tVgV, tKsK, tVsV, gmem_tiled_copy_KV, mK)
        elif is_producer:
            cute.arch.cp_async_commit_group()

    @cute.jit
    def _consumer_advance_state(self, in_mask_steps: cutlass.Constexpr):
        if cutlass.const_expr(in_mask_steps):
            cute.arch.cp_async_wait_group(0)
        elif cutlass.const_expr(self._num_stages_kv >= 3):
            cute.arch.cp_async_wait_group(self._consumer_wait_groups())
        else:
            cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

    @staticmethod
    def can_implement(
        dtype,
        head_dim: int,
        m_block_size: int,
        n_block_size: int,
        num_threads: int,
        num_stages_kv: int,
        is_causal: bool,
    ) -> bool:
        if dtype != cutlass.Float16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads != 256:
            return False
        if num_stages_kv not in {2, 3}:
            return False
        if n_block_size % 64 != 0:
            return False
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2 * num_stages_kv) * 2
        if smem_usage > utils.get_smem_capacity_in_bytes("sm_80"):
            return False
        if (m_block_size * 2) % 128 != 0:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(mQ.element_type != cutlass.Float16):
            raise TypeError("Only Float16 is supported")

        self._dtype = mQ.element_type
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )
        sO_layout = sQ_layout

        shared_annotations = {
            "sQ": cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024],
            "sK0": cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024],
            "sV0": cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024],
            "sK1": cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024],
            "sV1": cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024],
        }
        if cutlass.const_expr(self._num_stages_kv >= 3):
            shared_annotations["sK2"] = cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            shared_annotations["sV2"] = cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]

        @cute.struct
        class SharedStorage:
            __annotations__ = shared_annotations

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        consumer_layout = cute.make_layout(
            (self._consumer_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        producer_layout = cute.make_layout(
            (self._producer_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, consumer_layout, vQKV_layout)
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, producer_layout, vQKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, consumer_layout, vQKV_layout)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._consumer_threads // 32, 1, 1),
            permutation_mnk=(self._consumer_threads // 32 * 16, 16, 16),
        )

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_KV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_KV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        is_producer = warp_idx < 4
        is_consumer = not is_producer
        consumer_slice_idx = tidx % self._consumer_threads
        producer_slice_idx = tidx % self._producer_threads
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size), n_block_max)
        start_n_block = n_block_max - 1

        gQ = cute.local_tile(mQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gK = cute.local_tile(mK[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(mV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK0 = storage.sK0.get_tensor(sKV_layout)
        sV0 = storage.sV0.get_tensor(sKV_layout)
        sK1 = storage.sK1.get_tensor(sKV_layout)
        sV1 = storage.sV1.get_tensor(sKV_layout)
        sK2 = storage.sK2.get_tensor(sKV_layout) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        sV2 = storage.sV2.get_tensor(sKV_layout) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        sVt0 = cute.composition(sV0, cute.make_layout((self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))
        sVt1 = cute.composition(sV1, cute.make_layout((self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))
        sVt2 = (
            cute.composition(sV2, cute.make_layout((self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))
            if cutlass.const_expr(self._num_stages_kv >= 3)
            else None
        )

        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(consumer_slice_idx)
        tQgQ = gmem_thr_copy_Q.partition_S(gQ)
        tQsQ = gmem_thr_copy_Q.partition_D(sQ)
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        cQ = cute.local_tile(mcQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
        tQcQ = gmem_thr_copy_Q.partition_S(cQ)
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3])

        thr_mma = tiled_mma.get_slice(consumer_slice_idx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK0 = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
        tSrK1 = thr_mma.make_fragment_B(thr_mma.partition_B(sK1))
        tSrK2 = thr_mma.make_fragment_B(thr_mma.partition_B(sK2)) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tOrVt0 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        tOrVt1 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt1))
        tOrVt2 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt2)) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tSrK_slots = (tSrK0, tSrK1, tSrK2)
        tOrVt_slots = (tOrVt0, tOrVt1, tOrVt2)
        acc_shape_O = thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        smem_copy_atom_Q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(consumer_slice_idx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(consumer_slice_idx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(consumer_slice_idx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK0 = smem_thr_copy_K.partition_S(sK0)
        tSsK1 = smem_thr_copy_K.partition_S(sK1)
        tSsK2 = smem_thr_copy_K.partition_S(sK2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tSrK0_view = smem_thr_copy_K.retile(tSrK0)
        tSrK1_view = smem_thr_copy_K.retile(tSrK1)
        tSrK2_view = smem_thr_copy_K.retile(tSrK2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tOsVt0 = smem_thr_copy_V.partition_S(sVt0)
        tOsVt1 = smem_thr_copy_V.partition_S(sVt1)
        tOsVt2 = smem_thr_copy_V.partition_S(sVt2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tOrVt0_view = smem_thr_copy_V.retile(tOrVt0)
        tOrVt1_view = smem_thr_copy_V.retile(tOrVt1)
        tOrVt2_view = smem_thr_copy_V.retile(tOrVt2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tSsK_slots = (tSsK0, tSsK1, tSsK2)
        tSrK_view_slots = (tSrK0_view, tSrK1_view, tSrK2_view)
        tOsVt_slots = (tOsVt0, tOsVt1, tOsVt2)
        tOrVt_view_slots = (tOrVt0_view, tOrVt1_view, tOrVt2_view)

        row_max = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(producer_slice_idx)
        tKgK = gmem_thr_copy_KV.partition_S(gK)
        tVgV = gmem_thr_copy_KV.partition_S(gV)
        tKsK0 = gmem_thr_copy_KV.partition_D(sK0)
        tVsV0 = gmem_thr_copy_KV.partition_D(sV0)
        tKsK1 = gmem_thr_copy_KV.partition_D(sK1)
        tVsV1 = gmem_thr_copy_KV.partition_D(sV1)
        tKsK2 = gmem_thr_copy_KV.partition_D(sK2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tVsV2 = gmem_thr_copy_KV.partition_D(sV2) if cutlass.const_expr(self._num_stages_kv >= 3) else None
        tKsK_slots = (tKsK0, tKsK1, tKsK2)
        tVsV_slots = (tVsV0, tVsV1, tVsV2)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cKV = cute.local_tile(mcKV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (start_n_block, 0))
        tKVcKV = gmem_thr_copy_KV.partition_S(cKV)
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (tKsK0.shape[0][1], cute.size(tKsK0, mode=[1]), cute.size(tKsK0, mode=[2])),
                stride=(cute.size(tKsK0, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3])

        if is_consumer:
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                    cute.copy(gmem_tiled_copy_Q, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            cute.arch.cp_async_commit_group()
        else:
            self._copy_kv_tile(start_n_block, start_n_block, tKVcKV, tKVpKV, tKgK, tVgV, tKsK0, tVsV0, gmem_tiled_copy_KV, mK)

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        first_pipeline_block = n_block_max - mask_steps - 1
        for prefetch_slot in cutlass.range_constexpr(1, self._num_stages_kv):
            self._producer_prefetch_slot(
                first_pipeline_block,
                prefetch_slot,
                is_producer,
                tKVcKV,
                tKVpKV,
                tKgK,
                tVgV,
                tKsK_slots[prefetch_slot],
                tVsV_slots[prefetch_slot],
                gmem_tiled_copy_KV,
                mK,
            )

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            self._compute_one_block(
                is_consumer,
                is_producer,
                n_block,
                thr_mma,
                tiled_mma,
                tSrQ,
                tSrK0,
                tOrVt0,
                acc_O,
                smem_tiled_copy_Q,
                smem_tiled_copy_K,
                smem_tiled_copy_V,
                tSsQ,
                tSrQ_view,
                tSsK0,
                tSrK0_view,
                tOsVt0,
                tOrVt0_view,
                row_max,
                row_sum,
                softmax_scale_log2,
                mQ,
                mK,
                batch_size,
                num_head,
                m_block,
                is_first_n_block=(n_tile == 0),
                in_mask_steps=True,
                gmem_tiled_copy_KV=gmem_tiled_copy_KV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tVgV=tVgV,
                tKsK=tKsK0,
                tVsV=tVsV0,
                tKVpKV=tKVpKV,
                next_k_block=n_block - 1,
            )

        for n_tile in range(mask_steps, n_block_max, self._steady_state_step()):
            for stage_slot in cutlass.range_constexpr(self._num_stages_kv):
                n_block = self._consumer_compute_block(n_block_max, n_tile, stage_slot)
                if n_block >= 0:
                    self._compute_one_block(
                        is_consumer,
                        is_producer,
                        n_block,
                        thr_mma,
                        tiled_mma,
                        tSrQ,
                        tSrK_slots[stage_slot],
                        tOrVt_slots[stage_slot],
                        acc_O,
                        smem_tiled_copy_Q,
                        smem_tiled_copy_K,
                        smem_tiled_copy_V,
                        tSsQ,
                        tSrQ_view,
                        tSsK_slots[stage_slot],
                        tSrK_view_slots[stage_slot],
                        tOsVt_slots[stage_slot],
                        tOrVt_view_slots[stage_slot],
                        row_max,
                        row_sum,
                        softmax_scale_log2,
                        mQ,
                        mK,
                        batch_size,
                        num_head,
                        m_block,
                        is_first_n_block=False,
                        in_mask_steps=False,
                        gmem_tiled_copy_KV=gmem_tiled_copy_KV,
                        tKVcKV=tKVcKV,
                        tKgK=tKgK,
                        tVgV=tVgV,
                        tKsK=tKsK_slots[stage_slot],
                        tVsV=tVsV_slots[stage_slot],
                        tKVpKV=tKVpKV,
                        next_k_block=self._next_block_after_compute(n_block),
                    )

        if is_consumer:
            self.normalize_softmax(acc_O, row_sum)
            rO = cute.make_fragment_like(acc_O, self._dtype)
            rO.store(acc_O.load().to(self._dtype))
            sO = cute.make_tensor(sQ.iterator, sO_layout)

            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(consumer_slice_idx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            gO = cute.local_tile(mO[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(consumer_slice_idx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self._dtype)
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            mcO = cute.make_identity_tensor(mO.layout.shape)
            cO = cute.local_tile(mcO[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
            tOcO = gmem_thr_copy_O.partition_D(cO)
            tOpO = cute.make_rmem_tensor(
                cute.make_layout((tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]), stride=(tOgO.shape[2], 0, 1)),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
                for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                    tOpO[rest_v, 0, rest_n] = cute.elem_less(tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3])
            for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
                if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                    cute.copy(gmem_tiled_copy_O, tOrO[None, rest_m, None], tOgO[None, rest_m, None], pred=tOpO[None, rest_m, None])

    @cute.jit
    def _compute_one_block(
        self,
        is_consumer,
        is_producer,
        n_block,
        thr_mma,
        tiled_mma,
        tSrQ,
        tSrK,
        tOrVt,
        acc_O,
        smem_tiled_copy_Q,
        smem_tiled_copy_K,
        smem_tiled_copy_V,
        tSsQ,
        tSrQ_view,
        tSsK,
        tSrK_view,
        tOsVt,
        tOrVt_view,
        row_max,
        row_sum,
        softmax_scale_log2: cutlass.Float32,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        batch_size: cutlass.Int32,
        num_head: cutlass.Int32,
        m_block: cutlass.Int32,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
        gmem_tiled_copy_KV,
        tKVcKV,
        tKgK,
        tVgV,
        tKsK,
        tVsV,
        tKVpKV,
        next_k_block: cutlass.Int32,
    ):
        if is_consumer:
            acc_shape_S = thr_mma.partition_shape_C((self._m_block_size, self._n_block_size))
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_S.fill(0.0)

            cute.copy(smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
            cute.copy(smem_tiled_copy_K, tSsK[None, None, 0], tSrK_view[None, None, 0])
            for k_idx in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                k_next = (k_idx + 1) % cute.size(tSsQ.shape[2])
                cute.copy(smem_tiled_copy_Q, tSsQ[None, None, k_next], tSrQ_view[None, None, k_next])
                cute.copy(smem_tiled_copy_K, tSsK[None, None, k_next], tSrK_view[None, None, k_next])
                cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k_idx], tSrK[None, None, k_idx], acc_S)

            self.softmax_rescale_O(
                acc_S,
                acc_O,
                row_max,
                row_sum,
                softmax_scale_log2,
                mQ,
                mK,
                batch_size,
                num_head,
                m_block,
                n_block,
                is_first_n_block=is_first_n_block,
                in_mask_steps=in_mask_steps,
                thr_mma=thr_mma,
            )

            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                (
                    (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                    rP_layout_divided.shape[1],
                    rP_layout_divided.shape[2][1],
                ),
                stride=(
                    (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                    rP_layout_divided.stride[1],
                    rP_layout_divided.stride[2][1],
                ),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
            cute.copy(smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0])
            for k_idx in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k_idx + 1) % cute.size(tOrS.shape[2])
                cute.copy(smem_tiled_copy_V, tOsVt[None, None, k_next], tOrVt_view[None, None, k_next])
                cute.gemm(tiled_mma, acc_O, tOrS[None, None, k_idx], tOrVt[None, None, k_idx], acc_O)

        self.cta_sync_barrier.arrive_and_wait()

        self._producer_advance_state(
            is_producer,
            next_k_block,
            n_block,
            tKVcKV,
            tKVpKV,
            tKgK,
            tVgV,
            tKsK,
            tVsV,
            gmem_tiled_copy_KV,
            mK,
        )
        self._consumer_advance_state(in_mask_steps)

    @cute.jit
    def _copy_kv_tile(
        self,
        tile_block: cutlass.Int32,
        predicate_block: cutlass.Int32,
        tKVcKV,
        tKVpKV,
        tKgK,
        tVgV,
        tKsK,
        tVsV,
        gmem_tiled_copy_KV,
        mK: cute.Tensor,
    ):
        block_offset = (tile_block - predicate_block) * self._n_block_size
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            row_idx = tKVcKV[0, n, 0][1] + block_offset
            if cute.elem_less(row_idx, mK.layout.shape[1]):
                cute.copy(gmem_tiled_copy_KV, tKgK[None, n, None, tile_block], tKsK[None, n, None], pred=tKVpKV[None, n, None])
                cute.copy(gmem_tiled_copy_KV, tVgV[None, n, None, tile_block], tVsV[None, n, None], pred=tKVpKV[None, n, None])
            else:
                tKsK[None, n, None].fill(0)
                tVsV[None, n, None].fill(0)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def softmax_rescale_O(
        self,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        batch_size: cutlass.Int32,
        num_head: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
        thr_mma: cute.TiledMma,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(row_max, cutlass.Float32)
            cute.basic_copy(row_max, row_max_prev)
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor((mQ.shape[0], mQ.shape[1], mQ.shape[2], mK.shape[1]))
            cS = cute.local_tile(mcS[batch_size, None, num_head, None], (self._m_block_size, self._n_block_size), (m_block, n_block))
            tScS = thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        for r in cutlass.range_constexpr(cute.size(row_max)):
            if cutlass.const_expr(in_mask_steps):
                col_idx_limit = cutlass.min(tScS_mn[r, 0][1] + 1, mK.shape[1])
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = 0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_scale_log2 - row_max_cur_row * softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_scale_log2 - row_max_cur_row * softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = acc_S_row_sum + row_sum[r] * prev_minus_cur_exp
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            row_max[r] = row_max_cur_row
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)
