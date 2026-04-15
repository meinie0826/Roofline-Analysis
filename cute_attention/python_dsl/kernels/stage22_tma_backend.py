import cuda.bindings.driver as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp

from .common import cutlass, cute
from .stage21_state_machine_backend import Stage21FlashAttentionStateMachine


class Stage22FlashAttentionTma(Stage21FlashAttentionStateMachine):
    """Independent stage22 backend with a real TMA producer path.

    This keeps the proven stage21 consumer math / softmax / epilogue flow, but
    replaces the K/V producer path with Hopper-style TMA loads coordinated
    through ``PipelineTmaAsync``.
    """

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
        if smem_usage > utils.get_smem_capacity_in_bytes("sm_90"):
            return False
        if (m_block_size * 2) % 128 != 0:
            return False
        return is_causal

    def _make_shared_storage_type(self, dtype, sQ_layout, sKV_layout):
        annotations = {
            "mainloop_pipeline_array_ptr": cute.struct.MemRange[cutlass.Int64, self._num_stages_kv],
            "sQ": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024],
            "sK0": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024],
            "sV0": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024],
            "sK1": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024],
            "sV1": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024],
        }
        if self._num_stages_kv >= 3:
            annotations["sK2"] = cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024
            ]
            annotations["sV2"] = cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024
            ]

        @cute.struct
        class SharedStorage:
            __annotations__ = annotations
        return SharedStorage

    def _make_mainloop_pipeline(self, barrier_storage, tx_count):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self._consumer_threads // 32
        )
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=barrier_storage,
            num_stages=self._num_stages_kv,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def _make_tma_atom(self, tensor: cute.Tensor, smem_layout: cute.Layout):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            (self._n_block_size, self._head_dim_padded),
        )

    def _partition_tma_slot(self, tma_atom, g_tensor, s_tensor):
        group_rank_smem = cute.rank(s_tensor) - 1
        group_rank_gmem = cute.rank(g_tensor) - 1
        return cpasync.tma_partition(
            tma_atom,
            cutlass.Int32(0),
            cute.make_layout((1,)),
            cute.group_modes(s_tensor, 0, group_rank_smem),
            cute.group_modes(g_tensor, 0, group_rank_gmem),
        )

    @cute.jit
    def _prefetch_tma_descriptors_once(self, warp_idx, tma_atom_k, tma_atom_v):
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

    @cute.jit
    def _make_producer_state(self):
        return pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._num_stages_kv
        )

    @cute.jit
    def _make_consumer_state_pair(self):
        return (
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self._num_stages_kv),
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self._num_stages_kv),
        )

    @cute.jit
    def _consumer_tma_try_wait(self, mainloop_pipeline, mainloop_consumer_state, k_tile_cnt: cutlass.Int32):
        peek_kv_full_status = cutlass.Boolean(1)
        if mainloop_consumer_state.count < k_tile_cnt:
            peek_kv_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_state)
        return peek_kv_full_status

    @cute.jit
    def _issue_tma_kv_load(
        self,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tVgV,
        tKsK0,
        tVsV0,
        tKsK1,
        tVsV1,
        tKsK2,
        tVsV2,
    ):
        tKsK = tKsK0
        tVsV = tVsV0
        if mainloop_producer_state.index == 1:
            tKsK = tKsK1
            tVsV = tVsV1
        if cutlass.const_expr(self._num_stages_kv >= 3):
            if mainloop_producer_state.index == 2:
                tKsK = tKsK2
                tVsV = tVsV2
        cute.copy(
            tma_atom_k,
            tKgK[None, mainloop_producer_state.count],
            tKsK,
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
        )
        cute.copy(
            tma_atom_v,
            tVgV[None, mainloop_producer_state.count],
            tVsV,
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
        )

    @cute.jit
    def _producer_tma_prefetch_prologue(
        self,
        warp_idx,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tVgV,
        tKsK0,
        tVsV0,
        tKsK1,
        tVsV1,
        tKsK2,
        tVsV2,
        k_tile_cnt: cutlass.Int32,
    ):
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self._num_stages_kv, k_tile_cnt), 0)
        if warp_idx == 0:
            for _ in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                self._issue_tma_kv_load(
                    mainloop_pipeline,
                    mainloop_producer_state,
                    tma_atom_k,
                    tma_atom_v,
                    tKgK,
                    tVgV,
                    tKsK0,
                    tVsV0,
                    tKsK1,
                    tVsV1,
                    tKsK2,
                    tVsV2,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()
        return mainloop_producer_state

    @cute.jit
    def _producer_tma_step(
        self,
        warp_idx,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tVgV,
        tKsK0,
        tVsV0,
        tKsK1,
        tVsV1,
        tKsK2,
        tVsV2,
        k_tile_cnt: cutlass.Int32,
    ):
        if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
            mainloop_pipeline.producer_acquire(mainloop_producer_state)
            self._issue_tma_kv_load(
                mainloop_pipeline,
                mainloop_producer_state,
                tma_atom_k,
                tma_atom_v,
                tKgK,
                tVgV,
                tKsK0,
                tVsV0,
                tKsK1,
                tVsV1,
                tKsK2,
                tVsV2,
            )
            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()
        return mainloop_producer_state

    @cute.jit
    def _consumer_compute_loaded_block(
        self,
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
        n_block: cutlass.Int32,
        is_first_n_block: cutlass.Boolean,
        in_mask_steps: cutlass.Boolean,
    ):
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

        SharedStorage = self._make_shared_storage_type(self._dtype, sQ_layout, sKV_layout)

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
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, consumer_layout, vQKV_layout)
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
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        is_producer = warp_idx < 4
        is_consumer = not is_producer
        consumer_slice_idx = tidx % self._consumer_threads
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size), n_block_max)

        gQ = cute.local_tile(mQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))

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

        mK_tma = mK[batch_size, None, num_head, None]
        mV_tma = mV[batch_size, None, num_head, None]
        tma_atom_k, tma_tensor_k = self._make_tma_atom(mK_tma, sKV_layout)
        tma_atom_v, tma_tensor_v = self._make_tma_atom(mV_tma, sKV_layout)
        gK_tma = cute.local_tile(tma_tensor_k, (self._n_block_size, self._head_dim_padded), (None, 0))
        gV_tma = cute.local_tile(tma_tensor_v, (self._n_block_size, self._head_dim_padded), (None, 0))
        tKsK0, tKgK0 = self._partition_tma_slot(tma_atom_k, gK_tma, sK0)
        tKsK1, _ = self._partition_tma_slot(tma_atom_k, gK_tma, sK1)
        tVsV0, tVgV0 = self._partition_tma_slot(tma_atom_v, gV_tma, sV0)
        tVsV1, _ = self._partition_tma_slot(tma_atom_v, gV_tma, sV1)
        tKsK2 = None
        tVsV2 = None
        if cutlass.const_expr(self._num_stages_kv >= 3):
            tKsK2, _ = self._partition_tma_slot(tma_atom_k, gK_tma, sK2)
            tVsV2, _ = self._partition_tma_slot(tma_atom_v, gV_tma, sV2)

        if is_consumer:
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                    cute.copy(gmem_tiled_copy_Q, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        k_stage_bytes = cute.size_in_bytes(self._dtype, cute.select(sKV_layout, mode=[0, 1]))
        mainloop_pipeline = self._make_mainloop_pipeline(
            storage.mainloop_pipeline_array_ptr.data_ptr(),
            k_stage_bytes * 2,
        )
        self._prefetch_tma_descriptors_once(warp_idx, tma_atom_k, tma_atom_v)
        mainloop_producer_state = self._make_producer_state()
        mainloop_consumer_read_state, mainloop_consumer_release_state = self._make_consumer_state_pair()

        k_tile_cnt = n_block_max
        mainloop_producer_state = self._producer_tma_prefetch_prologue(
            warp_idx,
            mainloop_pipeline,
            mainloop_producer_state,
            tma_atom_k,
            tma_atom_v,
            tKgK0,
            tVgV0,
            tKsK0,
            tVsV0,
            tKsK1,
            tVsV1,
            tKsK2,
            tVsV2,
            k_tile_cnt,
        )

        peek_kv_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_kv_full_status = self._consumer_tma_try_wait(
                mainloop_pipeline, mainloop_consumer_read_state, k_tile_cnt
            )

        for _ in cutlass.range(k_tile_cnt, unroll=1):
            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_kv_full_status)
            n_block = mainloop_consumer_read_state.count
            in_mask_steps = n_block == (n_block_max - 1)
            is_first_n_block = n_block == 0

            if is_consumer:
                stage_slot = mainloop_consumer_read_state.index
                tSrK = tSrK_slots[0]
                tOrVt = tOrVt_slots[0]
                tSsK = tSsK_slots[0]
                tSrK_view = tSrK_view_slots[0]
                tOsVt = tOsVt_slots[0]
                tOrVt_view = tOrVt_view_slots[0]
                if stage_slot == 1:
                    tSrK = tSrK_slots[1]
                    tOrVt = tOrVt_slots[1]
                    tSsK = tSsK_slots[1]
                    tSrK_view = tSrK_view_slots[1]
                    tOsVt = tOsVt_slots[1]
                    tOrVt_view = tOrVt_view_slots[1]
                if cutlass.const_expr(self._num_stages_kv >= 3):
                    if stage_slot == 2:
                        tSrK = tSrK_slots[2]
                        tOrVt = tOrVt_slots[2]
                        tSsK = tSsK_slots[2]
                        tSrK_view = tSrK_view_slots[2]
                        tOsVt = tOsVt_slots[2]
                        tOrVt_view = tOrVt_view_slots[2]
                self._consumer_compute_loaded_block(
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
                    softmax_scale_log2,
                    mQ,
                    mK,
                    batch_size,
                    num_head,
                    m_block,
                    n_block,
                    is_first_n_block,
                    in_mask_steps,
                )

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()
            peek_kv_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_kv_full_status = self._consumer_tma_try_wait(
                    mainloop_pipeline, mainloop_consumer_read_state, k_tile_cnt
                )
            mainloop_producer_state = self._producer_tma_step(
                warp_idx,
                mainloop_pipeline,
                mainloop_producer_state,
                tma_atom_k,
                tma_atom_v,
                tKgK0,
                tVgV0,
                tKsK0,
                tVsV0,
                tKsK1,
                tVsV1,
                tKsK2,
                tVsV2,
                k_tile_cnt,
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
