import cuda.bindings.driver as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp

from .common import cutlass, cute


class Stage22FlashAttentionTmaExperimental:
    """Early stage22 TMA backend scaffold.

    This file is intentionally not wired into the public stage registry yet.
    The goal of this first step is to land real Hopper-style TMA building
    blocks we can iterate on safely:

      - TMA atom / tensor construction
      - TMA partitioning for staged K/V shared memory
      - PipelineTmaAsync setup helpers

    We keep the supported envelope narrow so later iterations can replace the
    cp.async producer path incrementally rather than all at once.
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
        return True

    @staticmethod
    def _make_tma_atom_and_tensor(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor

    def _make_mainloop_pipeline(self, barrier_storage, tx_count):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
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

    def _make_kv_layouts(self):
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
        sKV_layout_staged = cute.tile_to_shape(
            sQ_layout_atom,
            (self._n_block_size, self._head_dim_padded, self._num_stages_kv),
            (0, 1, 2),
        )
        return sQ_layout, sKV_layout, sKV_layout_staged

    def _make_q_layout(self):
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        return cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

    def _make_shared_storage_type(self, dtype, sQ_layout, sKV_layout_staged):
        @cute.struct
        class SharedStorage:
            __annotations__ = {
                "mainloop_pipeline_array_ptr": cute.struct.MemRange[cutlass.Int64, self._num_stages_kv],
                "sQ": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024],
                "sK": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout_staged)], 1024],
                "sV": cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sKV_layout_staged)], 1024],
            }

        return SharedStorage

    def _make_consumer_copy_atoms_and_mma(self):
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
        smem_copy_atom_Q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._consumer_threads // 32, 1, 1),
            permutation_mnk=(self._consumer_threads // 32 * 16, 16, 16),
        )
        return atom_async_copy, atom_universal_copy, smem_copy_atom_Q, smem_copy_atom_K, smem_copy_atom_V, tiled_mma, async_copy_elems

    @cute.jit
    def _load_q_to_smem(
        self,
        tidx,
        mQ: cute.Tensor,
        sQ,
        sQ_layout_atom,
        m_block,
        batch_size,
        num_head,
        atom_async_copy,
    ):
        async_copy_elems = 128 // self._dtype.width
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        consumer_layout = cute.make_layout(
            (self._consumer_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, consumer_layout, vQKV_layout)
        consumer_slice_idx = tidx % self._consumer_threads
        gQ = cute.local_tile(mQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
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
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(gmem_tiled_copy_Q, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        cute.arch.cp_async_commit_group()
        return gmem_tiled_copy_Q

    @cute.jit
    def _build_consumer_mma_views(
        self,
        tidx,
        tiled_mma,
        sQ,
        sKV_layout_staged,
        sK,
        sV,
        smem_copy_atom_Q,
        smem_copy_atom_K,
        smem_copy_atom_V,
    ):
        consumer_slice_idx = tidx % self._consumer_threads
        thr_mma = tiled_mma.get_slice(consumer_slice_idx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        sVt = cute.composition(sV, cute.make_layout((self._head_dim_padded, self._n_block_size, self._num_stages_kv), stride=(self._n_block_size, 1, self._n_block_size * self._head_dim_padded)))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(consumer_slice_idx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(consumer_slice_idx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(consumer_slice_idx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_view = smem_thr_copy_V.retile(tOrVt)
        return (
            thr_mma,
            tSrQ,
            tSrK,
            tOrVt,
            smem_tiled_copy_Q,
            smem_tiled_copy_K,
            smem_tiled_copy_V,
            tSsQ,
            tSrQ_view,
            tSsK,
            tSrK_view,
            tOsVt,
            tOrVt_view,
        )

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

    def _make_tma_kv_atoms_and_tensors(self, gK: cute.Tensor, gV: cute.Tensor, sKV_layout_staged):
        tma_tile = (self._n_block_size, self._head_dim_padded)
        tma_atom_k, tma_tensor_k = self._make_tma_atom_and_tensor(gK, sKV_layout_staged, tma_tile)
        tma_atom_v, tma_tensor_v = self._make_tma_atom_and_tensor(gV, sKV_layout_staged, tma_tile)
        return tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v

    def _partition_tma_kv(self, tma_atom, tma_tensor, sTensor, gTensor):
        s_for_tma_partition = cute.group_modes(sTensor, 0, 2)
        g_for_tma_partition = cute.group_modes(gTensor, 0, 2)
        cta_layout = cute.make_layout((1,))
        cta_coord = 0
        return cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            cta_coord,
            cta_layout,
            s_for_tma_partition,
            g_for_tma_partition,
        )

    @cute.jit
    def _tma_prefetch_descriptors(self, tma_atom_k, tma_atom_v, warp_idx):
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

    @cute.jit
    def _make_producer_state(self):
        return pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._num_stages_kv
        )

    @cute.jit
    def _make_consumer_state(self):
        return pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_stages_kv
        )

    @cute.jit
    def _prefetch_tma_descriptors_once(self, warp_idx, tma_atom_k, tma_atom_v):
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

    @cute.jit
    def _issue_tma_kv_load(
        self,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tKsK,
        tVgV,
        tVsV,
    ):
        cute.copy(
            tma_atom_k,
            tKgK[(None, mainloop_producer_state.count)],
            tKsK[(None, mainloop_producer_state.index)],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
        )
        cute.copy(
            tma_atom_v,
            tVgV[(None, mainloop_producer_state.count)],
            tVsV[(None, mainloop_producer_state.index)],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
        )

    @cute.jit
    def _producer_tma_step(
        self,
        warp_idx,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tKsK,
        tVgV,
        tVsV,
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
                tKsK,
                tVgV,
                tVsV,
            )
            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()
        return mainloop_producer_state

    @cute.jit
    def _consumer_tma_try_wait(
        self,
        mainloop_pipeline,
        mainloop_consumer_state,
        k_tile_cnt: cutlass.Int32,
    ):
        peek_kv_full_status = cutlass.Boolean(1)
        if mainloop_consumer_state.count < k_tile_cnt:
            peek_kv_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_state)
        return peek_kv_full_status

    @cute.jit
    def _producer_tma_prefetch_prologue(
        self,
        warp_idx,
        mainloop_pipeline,
        mainloop_producer_state,
        tma_atom_k,
        tma_atom_v,
        tKgK,
        tKsK,
        tVgV,
        tVsV,
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
                    tKsK,
                    tVgV,
                    tVsV,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()
        return mainloop_producer_state

    @cute.jit
    def _make_consumer_state_pair(self):
        return (
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self._num_stages_kv),
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self._num_stages_kv),
        )

    @cute.jit
    def _consumer_tma_advance_states(self, read_state, release_state):
        read_state.advance()
        release_state.advance()
        return read_state, release_state

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
        sQ_layout, sKV_layout, sKV_layout_staged = self._make_kv_layouts()
        _ = sKV_layout  # carried for parity with stage21 shape plumbing
        q_layout = self._make_q_layout()

        shared_storage = self._make_shared_storage_type(self._dtype, sQ_layout, sKV_layout_staged)
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        tx_count = cute.size_in_bytes(cute.slice_(sKV_layout_staged, (None, None, 0)))
        mainloop_pipeline = self._make_mainloop_pipeline(mainloop_pipeline_array_ptr, tx_count)

        (
            atom_async_copy,
            atom_universal_copy,
            smem_copy_atom_Q,
            smem_copy_atom_K,
            smem_copy_atom_V,
            tiled_mma,
            _async_copy_elems,
        ) = self._make_consumer_copy_atoms_and_mma()
        _ = atom_universal_copy

        gK = cute.local_tile(mK[0, None, 0, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(mV[0, None, 0, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout_staged)
        sV = storage.sV.get_tensor(sKV_layout_staged)
        tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v = self._make_tma_kv_atoms_and_tensors(
            gK, gV, sKV_layout_staged
        )
        tKsK, tKgK = self._partition_tma_kv(tma_atom_k, tma_tensor_k, sK, gK)
        tVsV, tVgV = self._partition_tma_kv(tma_atom_v, tma_tensor_v, sV, gV)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        _ = self._load_q_to_smem(
            tidx,
            mQ,
            sQ,
            q_layout,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            atom_async_copy,
        )
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        self._prefetch_tma_descriptors_once(warp_idx, tma_atom_k, tma_atom_v)
        mainloop_producer_state = self._make_producer_state()
        mainloop_consumer_state = self._make_consumer_state()
        mainloop_consumer_read_state, mainloop_consumer_release_state = self._make_consumer_state_pair()
        k_tile_cnt = cute.size(tKgK, mode=[1])
        _ = self._consumer_tma_try_wait(mainloop_pipeline, mainloop_consumer_state, k_tile_cnt)
        _ = self._producer_tma_prefetch_prologue(
            warp_idx,
            mainloop_pipeline,
            mainloop_producer_state,
            tma_atom_k,
            tma_atom_v,
            tKgK,
            tKsK,
            tVgV,
            tVsV,
            k_tile_cnt,
        )
        _ = self._consumer_tma_try_wait(mainloop_pipeline, mainloop_consumer_read_state, k_tile_cnt)
        _ = self._consumer_tma_advance_states(mainloop_consumer_read_state, mainloop_consumer_release_state)
        _ = self._build_consumer_mma_views(
            tidx,
            tiled_mma,
            sQ,
            sKV_layout_staged,
            sK,
            sV,
            smem_copy_atom_Q,
            smem_copy_atom_K,
            smem_copy_atom_V,
        )

        raise NotImplementedError(
            "stage22_tma now wires Q-side consumer setup together with the TMA producer pipeline, but the K/V TMA mainloop compute path is not yet runnable."
        )
