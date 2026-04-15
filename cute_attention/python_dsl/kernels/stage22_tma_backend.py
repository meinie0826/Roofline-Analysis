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

        shared_storage = self._make_shared_storage_type(self._dtype, sQ_layout, sKV_layout_staged)
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        tx_count = cute.size_in_bytes(cute.slice_(sKV_layout_staged, (None, None, 0)))
        _ = self._make_mainloop_pipeline(mainloop_pipeline_array_ptr, tx_count)

        gK = cute.local_tile(mK[0, None, 0, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(mV[0, None, 0, None], (self._n_block_size, self._head_dim_padded), (None, 0))
        sK = storage.sK.get_tensor(sKV_layout_staged)
        sV = storage.sV.get_tensor(sKV_layout_staged)
        tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v = self._make_tma_kv_atoms_and_tensors(
            gK, gV, sKV_layout_staged
        )
        _ = tma_tensor_k
        _ = tma_tensor_v
        _ = self._partition_tma_kv(tma_atom_k, tma_tensor_k, sK, gK)
        _ = self._partition_tma_kv(tma_atom_v, tma_tensor_v, sV, gV)

        raise NotImplementedError(
            "stage22_tma now has real TMA atom/pipeline scaffolding, but the K/V mainloop load path is not wired into a runnable kernel yet."
        )
