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
        # The backend is intentionally not launched yet. We still rely on the
        # stage21 cp.async path for correctness/performance while stage22 grows
        # real TMA load plumbing.
        raise NotImplementedError(
            "stage22_tma backend scaffold is not wired into a runnable kernel yet."
        )
