import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp

from .common import (
    AttentionConfig,
    HAS_CUTE,
    cutlass,
    cute,
    from_dlpack,
    require_torch,
    torch,
    validate_qkv,
)
MAX_SEQ_LEN_FOR_STAGE12_CUTE = 4096
_STAGE12_COMPILED_CACHE = {}
_STAGE12_AUTOTUNE_CACHE = {}
_STAGE12_AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "stage12_autotune.json"


if HAS_CUTE:
    class Stage12FlashAttentionAmpere:
        def __init__(self, head_dim: int, m_block_size: int, n_block_size: int, num_threads: int, is_causal: bool):
            self._head_dim = head_dim
            self._m_block_size = m_block_size
            self._n_block_size = n_block_size
            self._head_dim_padded = (head_dim + 31) // 32 * 32
            self._num_threads = num_threads
            self._is_causal = is_causal
            self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

        @staticmethod
        def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal) -> bool:
            if dtype != cutlass.Float16:
                return False
            if head_dim % 8 != 0:
                return False
            if num_threads % 32 != 0:
                return False
            smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 4) * 2
            if smem_usage > utils.get_smem_capacity_in_bytes("sm_80"):
                return False
            if (m_block_size * 2) % num_threads != 0:
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

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
                sK0: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
                sK1: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
                sV0: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
                sV1: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]

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
            tQKV_layout = cute.make_layout(
                (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
                stride=(tQKV_shape_dim_1, 1),
            )
            vQKV_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy, tQKV_layout, vQKV_layout)
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)

            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
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
                gmem_tiled_copy_QKV,
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
            gmem_tiled_copy_QKV: cute.TiledCopy,
            gmem_tiled_copy_O: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            m_block, batch_size, num_head = cute.arch.block_idx()

            n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
            if self._is_causal:
                n_block_max = min(cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size), n_block_max)
            n_block = n_block_max - 1

            gQ = cute.local_tile(mQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
            gK = cute.local_tile(mK[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))
            gV = cute.local_tile(mV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (None, 0))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK0 = storage.sK0.get_tensor(sKV_layout)
            sK1 = storage.sK1.get_tensor(sKV_layout)
            sV0 = storage.sV0.get_tensor(sKV_layout)
            sV1 = storage.sV1.get_tensor(sKV_layout)
            sVt0 = cute.composition(
                sV0,
                cute.make_layout(
                    (self._head_dim_padded, self._n_block_size),
                    stride=(self._n_block_size, 1),
                ),
            )
            sVt1 = cute.composition(
                sV1,
                cute.make_layout(
                    (self._head_dim_padded, self._n_block_size),
                    stride=(self._n_block_size, 1),
                ),
            )

            gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
            tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
            tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
            tKsK0 = gmem_thr_copy_QKV.partition_D(sK0)
            tKsK1 = gmem_thr_copy_QKV.partition_D(sK1)
            tVsV0 = gmem_thr_copy_QKV.partition_D(sV0)
            tVsV1 = gmem_thr_copy_QKV.partition_D(sV1)

            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK0 = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
            tSrK1 = thr_mma.make_fragment_B(thr_mma.partition_B(sK1))
            tOrVt0 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
            tOrVt1 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt1))
            acc_shape_O = thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded))
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
            acc_O.fill(0.0)

            smem_copy_atom_Q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
            smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
            smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK0 = smem_thr_copy_K.partition_S(sK0)
            tSsK1 = smem_thr_copy_K.partition_S(sK1)
            tSrK0_copy_view = smem_thr_copy_K.retile(tSrK0)
            tSrK1_copy_view = smem_thr_copy_K.retile(tSrK1)
            tOsVt0 = smem_thr_copy_V.partition_S(sVt0)
            tOsVt1 = smem_thr_copy_V.partition_S(sVt1)
            tOrVt0_copy_view = smem_thr_copy_V.retile(tOrVt0)
            tOrVt1_copy_view = smem_thr_copy_V.retile(tOrVt1)

            mcQ = cute.make_identity_tensor(mQ.layout.shape)
            mcKV = cute.make_identity_tensor(mK.layout.shape)
            cQ = cute.local_tile(mcQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
            cKV = cute.local_tile(mcKV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (n_block, 0))
            tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

            tQpQ = cute.make_rmem_tensor(
                cute.make_layout((tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])), stride=(cute.size(tQsQ, mode=[2]), 0, 1)),
                cutlass.Boolean,
            )
            tKVpKV = cute.make_rmem_tensor(
                cute.make_layout((tKsK0.shape[0][1], cute.size(tKsK0, mode=[1]), cute.size(tKsK0, mode=[2])), stride=(cute.size(tKsK0, mode=[2]), 0, 1)),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
                for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                    tQpQ[rest_v, 0, rest_k] = cute.elem_less(tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3])
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3])

            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                    cute.copy(gmem_tiled_copy_QKV, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            for n in cutlass.range_constexpr(cute.size(tKsK0.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                    cute.copy(gmem_tiled_copy_QKV, tKgK[None, n, None, n_block], tKsK0[None, n, None], pred=tKVpKV[None, n, None])
                else:
                    tKsK0[None, n, None].fill(0)

            cute.arch.cp_async_commit_group()

            row_max = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            basic_params = SimpleNamespace(m_block=m_block, n_block=n_block, mQ=mQ, mK=mK, batch_size=batch_size, num_head=num_head)
            mma_params0 = SimpleNamespace(thr_mma=thr_mma, tiled_mma=tiled_mma, tSrQ=tSrQ, tSrK=tSrK0, tOrVt=tOrVt0, acc_O=acc_O)
            mma_params1 = SimpleNamespace(thr_mma=thr_mma, tiled_mma=tiled_mma, tSrQ=tSrQ, tSrK=tSrK1, tOrVt=tOrVt1, acc_O=acc_O)
            gmem_copy_params0 = SimpleNamespace(
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tVgV=tVgV,
                tKsK=tKsK0,
                tVsV=tVsV0,
                tKVpKV=tKVpKV,
            )
            gmem_copy_params1 = SimpleNamespace(
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tVgV=tVgV,
                tKsK=tKsK1,
                tVsV=tVsV1,
                tKVpKV=tKVpKV,
            )
            smem_copy_params0 = SimpleNamespace(
                smem_tiled_copy_Q=smem_tiled_copy_Q,
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsQ=tSsQ,
                tSrQ_copy_view=tSrQ_copy_view,
                tSsK=tSsK0,
                tSrK_copy_view=tSrK0_copy_view,
                tOsVt=tOsVt0,
                tOrVt_copy_view=tOrVt0_copy_view,
            )
            smem_copy_params1 = SimpleNamespace(
                smem_tiled_copy_Q=smem_tiled_copy_Q,
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsQ=tSsQ,
                tSrQ_copy_view=tSrQ_copy_view,
                tSsK=tSsK1,
                tSrK_copy_view=tSrK1_copy_view,
                tOsVt=tOsVt1,
                tOrVt_copy_view=tOrVt1_copy_view,
            )
            softmax_params = SimpleNamespace(row_max=row_max, row_sum=row_sum, softmax_scale_log2=softmax_scale_log2)

            mask_steps = 1
            if cutlass.const_expr(self._is_causal):
                mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

            for n_tile in cutlass.range_constexpr(mask_steps):
                n_block = n_block_max - n_tile - 1
                basic_params.n_block = n_block
                if cutlass.const_expr(self._is_causal):
                    if n_block >= 0:
                        self.compute_one_n_block(
                            basic_params,
                            mma_params0,
                            gmem_copy_params0,
                            smem_copy_params0,
                            softmax_params,
                            is_first_n_block=(n_tile == 0),
                            in_mask_steps=True,
                        )
                else:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params0,
                        gmem_copy_params0,
                        smem_copy_params0,
                        softmax_params,
                        is_first_n_block=True,
                        in_mask_steps=True,
                    )

            first_pipeline_block = n_block_max - mask_steps - 1
            second_pipeline_block = first_pipeline_block - 1
            if second_pipeline_block >= 0:
                cute.copy(
                    gmem_copy_params1.gmem_tiled_copy_QKV,
                    gmem_copy_params1.tKgK[None, None, None, second_pipeline_block],
                    gmem_copy_params1.tKsK,
                    pred=gmem_copy_params1.tKVpKV,
                )
                cute.arch.cp_async_commit_group()

            for n_tile in range(mask_steps, n_block_max, 2):
                n_block = n_block_max - n_tile - 1
                basic_params.n_block = n_block
                self.compute_one_n_block_pipelined(
                    basic_params,
                    mma_params0,
                    gmem_copy_params0,
                    smem_copy_params0,
                    softmax_params,
                    next_k_block=n_block - 2,
                )

                n_block_alt = n_block - 1
                if n_block_alt >= 0:
                    basic_params.n_block = n_block_alt
                    self.compute_one_n_block_pipelined(
                        basic_params,
                        mma_params1,
                        gmem_copy_params1,
                        smem_copy_params1,
                        softmax_params,
                        next_k_block=n_block_alt - 2,
                    )

            self.normalize_softmax(acc_O, row_sum)
            rO = cute.make_fragment_like(acc_O, self._dtype)
            rO.store(acc_O.load().to(self._dtype))
            sO = cute.make_tensor(sQ.iterator, sO_layout)

            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            gO = cute.local_tile(mO[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self._dtype)
            self.cta_sync_barrier.arrive_and_wait()
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
        def compute_one_n_block(
            self,
            basic_params: SimpleNamespace,
            mma_params: SimpleNamespace,
            gmem_copy_params: SimpleNamespace,
            smem_copy_params: SimpleNamespace,
            softmax_params: SimpleNamespace,
            is_first_n_block: cutlass.Constexpr,
            in_mask_steps: cutlass.Constexpr,
        ):
            acc_shape_S = mma_params.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size))
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_S.fill(0.0)

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            if is_first_n_block:
                for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(gmem_copy_params.tKVcKV[0, n, 0][1], basic_params.mK.layout.shape[1]):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tKVpKV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tKVpKV,
                )
            cute.arch.cp_async_commit_group()

            cute.copy(smem_copy_params.smem_tiled_copy_Q, smem_copy_params.tSsQ[None, None, 0], smem_copy_params.tSrQ_copy_view[None, None, 0])
            cute.copy(smem_copy_params.smem_tiled_copy_K, smem_copy_params.tSsK[None, None, 0], smem_copy_params.tSrK_copy_view[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
                k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
                cute.copy(smem_copy_params.smem_tiled_copy_Q, smem_copy_params.tSsQ[None, None, k_next], smem_copy_params.tSrQ_copy_view[None, None, k_next])
                cute.copy(smem_copy_params.smem_tiled_copy_K, smem_copy_params.tSsK[None, None, k_next], smem_copy_params.tSrK_copy_view[None, None, k_next])
                cute.gemm(mma_params.tiled_mma, acc_S, mma_params.tSrQ[None, None, k], mma_params.tSrK[None, None, k], acc_S)

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            if basic_params.n_block > 0:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
                cute.arch.cp_async_commit_group()

            self.softmax_rescale_O(basic_params, mma_params, softmax_params, acc_S, is_first_n_block, in_mask_steps)

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
            cute.copy(smem_copy_params.smem_tiled_copy_V, smem_copy_params.tOsVt[None, None, 0], smem_copy_params.tOrVt_copy_view[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(smem_copy_params.smem_tiled_copy_V, smem_copy_params.tOsVt[None, None, k_next], smem_copy_params.tOrVt_copy_view[None, None, k_next])
                cute.gemm(mma_params.tiled_mma, mma_params.acc_O, tOrS[None, None, k], mma_params.tOrVt[None, None, k], mma_params.acc_O)

        @cute.jit
        def compute_one_n_block_pipelined(
            self,
            basic_params: SimpleNamespace,
            mma_params: SimpleNamespace,
            gmem_copy_params: SimpleNamespace,
            smem_copy_params: SimpleNamespace,
            softmax_params: SimpleNamespace,
            next_k_block: cutlass.Int32,
        ):
            acc_shape_S = mma_params.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size))
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_S.fill(0.0)

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()

            cute.copy(smem_copy_params.smem_tiled_copy_Q, smem_copy_params.tSsQ[None, None, 0], smem_copy_params.tSrQ_copy_view[None, None, 0])
            cute.copy(smem_copy_params.smem_tiled_copy_K, smem_copy_params.tSsK[None, None, 0], smem_copy_params.tSrK_copy_view[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
                k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
                cute.copy(smem_copy_params.smem_tiled_copy_Q, smem_copy_params.tSsQ[None, None, k_next], smem_copy_params.tSrQ_copy_view[None, None, k_next])
                cute.copy(smem_copy_params.smem_tiled_copy_K, smem_copy_params.tSsK[None, None, k_next], smem_copy_params.tSrK_copy_view[None, None, k_next])
                cute.gemm(mma_params.tiled_mma, acc_S, mma_params.tSrQ[None, None, k], mma_params.tSrK[None, None, k], acc_S)

            if next_k_block >= 0:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None, next_k_block],
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            self.softmax_rescale_O(basic_params, mma_params, softmax_params, acc_S, False, False)

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
            cute.copy(smem_copy_params.smem_tiled_copy_V, smem_copy_params.tOsVt[None, None, 0], smem_copy_params.tOrVt_copy_view[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(smem_copy_params.smem_tiled_copy_V, smem_copy_params.tOsVt[None, None, k_next], smem_copy_params.tOrVt_copy_view[None, None, k_next])
                cute.gemm(mma_params.tiled_mma, mma_params.acc_O, tOrS[None, None, k], mma_params.tOrVt[None, None, k], mma_params.acc_O)

        @cute.jit
        def softmax_rescale_O(
            self,
            basic_params: SimpleNamespace,
            mma_params: SimpleNamespace,
            softmax_params: SimpleNamespace,
            acc_S: cute.Tensor,
            is_first_n_block: cutlass.Constexpr,
            in_mask_steps: cutlass.Constexpr,
        ):
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
            row_max_prev = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev = cute.make_fragment_like(softmax_params.row_max, cutlass.Float32)
                cute.basic_copy(softmax_params.row_max, row_max_prev)
            tScS_mn = None
            if cutlass.const_expr(in_mask_steps):
                mcS = cute.make_identity_tensor((basic_params.mQ.shape[0], basic_params.mQ.shape[1], basic_params.mQ.shape[2], basic_params.mK.shape[1]))
                cS = cute.local_tile(mcS[basic_params.batch_size, None, basic_params.num_head, None], (self._m_block_size, self._n_block_size), (basic_params.m_block, basic_params.n_block))
                tScS = mma_params.thr_mma.partition_C(cS)
                tScS_mn = self._make_acc_tensor_mn_view(tScS)

            for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
                if cutlass.const_expr(in_mask_steps):
                    if cutlass.const_expr(not self._is_causal):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            if cute.elem_less(basic_params.mK.shape[1], tScS_mn[0, c][3] + 1):
                                acc_S_mn[r, c] = -cutlass.Float32.inf
                    else:
                        col_idx_limit = cutlass.min(tScS_mn[r, 0][1] + 1, basic_params.mK.shape[1])
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
                    acc_S_row * softmax_params.softmax_scale_log2 - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                if cutlass.const_expr(not is_first_n_block):
                    prev_minus_cur_exp = cute.math.exp2(
                        row_max_prev_row * softmax_params.softmax_scale_log2 - row_max_cur_row * softmax_params.softmax_scale_log2,
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
                softmax_params.row_max[r] = row_max_cur_row
                softmax_params.row_sum[r] = acc_S_row_sum
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


def _make_stage12_config(config: AttentionConfig, *, block_m: int, block_n: int, num_stages_kv: int) -> AttentionConfig:
    return AttentionConfig(
        softmax_scale=config.softmax_scale,
        causal=config.causal,
        block_m=block_m,
        block_n=block_n,
        num_threads=config.num_threads,
        num_stages_kv=num_stages_kv,
        autotune=False,
    )


def _stage12_candidate_values(preferred: int, values: list[int], *, limit: int) -> list[int]:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value > limit or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _stage12_autotune_cache_key(config: AttentionConfig, q) -> str:
    device_name = torch.cuda.get_device_name(q.device)
    return "|".join(
        [
            device_name,
            str(tuple(q.shape)),
            str(q.dtype),
            str(config.num_threads),
            str(config.block_m),
            str(config.block_n),
        ]
    )


def _load_stage12_autotune_cache_from_disk() -> dict[str, dict[str, int]]:
    if not _STAGE12_AUTOTUNE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_STAGE12_AUTOTUNE_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_stage12_autotune_cache_to_disk(entries: dict[str, dict[str, int]]) -> None:
    _STAGE12_AUTOTUNE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STAGE12_AUTOTUNE_CACHE_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))


def autotune_stage12_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage12 autotune only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage12 autotune requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage12 autotune currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    cache_key = (
        tuple(q.shape),
        str(q.dtype),
        config.num_threads,
        config.block_m,
        config.block_n,
    )
    cached = _STAGE12_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache_key_disk = _stage12_autotune_cache_key(config, q)
    disk_cache = _load_stage12_autotune_cache_from_disk()
    cached_disk = disk_cache.get(cache_key_disk)
    if cached_disk is not None:
        tuned = _make_stage12_config(
            config,
            block_m=int(cached_disk["block_m"]),
            block_n=int(cached_disk["block_n"]),
            num_stages_kv=int(cached_disk["num_stages_kv"]),
        )
        _STAGE12_AUTOTUNE_CACHE[cache_key] = tuned
        return tuned

    block_m_values = _stage12_candidate_values(config.block_m, [128, 96, 64, 48, 32, 16], limit=seq_len)
    block_n_values = _stage12_candidate_values(config.block_n, [256, 192, 128, 96, 64], limit=seq_len)
    stage_values = _stage12_candidate_values(config.num_stages_kv or 2, [2], limit=2)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    best_config = None
    best_ms = None

    for num_stages_kv in stage_values:
        for block_m in block_m_values:
            for block_n in block_n_values:
                tuned = _make_stage12_config(
                    config,
                    block_m=block_m,
                    block_n=block_n,
                    num_stages_kv=num_stages_kv,
                )
                try:
                    for _ in range(warmup):
                        _stage12_forward_impl(q, k, v, tuned)

                    torch.cuda.synchronize()
                    elapsed = 0.0
                    for _ in range(repeat):
                        start_event.record()
                        _stage12_forward_impl(q, k, v, tuned)
                        end_event.record()
                        torch.cuda.synchronize()
                        elapsed += start_event.elapsed_time(end_event)
                    elapsed /= repeat
                except ValueError:
                    continue

                if best_ms is None or elapsed < best_ms:
                    best_ms = elapsed
                    best_config = tuned

    if best_config is None:
        raise ValueError(
            f"stage12 autotune failed to find a valid config for shape={(batch, heads, seq_len, head_dim)} "
            f"with num_threads={config.num_threads}."
        )

    _STAGE12_AUTOTUNE_CACHE[cache_key] = best_config
    disk_cache[cache_key_disk] = {
        "block_m": best_config.block_m,
        "block_n": best_config.block_n,
        "num_stages_kv": best_config.num_stages_kv,
    }
    _save_stage12_autotune_cache_to_disk(disk_cache)
    return best_config


def _stage12_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage12 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage12 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage12 currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE12_CUTE:
        raise ValueError(
            f"stage12 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE12_CUTE}, got {seq_len}."
        )
    if not Stage12FlashAttentionAmpere.can_implement(cutlass.Float16, head_dim, config.block_m, config.block_n, config.num_threads, True):
        raise ValueError("stage12 config is not supported by the MMA kernel constraints.")

    q_perm = q.permute(0, 2, 1, 3).contiguous()
    k_perm = k.permute(0, 2, 1, 3).contiguous()
    v_perm = v.permute(0, 2, 1, 3).contiguous()
    o_perm = torch.empty_like(q_perm)

    q_cute = from_dlpack(q_perm, assumed_align=16)
    k_cute = from_dlpack(k_perm, assumed_align=16)
    v_cute = from_dlpack(v_perm, assumed_align=16)
    o_cute = from_dlpack(o_perm, assumed_align=16)
    scale = config.resolve_scale(head_dim)
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    cache_key = (tuple(q_perm.shape), str(q_perm.dtype), config.block_m, config.block_n, config.num_threads)
    compiled = _stage12_compile(
        cache_key,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        scale,
        current_stream,
        head_dim,
        config.block_m,
        config.block_n,
        config.num_threads,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, scale, current_stream)
    return o_perm.permute(0, 2, 1, 3).contiguous()


def stage12_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig()
    tuned = config
    if config.autotune:
        tuned = autotune_stage12_config(q, k, v, config)
    elif tuned.num_stages_kv == 0:
        tuned = replace(tuned, num_stages_kv=2)

    if tuned.num_stages_kv != 2:
        raise ValueError(f"stage12 currently supports num_stages_kv == 2, got {tuned.num_stages_kv}.")
    return _stage12_forward_impl(q, k, v, replace(tuned, autotune=False))


def _stage12_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, current_stream, head_dim, block_m, block_n, num_threads):
    compiled = _STAGE12_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        kernel = Stage12FlashAttentionAmpere(
            head_dim=head_dim,
            m_block_size=block_m,
            n_block_size=block_n,
            num_threads=num_threads,
            is_causal=True,
        )
        compiled = cute.compile(kernel, q_cute, k_cute, v_cute, o_cute, scale, current_stream)
        _STAGE12_COMPILED_CACHE[cache_key] = compiled
    return compiled
