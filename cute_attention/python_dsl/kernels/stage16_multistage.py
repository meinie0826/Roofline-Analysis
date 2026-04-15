"""
Stage16: SM90-style warp-specialized attention with double-buffered K/V pipeline.

Stage15 uses a single K/V shared-memory buffer and synchronously waits for each
tile to be loaded before computing (cp_async_wait_group 0).  Stage16 extends it
with two K/V buffer slots (sK0/sK1 + sV0/sV1).  Producer warps prefetch the
*next* tile while consumer warps compute on the *current* tile
(cp_async_wait_group 1), hiding global-memory latency behind computation.

Key idea (borrowed from Stage13's approach): the main loop advances in steps of
2, processing slot0 and slot1 explicitly in each iteration instead of selecting
the slot with a runtime index.

Warp assignment:
  * warps 0-3  → producer  (128 threads): issue cp.async for K/V
  * warps 4-7  → consumer  (128 threads): compute QK^T softmax PV
"""
from __future__ import annotations

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
MAX_SEQ_LEN_FOR_STAGE16_CUTE = 4096
_STAGE16_COMPILED_CACHE: dict = {}
_STAGE16_AUTOTUNE_CACHE: dict = {}
_STAGE16_AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "stage16_autotune.json"


if HAS_CUTE:
    class Stage16FlashAttentionSm90DoubleBuffer:
        """SM90-style warp-specialized attention with double-buffered K/V tiles.

        Independent implementation (does not inherit Stage15) to avoid CuTe JIT
        Constexpr resolution issues with inherited @cute.jit/__call__ overrides.
        """

        def __init__(self, head_dim: int, m_block_size: int, n_block_size: int,
                     num_threads: int, is_causal: bool):
            self._head_dim = head_dim
            self._m_block_size = m_block_size
            self._n_block_size = n_block_size
            self._head_dim_padded = (head_dim + 31) // 32 * 32
            self._num_threads = num_threads
            self._is_causal = is_causal
            self._producer_threads = 128
            self._consumer_threads = 128
            self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

        @staticmethod
        def can_implement(dtype, head_dim: int, m_block_size: int, n_block_size: int,
                          num_threads: int, is_causal: bool) -> bool:
            if dtype != cutlass.Float16:
                return False
            if head_dim % 8 != 0:
                return False
            if num_threads != 256:
                return False
            # double-buffer: 1×sQ + 2×(sK + sV), all fp16
            smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 4) * 2
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
            gmem_tiled_copy_Q  = cute.make_tiled_copy_tv(atom_async_copy,   consumer_layout, vQKV_layout)
            gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy,   producer_layout, vQKV_layout)
            gmem_tiled_copy_O  = cute.make_tiled_copy_tv(atom_universal_copy, consumer_layout, vQKV_layout)

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
                mQ, mK, mV, mO,
                softmax_scale_log2,
                sQ_layout, sKV_layout, sO_layout,
                gmem_tiled_copy_Q, gmem_tiled_copy_KV, gmem_tiled_copy_O,
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
            gmem_tiled_copy_Q:  cute.TiledCopy,
            gmem_tiled_copy_KV: cute.TiledCopy,
            gmem_tiled_copy_O:  cute.TiledCopy,
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
            sQ  = storage.sQ.get_tensor(sQ_layout)
            sK0 = storage.sK0.get_tensor(sKV_layout)
            sV0 = storage.sV0.get_tensor(sKV_layout)
            sK1 = storage.sK1.get_tensor(sKV_layout)
            sV1 = storage.sV1.get_tensor(sKV_layout)
            sVt0 = cute.composition(sV0, cute.make_layout(
                (self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))
            sVt1 = cute.composition(sV1, cute.make_layout(
                (self._head_dim_padded, self._n_block_size), stride=(self._n_block_size, 1)))

            # ── Consumer: Q copy setup ────────────────────────────────────────
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(consumer_slice_idx)
            tQgQ = gmem_thr_copy_Q.partition_S(gQ)
            tQsQ = gmem_thr_copy_Q.partition_D(sQ)
            mcQ = cute.make_identity_tensor(mQ.layout.shape)
            cQ  = cute.local_tile(mcQ[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
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

            # ── Consumer: MMA setup ───────────────────────────────────────────
            thr_mma  = tiled_mma.get_slice(consumer_slice_idx)
            tSrQ     = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK0    = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
            tSrK1    = thr_mma.make_fragment_B(thr_mma.partition_B(sK1))
            tOrVt0   = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
            tOrVt1   = thr_mma.make_fragment_B(thr_mma.partition_B(sVt1))
            acc_shape_O = thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded))
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
            acc_O.fill(0.0)

            smem_copy_atom_Q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
            smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
            smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True,  num_matrices=4), self._dtype)
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(consumer_slice_idx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(consumer_slice_idx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(consumer_slice_idx)

            tSsQ         = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_view    = smem_thr_copy_Q.retile(tSrQ)
            tSsK0        = smem_thr_copy_K.partition_S(sK0)
            tSsK1        = smem_thr_copy_K.partition_S(sK1)
            tSrK0_view   = smem_thr_copy_K.retile(tSrK0)
            tSrK1_view   = smem_thr_copy_K.retile(tSrK1)
            tOsVt0       = smem_thr_copy_V.partition_S(sVt0)
            tOsVt1       = smem_thr_copy_V.partition_S(sVt1)
            tOrVt0_view  = smem_thr_copy_V.retile(tOrVt0)
            tOrVt1_view  = smem_thr_copy_V.retile(tOrVt1)

            row_max = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            # ── Producer: KV copy setup ───────────────────────────────────────
            gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(producer_slice_idx)
            tKgK  = gmem_thr_copy_KV.partition_S(gK)
            tVgV  = gmem_thr_copy_KV.partition_S(gV)
            tKsK0 = gmem_thr_copy_KV.partition_D(sK0)
            tVsV0 = gmem_thr_copy_KV.partition_D(sV0)
            tKsK1 = gmem_thr_copy_KV.partition_D(sK1)
            tVsV1 = gmem_thr_copy_KV.partition_D(sV1)
            mcKV = cute.make_identity_tensor(mK.layout.shape)
            cKV  = cute.local_tile(mcKV[batch_size, None, num_head, None], (self._n_block_size, self._head_dim_padded), (start_n_block, 0))
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

            # ── Prologue ──────────────────────────────────────────────────────
            if is_consumer:
                for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                    if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                        cute.copy(gmem_tiled_copy_Q, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                    else:
                        tQsQ[None, m, None].fill(0)
                cute.arch.cp_async_commit_group()
            else:
                # slot0 ← tile[n_block_max-1]
                self._copy_kv_tile(
                    start_n_block,
                    start_n_block,
                    tKVcKV,
                    tKVpKV,
                    tKgK,
                    tVgV,
                    tKsK0,
                    tVsV0,
                    gmem_tiled_copy_KV,
                    mK,
                )
                # slot1 ← tile[n_block_max-2]
                second = start_n_block - 1
                if second >= 0:
                    self._copy_kv_tile(
                        second,
                        start_n_block,
                        tKVcKV,
                        tKVpKV,
                        tKgK,
                        tVgV,
                        tKsK1,
                        tVsV1,
                        gmem_tiled_copy_KV,
                        mK,
                    )
                else:
                    cute.arch.cp_async_commit_group()

            # Q + slot0 ready; slot1 may still be in-flight → wait_group(1)
            cute.arch.cp_async_wait_group(1)
            self.cta_sync_barrier.arrive_and_wait()

            # ── Mask steps ────────────────────────────────────────────────────
            mask_steps = 1
            if cutlass.const_expr(self._is_causal):
                mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

            for n_tile in cutlass.range_constexpr(mask_steps):
                n_block = n_block_max - n_tile - 1
                self._compute_one_block(
                    is_consumer, is_producer,
                    n_block, thr_mma, tiled_mma,
                    tSrQ, tSrK0, tOrVt0, acc_O,
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                    tSsQ, tSrQ_view, tSsK0, tSrK0_view, tOsVt0, tOrVt0_view,
                    row_max, row_sum, softmax_scale_log2,
                    mQ, mK, batch_size, num_head, m_block,
                    is_first_n_block=(n_tile == 0),
                    in_mask_steps=True,
                    gmem_tiled_copy_KV=gmem_tiled_copy_KV,
                    tKVcKV=tKVcKV, tKgK=tKgK, tVgV=tVgV,
                    tKsK=tKsK0, tVsV=tVsV0, tKVpKV=tKVpKV,
                    next_k_block=n_block - 2,
                    wait_group=1,
                )

            # Drain the in-flight slot1 prefetch before pipeline loop
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            # ── Pipeline loop (step=2) ────────────────────────────────────────
            for n_tile in range(mask_steps, n_block_max, 2):
                # --- slot0 ---
                n_block0 = n_block_max - n_tile - 1
                self._compute_one_block(
                    is_consumer, is_producer,
                    n_block0, thr_mma, tiled_mma,
                    tSrQ, tSrK0, tOrVt0, acc_O,
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                    tSsQ, tSrQ_view, tSsK0, tSrK0_view, tOsVt0, tOrVt0_view,
                    row_max, row_sum, softmax_scale_log2,
                    mQ, mK, batch_size, num_head, m_block,
                    is_first_n_block=False,
                    in_mask_steps=False,
                    gmem_tiled_copy_KV=gmem_tiled_copy_KV,
                    tKVcKV=tKVcKV, tKgK=tKgK, tVgV=tVgV,
                    tKsK=tKsK0, tVsV=tVsV0, tKVpKV=tKVpKV,
                    next_k_block=n_block0 - 2,
                    wait_group=1,
                )

                # --- slot1 ---
                n_block1 = n_block0 - 1
                if n_block1 >= 0:
                    self._compute_one_block(
                        is_consumer, is_producer,
                        n_block1, thr_mma, tiled_mma,
                        tSrQ, tSrK1, tOrVt1, acc_O,
                        smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                        tSsQ, tSrQ_view, tSsK1, tSrK1_view, tOsVt1, tOrVt1_view,
                        row_max, row_sum, softmax_scale_log2,
                        mQ, mK, batch_size, num_head, m_block,
                        is_first_n_block=False,
                        in_mask_steps=False,
                        gmem_tiled_copy_KV=gmem_tiled_copy_KV,
                        tKVcKV=tKVcKV, tKgK=tKgK, tVgV=tVgV,
                        tKsK=tKsK1, tVsV=tVsV1, tKVpKV=tKVpKV,
                        next_k_block=n_block1 - 2,
                        wait_group=1,
                    )

            # ── Epilogue ──────────────────────────────────────────────────────
            if is_consumer:
                self.normalize_softmax(acc_O, row_sum)
                rO = cute.make_fragment_like(acc_O, self._dtype)
                rO.store(acc_O.load().to(self._dtype))
                sO = cute.make_tensor(sQ.iterator, sO_layout)

                smem_copy_atom_O  = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
                smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
                smem_thr_copy_O   = smem_tiled_copy_O.get_slice(consumer_slice_idx)
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
                cO  = cute.local_tile(mcO[batch_size, None, num_head, None], (self._m_block_size, self._head_dim_padded), (m_block, 0))
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

        # ── Per-block compute helper ──────────────────────────────────────────

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
            next_k_block,
            wait_group: cutlass.Constexpr,
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
                    acc_S, acc_O, row_max, row_sum, softmax_scale_log2,
                    mQ, mK, batch_size, num_head, m_block, n_block,
                    is_first_n_block=is_first_n_block,
                    in_mask_steps=in_mask_steps,
                    thr_mma=thr_mma,
                )

                rP = cute.make_fragment_like(acc_S, self._dtype)
                rP.store(acc_S.load().to(self._dtype))
                rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
                rP_mma_view = cute.make_layout(
                    ((rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                     rP_layout_divided.shape[1],
                     rP_layout_divided.shape[2][1]),
                    stride=((rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                            rP_layout_divided.stride[1],
                            rP_layout_divided.stride[2][1]),
                )
                tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
                cute.copy(smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0])
                for k_idx in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                    k_next = (k_idx + 1) % cute.size(tOrS.shape[2])
                    cute.copy(smem_tiled_copy_V, tOsVt[None, None, k_next], tOrVt_view[None, None, k_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, k_idx], tOrVt[None, None, k_idx], acc_O)

            self.cta_sync_barrier.arrive_and_wait()

            if is_producer and next_k_block >= 0:
                self._copy_kv_tile(
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
            else:
                if is_producer:
                    cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(wait_group)
            self.cta_sync_barrier.arrive_and_wait()

        @cute.jit
        def _copy_kv_tile(
            self,
            tile_block,
            predicate_block,
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

        # ── Softmax helpers (copied from Stage15) ────────────────────────────

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
                cS  = cute.local_tile(mcS[batch_size, None, num_head, None], (self._m_block_size, self._n_block_size), (m_block, n_block))
                tScS    = thr_mma.partition_C(cS)
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
                    row_max_cur_row  = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
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


# ---------------------------------------------------------------------------
# Autotune helpers
# ---------------------------------------------------------------------------

def _make_stage16_config(config: AttentionConfig, *, block_m: int, block_n: int) -> AttentionConfig:
    return AttentionConfig(
        softmax_scale=config.softmax_scale,
        causal=config.causal,
        block_m=block_m,
        block_n=block_n,
        num_threads=256,
        num_stages_kv=2,
        autotune=False,
    )


def _stage16_autotune_cache_key(config: AttentionConfig, q) -> str:
    device_name = torch.cuda.get_device_name(q.device)
    return "|".join([device_name, str(tuple(q.shape)), str(q.dtype), str(config.block_m), str(config.block_n)])


def _load_stage16_autotune_cache_from_disk() -> dict:
    if not _STAGE16_AUTOTUNE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_STAGE16_AUTOTUNE_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_stage16_autotune_cache_to_disk(entries: dict) -> None:
    _STAGE16_AUTOTUNE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STAGE16_AUTOTUNE_CACHE_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))


def _stage16_candidate_values(preferred: int, values: list, *, limit: int) -> list:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value > limit or value in ordered:
            continue
        ordered.append(value)
    return ordered


def autotune_stage16_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    """Find the best block_m / block_n configuration for stage16."""
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage16 autotune only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage16 autotune requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage16 autotune currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    cache_key = (tuple(q.shape), str(q.dtype), config.block_m, config.block_n)
    cached = _STAGE16_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_key_disk = _stage16_autotune_cache_key(config, q)
    disk_cache = _load_stage16_autotune_cache_from_disk()
    cached_disk = disk_cache.get(cache_key_disk)
    if cached_disk is not None:
        tuned = _make_stage16_config(config, block_m=int(cached_disk["block_m"]), block_n=int(cached_disk["block_n"]))
        _STAGE16_AUTOTUNE_CACHE[cache_key] = tuned
        return tuned

    block_m_values = _stage16_candidate_values(config.block_m, [128, 96, 64, 48, 32], limit=seq_len)
    block_n_values = _stage16_candidate_values(config.block_n, [256, 192, 128, 96, 64], limit=seq_len)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    best_config = None
    best_ms     = None

    for block_m in block_m_values:
        for block_n in block_n_values:
            tuned = _make_stage16_config(config, block_m=block_m, block_n=block_n)
            if not Stage16FlashAttentionSm90DoubleBuffer.can_implement(
                cutlass.Float16, head_dim, block_m, block_n, 256, True
            ):
                continue
            try:
                for _ in range(warmup):
                    _stage16_forward_impl(q, k, v, tuned)
                torch.cuda.synchronize()
                elapsed = 0.0
                for _ in range(repeat):
                    start_event.record()
                    _stage16_forward_impl(q, k, v, tuned)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed += start_event.elapsed_time(end_event)
                elapsed /= repeat
            except (ValueError, RuntimeError):
                continue
            if best_ms is None or elapsed < best_ms:
                best_ms     = elapsed
                best_config = tuned

    if best_config is None:
        raise ValueError("stage16 autotune failed to find a valid config.")

    _STAGE16_AUTOTUNE_CACHE[cache_key] = best_config
    disk_cache[cache_key_disk] = {"block_m": best_config.block_m, "block_n": best_config.block_n}
    _save_stage16_autotune_cache_to_disk(disk_cache)
    return best_config


# ---------------------------------------------------------------------------
# Forward entry points
# ---------------------------------------------------------------------------

def _stage16_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage16 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage16 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage16 currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE16_CUTE:
        raise ValueError(f"stage16 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE16_CUTE}, got {seq_len}.")

    if not Stage16FlashAttentionSm90DoubleBuffer.can_implement(
        cutlass.Float16, head_dim, config.block_m, config.block_n, 256, True
    ):
        raise ValueError("stage16 config is not supported by the double-buffer kernel constraints.")

    q_perm = q.permute(0, 2, 1, 3).contiguous()
    k_perm = k.permute(0, 2, 1, 3).contiguous()
    v_perm = v.permute(0, 2, 1, 3).contiguous()
    o_perm = torch.empty_like(q_perm)

    q_cute = from_dlpack(q_perm, assumed_align=16)
    k_cute = from_dlpack(k_perm, assumed_align=16)
    v_cute = from_dlpack(v_perm, assumed_align=16)
    o_cute = from_dlpack(o_perm, assumed_align=16)
    scale          = config.resolve_scale(head_dim)
    torch_stream   = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    cache_key = (tuple(q_perm.shape), str(q_perm.dtype), config.block_m, config.block_n)
    compiled  = _stage16_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, current_stream,
                                  head_dim, config.block_m, config.block_n)
    compiled(q_cute, k_cute, v_cute, o_cute, scale, current_stream)
    return o_perm.permute(0, 2, 1, 3).contiguous()


def stage16_forward(q, k, v, config: AttentionConfig | None = None):
    """Stage16: SM90-style warp-specialization with double-buffered K/V pipeline."""
    config = config or AttentionConfig(block_m=64, block_n=128, num_threads=256)
    tuned  = replace(config, num_threads=256, num_stages_kv=2, autotune=False)
    if config.autotune:
        tuned = autotune_stage16_config(q, k, v, config)
    return _stage16_forward_impl(q, k, v, tuned)


def _stage16_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, current_stream,
                      head_dim, block_m, block_n):
    compiled = _STAGE16_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        kernel = Stage16FlashAttentionSm90DoubleBuffer(
            head_dim=head_dim,
            m_block_size=block_m,
            n_block_size=block_n,
            num_threads=256,
            is_causal=True,
        )
        compiled = cute.compile(kernel, q_cute, k_cute, v_cute, o_cute, scale, current_stream)
        _STAGE16_COMPILED_CACHE[cache_key] = compiled
    return compiled
