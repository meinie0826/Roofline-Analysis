from dataclasses import replace

import cuda.bindings.driver as cuda

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

import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

MAX_SEQ_LEN_FOR_STAGE22_CUTE = 4096
_STAGE22_COMPILED_CACHE = {}


class Stage22FlashAttentionTma:
    def __init__(self, head_dim: int, block_m: int = 128, block_n: int = 128):
        self.head_dim = head_dim
        self.block_m = block_m
        self.block_n = block_n
        self.num_threads = 128

    @staticmethod
    def can_implement(head_dim: int, config: AttentionConfig) -> bool:
        return (
            config.causal
            and config.block_m == 128
            and config.block_n == 128
            and head_dim in {32, 64, 128}
        )

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor, softmax_scale: cutlass.Float32, stream: cuda.CUstream):
        dtype = q.element_type
        if cutlass.const_expr(dtype != cutlass.Float16):
            raise TypeError("stage22 only supports fp16")

        batch_heads, seq_len, head_dim = q.shape
        q_layout = cute.make_layout((seq_len, head_dim, batch_heads), stride=(q.stride[1], q.stride[2], q.stride[0]))
        k_layout = cute.make_layout((seq_len, head_dim, batch_heads), stride=(k.stride[1], k.stride[2], k.stride[0]))
        v_layout = cute.make_layout((seq_len, head_dim, batch_heads), stride=(v.stride[1], v.stride[2], v.stride[0]))
        o_layout = cute.make_layout((seq_len, head_dim, batch_heads), stride=(o.stride[1], o.stride[2], o.stride[0]))
        q_sdb = cute.make_tensor(q.iterator, q_layout)
        k_sdb = cute.make_tensor(k.iterator, k_layout)
        v_sdb = cute.make_tensor(v.iterator, v_layout)
        o_sdb = cute.make_tensor(o.iterator, o_layout)

        mma_tiler = (self.block_m, self.block_n, self.head_dim)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.block_m, self.block_n),
        )

        q_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, dtype, 1
        )
        kv_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, dtype, 1
        )
        q_smem_layout_one_stage = cute.select(q_smem_layout, mode=[0, 1, 2])
        kv_smem_layout_one_stage = cute.select(kv_smem_layout, mode=[0, 1, 2])

        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, q_sdb, q_smem_layout_one_stage, mma_tiler, tiled_mma
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, k_sdb, kv_smem_layout_one_stage, mma_tiler, tiled_mma
        )
        v_tma_atom, v_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, v_sdb, kv_smem_layout_one_stage, mma_tiler, tiled_mma
        )

        self.kernel(
            tiled_mma,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            v_tma_atom,
            v_tma_tensor,
            o_sdb,
            q_smem_layout,
            kv_smem_layout,
            softmax_scale,
        ).launch(
            grid=(cute.ceil_div(seq_len, self.block_m), batch_heads, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        v_tma_atom: cute.CopyAtom,
        v_tma_tensor: cute.Tensor,
        o: cute.Tensor,
        q_smem_layout: cute.ComposedLayout,
        kv_smem_layout: cute.ComposedLayout,
        softmax_scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = tidx // 32
        m_block, batch_head, _ = cute.arch.block_idx()
        row = tidx

        smem = cutlass.utils.SmemAllocator()
        q_mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        k_mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        v_mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        sQ = smem.allocate_tensor(
            cutlass.Float16,
            q_smem_layout.outer,
            byte_alignment=128,
            swizzle=q_smem_layout.inner,
        )
        sK = smem.allocate_tensor(
            cutlass.Float16,
            kv_smem_layout.outer,
            byte_alignment=128,
            swizzle=kv_smem_layout.inner,
        )
        sV = smem.allocate_tensor(
            cutlass.Float16,
            kv_smem_layout.outer,
            byte_alignment=128,
            swizzle=kv_smem_layout.inner,
        )

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads)
        cta_layout = cute.make_layout((1, 1, 1, 1))
        q_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=q_mbar,
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.block_m * self.head_dim * 2,
            cta_layout_vmnk=cta_layout,
        )
        k_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=k_mbar,
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.block_n * self.head_dim * 2,
            cta_layout_vmnk=cta_layout,
        )
        v_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=v_mbar,
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.block_n * self.head_dim * 2,
            cta_layout_vmnk=cta_layout,
        )

        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        thr_mma = tiled_mma.get_slice(0)
        gQ = cute.local_tile(q_tma_tensor[None, None, batch_head], (self.block_m, self.head_dim), (m_block, 0))
        gK = cute.local_tile(k_tma_tensor[None, None, batch_head], (self.block_n, self.head_dim), (None, 0))
        gV = cute.local_tile(v_tma_tensor[None, None, batch_head], (self.block_n, self.head_dim), (None, 0))
        gO = cute.local_tile(o[None, None, batch_head], (self.block_m, self.head_dim), (m_block, 0))

        tQgQ = thr_mma.partition_A(gQ)
        tKgK = thr_mma.partition_B(gK)
        tVgV = thr_mma.partition_B(gV)
        tQsQ, tQgQ = cpasync.tma_partition(
            q_tma_atom,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tQgQ, 0, 3),
        )
        tKsK, tKgK = cpasync.tma_partition(
            k_tma_atom,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tKgK, 0, 3),
        )
        tVsV, tVgV = cpasync.tma_partition(
            v_tma_atom,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tVgV, 0, 3),
        )

        if warp_idx == 0:
            cpasync.prefetch_descriptor(q_tma_atom)
            cpasync.prefetch_descriptor(k_tma_atom)
            cpasync.prefetch_descriptor(v_tma_atom)
            q_pipe.producer_acquire(q_prod)
            cute.copy(
                q_tma_atom,
                tQgQ[None, 0],
                tQsQ[None, 0],
                tma_bar_ptr=q_pipe.producer_get_barrier(q_prod),
            )
            q_pipe.producer_commit(q_prod)
            q_prod.advance()

        q_pipe.consumer_wait(q_cons)
        cute.arch.barrier()
        q_pipe.consumer_release(q_cons)
        q_cons.advance()

        seq_len = o.shape[0]
        q_row = m_block * self.block_m + row
        row_in_bounds = q_row < seq_len
        block_row_limit = cute.min((m_block + 1) * self.block_m, seq_len)
        n_tiles = cute.ceil_div(block_row_limit, self.block_n)

        acc_o = cute.make_rmem_tensor((self.head_dim,), cutlass.Float32)
        acc_o.fill(0.0)
        row_max = -cutlass.Float32.inf
        row_sum = cutlass.Float32(0.0)

        n_tile = 0
        while n_tile < n_tiles:
            if warp_idx == 0:
                k_pipe.producer_acquire(k_prod)
                cute.copy(
                    k_tma_atom,
                    tKgK[None, n_tile],
                    tKsK[None, 0],
                    tma_bar_ptr=k_pipe.producer_get_barrier(k_prod),
                )
                k_pipe.producer_commit(k_prod)
                k_prod.advance()

                v_pipe.producer_acquire(v_prod)
                cute.copy(
                    v_tma_atom,
                    tVgV[None, n_tile],
                    tVsV[None, 0],
                    tma_bar_ptr=v_pipe.producer_get_barrier(v_prod),
                )
                v_pipe.producer_commit(v_prod)
                v_prod.advance()

            k_pipe.consumer_wait(k_cons)
            v_pipe.consumer_wait(v_cons)
            cute.arch.barrier()

            if row_in_bounds:
                tile_start = n_tile * self.block_n
                tile_stop = cute.min(tile_start + self.block_n, q_row + 1)
                tile_cols = tile_stop - tile_start

                block_max = -cutlass.Float32.inf
                col = 0
                while col < tile_cols:
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self.head_dim):
                        score += sQ[row, d, 0].to(cutlass.Float32) * sK[col, d, 0].to(cutlass.Float32)
                    score *= softmax_scale
                    block_max = cute.arch.fmax(block_max, score)
                    col += 1

                new_max = cute.arch.fmax(row_max, block_max)
                old_scale = cute.math.exp(row_max - new_max)
                for d in cutlass.range_constexpr(self.head_dim):
                    acc_o[d] = acc_o[d] * old_scale

                block_sum = cutlass.Float32(0.0)
                col = 0
                while col < tile_cols:
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self.head_dim):
                        score += sQ[row, d, 0].to(cutlass.Float32) * sK[col, d, 0].to(cutlass.Float32)
                    score *= softmax_scale
                    p = cute.math.exp(score - new_max)
                    block_sum += p
                    for d in cutlass.range_constexpr(self.head_dim):
                        acc_o[d] = acc_o[d] + p * sV[col, d, 0].to(cutlass.Float32)
                    col += 1

                row_sum = row_sum * old_scale + block_sum
                row_max = new_max

            k_pipe.consumer_release(k_cons)
            v_pipe.consumer_release(v_cons)
            k_cons.advance()
            v_cons.advance()
            n_tile += 1

        if row_in_bounds:
            for d in cutlass.range_constexpr(self.head_dim):
                gO[row, d] = (acc_o[d] / row_sum).to(cutlass.Float16)


def _normalize_stage22_config(config: AttentionConfig | None) -> AttentionConfig:
    return replace(
        config or AttentionConfig(),
        causal=True,
        block_m=128,
        block_n=128,
        num_threads=128,
        num_stages_kv=1,
        autotune=False,
    )


def autotune_stage22_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    _ = warmup
    _ = repeat
    require_torch()
    validate_qkv(q, k, v)
    return _normalize_stage22_config(config)


def _stage22_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage22 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage22 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage22 currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE22_CUTE:
        raise ValueError(f"stage22 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE22_CUTE}, got {seq_len}.")

    normalized = _normalize_stage22_config(config)
    if not Stage22FlashAttentionTma.can_implement(head_dim, normalized):
        raise ValueError("stage22 config is not supported by the minimal SM100 TMA kernel constraints.")

    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = torch.empty_like(q_flat)

    q_cute = from_dlpack(q_flat, assumed_align=16)
    k_cute = from_dlpack(k_flat, assumed_align=16)
    v_cute = from_dlpack(v_flat, assumed_align=16)
    o_cute = from_dlpack(o_flat, assumed_align=16)
    scale = normalized.resolve_scale(head_dim)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    cache_key = (
        tuple(q_flat.shape),
        str(q_flat.dtype),
        normalized.block_m,
        normalized.block_n,
    )
    compiled = _STAGE22_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            Stage22FlashAttentionTma(
                head_dim,
                normalized.block_m,
                normalized.block_n,
            ),
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            scale,
            current_stream,
        )
        _STAGE22_COMPILED_CACHE[cache_key] = compiled

    compiled(q_cute, k_cute, v_cute, o_cute, scale, current_stream)
    return o_flat.reshape(batch, heads, seq_len, head_dim)


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=128, block_n=128, num_threads=128)
    tuned = _normalize_stage22_config(config)
    if config.autotune:
        tuned = autotune_stage22_config(q, k, v, tuned)
    return _stage22_forward_impl(q, k, v, tuned)
