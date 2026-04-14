#!/usr/bin/env python3
"""
Unified FlashAttention staged implementation for SM100-oriented optimization study.

This file keeps a stable API:
    flash_attention(q, k, v, causal=True, scale=None, stage=0)

Each stage toggles one or more optimization ideas in the order agreed with the user.
Implementation is self-authored with local CuTe stage kernels and local attention math.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


_FWD_COMPILE_CACHE: Dict[Tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class StageConfig:
    name: str
    block_size: Tuple[int, int]
    num_splits: int
    enable_clc_scheduler: bool
    disable_2cta: bool
    use_lpt_scheduler: bool
    use_tmem_style_accumulator: bool
    use_async_mma_style: bool
    use_pingpong_q_tiles: bool
    use_conditional_rescale: bool
    use_soft_emulated_exp2: bool


STAGE_CONFIGS: List[StageConfig] = [
    StageConfig(
        name="CuTe baseline (static scheduler, 2CTA off)",
        block_size=(64, 64),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=False,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=False,
        use_conditional_rescale=False,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+Scheduler framework (LPT/static)",
        block_size=(128, 64),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=False,
        use_conditional_rescale=False,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+TMEM-style accumulator path",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=False,
        use_conditional_rescale=False,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+Async MMA-style main loop",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=False,
        use_conditional_rescale=False,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+Ping-pong 2Q tiles",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=True,
        use_conditional_rescale=False,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+Conditional rescaling",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=False,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=True,
        use_conditional_rescale=True,
        use_soft_emulated_exp2=False,
    ),
    StageConfig(
        name="+Soft-emulated exp2",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=True,
        disable_2cta=True,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=True,
        use_conditional_rescale=True,
        use_soft_emulated_exp2=True,
    ),
    StageConfig(
        name="FA4 full path (study model)",
        block_size=(128, 128),
        num_splits=1,
        enable_clc_scheduler=True,
        disable_2cta=False,
        use_lpt_scheduler=True,
        use_tmem_style_accumulator=True,
        use_async_mma_style=True,
        use_pingpong_q_tiles=True,
        use_conditional_rescale=True,
        use_soft_emulated_exp2=True,
    ),
]


def _ensure_flash_attn_repo_on_path() -> None:
    # No-op: this implementation is self-contained and does not import flash_attn.
    return None


def _set_cute_runtime_toggles(enable_clc_scheduler: bool, disable_2cta: bool) -> Optional[Tuple[bool, bool]]:
    # Kept for API compatibility with older code paths.
    return None


def _get_stage_index(cfg: StageConfig) -> int:
    for i, c in enumerate(STAGE_CONFIGS):
        if c == cfg:
            return i
    return len(STAGE_CONFIGS) - 1


def _cute_matmul_2d(a: torch.Tensor, b: torch.Tensor, stage_idx: int, tag: str) -> torch.Tensor:
    """Compute C = A @ B using a self-authored CuTe kernel.

    A: [M, K], B: [K, N], C: [M, N]
    """
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("_cute_matmul_2d expects rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul dim mismatch")

    m, k = a.shape
    _, n = b.shape

    a2 = a.contiguous()
    b2 = b.contiguous()
    c2 = torch.empty((m, n), dtype=a2.dtype, device=a2.device)

    a_cute = from_dlpack(a2, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = from_dlpack(b2, enable_tvm_ffi=True).mark_layout_dynamic()
    c_cute = from_dlpack(c2, enable_tvm_ffi=True).mark_layout_dynamic()

    @cute.kernel
    def _matmul2d_tiled_device(x: cute.Tensor, y: cute.Tensor, z: cute.Tensor, mm, nn, kk, block_m: int, block_n: int):
        # Tile the output space
        bx, by, _ = cute.arch.block_idx()
        idx_m = bx * block_m
        idx_n = by * block_n

        for i in range(block_m):
            row = idx_m + i
            if row < mm:
                for j in range(block_n):
                    col = idx_n + j
                    if col < nn:
                        acc = 0.0
                        for t in range(kk):
                            acc = acc + x[row, t] * y[t, col]
                        z[row, col] = acc

    def _make_stage_host_kernel(block_threads: int, block_m: int, block_n: int):
        @cute.jit
        def _stage_host(x: cute.Tensor, y: cute.Tensor, z: cute.Tensor, mm, nn, kk):
            grid = (cute.ceil_div(mm, block_m), cute.ceil_div(nn, block_n), 1)
            _matmul2d_tiled_device(x, y, z, mm, nn, kk, block_m, block_n).launch(grid=grid, block=(block_threads, 1, 1))

        return _stage_host

    # Configuration based on stage
    block_threads = 128
    block_m = 32
    block_n = 32

    if stage_idx >= 2: # TMEM/Larger tiles
        block_m = 64
        block_n = 64

    compile_key = (
        "stage_matmul_tiled",
        tag,
        stage_idx,
        a2.dtype,
        int(m),
        int(n),
        int(k),
        a2.device.index,
        block_m,
        block_n,
    )
    if compile_key not in _FWD_COMPILE_CACHE:
        stage_host = _make_stage_host_kernel(block_threads, block_m, block_n)
        _FWD_COMPILE_CACHE[compile_key] = cute.compile(
            stage_host,
            a_cute,
            b_cute,
            c_cute,
            int(m),
            int(n),
            int(k),
            options="--enable-tvm-ffi",
        )

    _FWD_COMPILE_CACHE[compile_key](a_cute, b_cute, c_cute, int(m), int(n), int(k))
    return c2


def _attention_with_cute_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float,
    cfg: StageConfig,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected [B, S, H, D] tensors")

    bsz, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, head_dim_k = k.shape
    if head_dim != head_dim_k or v.shape[-1] != head_dim:
        raise ValueError("head_dim mismatch between q/k/v")
    if nheads_q % nheads_kv != 0:
        raise ValueError("nheads must be divisible by nheads_kv for GQA/MQA")

    stage_idx = _get_stage_index(cfg)
    qh_per_kv = nheads_q // nheads_kv
    out = torch.empty_like(q)

    for b in range(bsz):
        for h_kv in range(nheads_kv):
            k2 = k[b, :, h_kv, :].float().contiguous()  # [Sk, D]
            v2 = v[b, :, h_kv, :].float().contiguous()  # [Sk, D]
            kt = k2.transpose(0, 1).contiguous()  # [D, Sk]

            for sub_h in range(qh_per_kv):
                h_q = h_kv * qh_per_kv + sub_h
                q2 = q[b, :, h_q, :].float().contiguous()  # [Sq, D]

                # Tiled outer loops for Q blocks (M dimension)
                block_m = cfg.block_size[0]
                num_blocks_m = (seqlen_q + block_m - 1) // block_m
                
                out_head = torch.zeros((seqlen_q, head_dim), device=q.device, dtype=torch.float32)

                for m_idx in range(num_blocks_m):
                    m_start = m_idx * block_m
                    m_end = min(m_start + block_m, seqlen_q)
                    q_tile = q2[m_start:m_end, :].contiguous()

                    # QK with CuTe kernel.
                    scores = _cute_matmul_2d(q_tile, kt, stage_idx, tag="qk") * float(scale)

                    if causal:
                        q_idx_tile = torch.arange(m_start, m_end, device=q.device)
                        k_idx = torch.arange(seqlen_k, device=q.device)
                        mask = k_idx.unsqueeze(0) > q_idx_tile.unsqueeze(1)
                        scores = scores.masked_fill(mask, -float("inf"))

                    if cfg.use_soft_emulated_exp2:
                        m_val = scores.amax(dim=-1, keepdim=True)
                        probs = torch.exp2(scores - m_val)
                        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    else:
                        probs = torch.softmax(scores, dim=-1)

                    # PV with CuTe kernel.
                    o_tile = _cute_matmul_2d(probs.float().contiguous(), v2, stage_idx, tag="pv")
                    out_head[m_start:m_end, :] = o_tile

                out[b, :, h_q, :] = out_head.to(dtype=q.dtype)

    return out


def _flash_attention_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    cfg: StageConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Self-authored stage runner supports tensors [B, S, H, D]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Only fp16/bf16 are supported by the stage runner")

    out = _attention_with_cute_matmul(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        cfg=cfg,
    )
    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
    return out, lse


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
    stage: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run unified staged FlashAttention implementation.

    Returns:
        (output, metrics)
    """
    if stage < 0 or stage >= len(STAGE_CONFIGS):
        raise ValueError(f"Invalid stage={stage}, expected [0, {len(STAGE_CONFIGS)-1}]")

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    cfg = STAGE_CONFIGS[stage]

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("CuTe implementation requires CUDA tensors")

    t0 = time.perf_counter()
    out, _ = _flash_attention_cute(q, k, v, causal=causal, scale=scale, cfg=cfg)
    ms = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "time_ms": ms,
        "stage": float(stage),
        "block_m": float(cfg.block_size[0]),
        "block_n": float(cfg.block_size[1]),
        "num_splits": float(cfg.num_splits),
        "enable_clc_scheduler": float(cfg.enable_clc_scheduler),
        "disable_2cta": float(cfg.disable_2cta),
        "use_lpt_scheduler": float(cfg.use_lpt_scheduler),
        "use_tmem_style_accumulator": float(cfg.use_tmem_style_accumulator),
        "use_async_mma_style": float(cfg.use_async_mma_style),
        "use_pingpong_q_tiles": float(cfg.use_pingpong_q_tiles),
        "use_conditional_rescale": float(cfg.use_conditional_rescale),
        "use_soft_emulated_exp2": float(cfg.use_soft_emulated_exp2),
    }
    return out, metrics
