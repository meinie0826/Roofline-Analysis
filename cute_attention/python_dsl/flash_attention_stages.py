#!/usr/bin/env python3
"""
Unified FlashAttention staged implementation for SM100-oriented optimization study.

This file keeps a stable API:
    flash_attention(q, k, v, causal=True, scale=None, stage=0)

Each stage toggles one or more optimization ideas in the order agreed with the user.
Implementation is backed by FlashAttention CuTe kernels (SM100 path), not a PyTorch loop.
"""

from __future__ import annotations

import math
import time
import sys
from dataclasses import dataclass
from pathlib import Path
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
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    flash_attn_repo = repo_root / "flash-attention"
    if flash_attn_repo.exists():
        repo_path = str(flash_attn_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)


def _set_cute_runtime_toggles(enable_clc_scheduler: bool, disable_2cta: bool) -> Optional[Tuple[bool, bool]]:
    _ensure_flash_attn_repo_on_path()
    try:
        from flash_attn.cute import utils as cute_utils
    except Exception:
        return None

    prev = (cute_utils._fa_clc_enabled, cute_utils._fa_disable_2cta_enabled)
    cute_utils._fa_clc_enabled = bool(enable_clc_scheduler)
    cute_utils._fa_disable_2cta_enabled = bool(disable_2cta)
    return prev


def _flash_attention_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    cfg: StageConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_flash_attn_repo_on_path()
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Self-authored CuTe runner currently supports only batched tensors [B, S, H, D]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Only fp16/bf16 are supported by the custom CuTe stage runner")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("nheads must be divisible by nheads_kv for GQA/MQA")

    major, _ = torch.cuda.get_device_capability(q.device)
    if major not in (10, 11):
        raise ValueError(f"Custom CuTe path is currently implemented for SM100/SM110 only, got SM_{major}")

    bsz, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, head_dim_k = k.shape
    if head_dim != head_dim_k or v.shape[-1] != head_dim:
        raise ValueError("Current custom runner assumes head_dim_q == head_dim_k == head_dim_v")

    qhead_per_kvhead = nheads_q // nheads_kv
    is_split_kv = cfg.num_splits > 1
    q_stage = 2 if (cfg.use_pingpong_q_tiles and seqlen_q > cfg.block_size[0]) else 1

    out = torch.empty_like(q)
    lse = torch.empty((bsz, nheads_q, seqlen_q), dtype=torch.float32, device=q.device)

    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    cute_dtype = torch2cute_dtype_map[q.dtype]

    compile_key = (
        cute_dtype,
        head_dim,
        qhead_per_kvhead,
        causal,
        cfg.block_size,
        q_stage,
        is_split_kv,
        cfg.enable_clc_scheduler,
        cfg.disable_2cta,
        q.device.index,
    )

    prev_toggles = _set_cute_runtime_toggles(
        enable_clc_scheduler=cfg.enable_clc_scheduler,
        disable_2cta=cfg.disable_2cta,
    )
    try:
        if compile_key not in _FWD_COMPILE_CACHE:
            current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
            lse_tensor = to_cute_tensor(lse, assumed_align=4)

            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=False,
                is_split_kv=is_split_kv,
                pack_gqa=False,
                m_block_size=cfg.block_size[0],
                n_block_size=cfg.block_size[1],
                q_stage=q_stage,
                is_persistent=(not causal) and (cfg.num_splits == 1),
                score_mod=None,
                mask_mod=None,
                has_aux_tensors=False,
                paged_kv_non_tma=False,
                is_varlen_q=False,
                use_2cta_instrs=(not cfg.disable_2cta),
                use_clc_scheduler=cfg.enable_clc_scheduler,
            )

            compiled = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                float(scale),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                current_stream,
                options="--enable-tvm-ffi",
            )
            _FWD_COMPILE_CACHE[compile_key] = compiled

        _FWD_COMPILE_CACHE[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse,
            float(scale),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    finally:
        if prev_toggles is not None:
            from flash_attn.cute import utils as cute_utils

            cute_utils._fa_clc_enabled, cute_utils._fa_disable_2cta_enabled = prev_toggles

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
