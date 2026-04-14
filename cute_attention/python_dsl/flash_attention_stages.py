#!/usr/bin/env python3
"""
Unified FlashAttention staged implementation for SM100-oriented optimization study.

This file keeps a stable API:
    flash_attention(q, k, v, causal=True, scale=None, stage=0)

Each stage toggles one or more optimization ideas in the order agreed with the user.
Implementation is backed by FlashAttention CuTe kernels (SM100 path), not a PyTorch loop.
"""

from __future__ import annotations

import ctypes
import math
import os
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


_CUDART_PRELOAD_DONE = False


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


def _candidate_cudart_paths() -> List[Path]:
    candidates: List[Path] = []

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "lib64" / "libcudart.so.12")

    candidates.append(Path("/usr/local/cuda/lib64/libcudart.so.12"))

    try:
        torch_site = Path(torch.__file__).resolve().parent
        candidates.append(torch_site / "lib" / "libcudart.so.12")
        candidates.append(torch_site.parent / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12")
    except Exception:
        pass

    for p in list(sys.path):
        if "site-packages" in p:
            candidates.append(Path(p) / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12")

    # Keep order and remove duplicates.
    unique: List[Path] = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            unique.append(c)
    return unique


def _preload_cudart() -> None:
    global _CUDART_PRELOAD_DONE
    if _CUDART_PRELOAD_DONE:
        return

    errors: List[str] = []
    for cand in _candidate_cudart_paths():
        if not cand.exists():
            continue
        try:
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            _CUDART_PRELOAD_DONE = True
            return
        except OSError as e:
            errors.append(f"{cand}: {e}")

    # Fallback: rely on system linker path.
    try:
        ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
        _CUDART_PRELOAD_DONE = True
        return
    except OSError as e:
        errors.append(f"libcudart.so.12: {e}")

    checked = "\n  - ".join(str(p) for p in _candidate_cudart_paths())
    details = "\n  - " + "\n  - ".join(errors) if errors else ""
    raise RuntimeError(
        "Cannot load libcudart.so.12 for FlashAttention CuTe path.\n"
        "Checked candidate paths:\n"
        f"  - {checked}\n"
        f"Load errors:{details}\n"
        "Fix options:\n"
        "1) export LD_LIBRARY_PATH=<cuda_or_site_packages_nvidia_cuda_runtime_lib>:$LD_LIBRARY_PATH\n"
        "2) ensure CUDA 12 runtime is installed and visible to dynamic loader."
    )


def _flash_attention_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    cfg: StageConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_flash_attn_repo_on_path()
    _preload_cudart()
    from flash_attn.cute.interface import flash_attn_func

    prev_toggles = _set_cute_runtime_toggles(
        enable_clc_scheduler=cfg.enable_clc_scheduler,
        disable_2cta=cfg.disable_2cta,
    )
    try:
        out, lse = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            num_splits=cfg.num_splits,
            block_size=cfg.block_size,
            return_lse=True,
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
