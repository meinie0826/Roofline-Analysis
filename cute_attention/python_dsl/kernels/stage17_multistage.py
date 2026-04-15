"""
Stage17: dedicated entrypoint for the next SM90 warp-specialized multistage kernel.

This stage is intentionally introduced as a separate public surface so we can
iterate on deeper K/V staging without destabilizing the validated stage16
double-buffer baseline. Until the dedicated stage17 kernel lands, the forward
path conservatively reuses stage16's implementation while preserving stage17's
own config/defaults, tests, and benchmark hooks.
"""

from __future__ import annotations

from dataclasses import replace

from .common import AttentionConfig, HAS_CUTE, require_torch, torch, validate_qkv
from .stage16_multistage import _stage16_forward_impl, autotune_stage16_config


def _make_stage17_config(
    config: AttentionConfig,
    *,
    block_m: int,
    block_n: int,
    num_stages_kv: int,
) -> AttentionConfig:
    return AttentionConfig(
        softmax_scale=config.softmax_scale,
        causal=config.causal,
        block_m=block_m,
        block_n=block_n,
        num_threads=256,
        num_stages_kv=num_stages_kv,
        autotune=False,
    )


def autotune_stage17_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    """Temporary stage17 autotune: reuse stage16 block tuning, preserve stage17 API."""
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage17 autotune only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage17 autotune requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage17 autotune currently only supports fp16 inputs, got {q.dtype}.")

    requested_stages = config.num_stages_kv or 3
    if requested_stages < 2:
        raise ValueError(f"stage17 expects num_stages_kv >= 2, got {requested_stages}.")

    tuned_stage16 = autotune_stage16_config(
        q,
        k,
        v,
        replace(config, num_threads=256, num_stages_kv=2, autotune=False),
        warmup=warmup,
        repeat=repeat,
    )
    return _make_stage17_config(
        config,
        block_m=tuned_stage16.block_m,
        block_n=tuned_stage16.block_n,
        num_stages_kv=requested_stages,
    )


def _stage17_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage17 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage17 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage17 currently only supports fp16 inputs, got {q.dtype}.")

    requested_stages = config.num_stages_kv or 3
    if requested_stages < 2:
        raise ValueError(f"stage17 expects num_stages_kv >= 2, got {requested_stages}.")

    # Temporary compatibility path: keep stage17 as a separate surface while the
    # dedicated multistage kernel is being developed on top of the stage16 base.
    delegated = replace(config, num_threads=256, num_stages_kv=2, autotune=False)
    return _stage16_forward_impl(q, k, v, delegated)


def stage17_forward(q, k, v, config: AttentionConfig | None = None):
    """Stage17: independent entrypoint for the upcoming deeper multistage pipeline."""
    config = config or AttentionConfig(block_m=64, block_n=128, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256, autotune=False)
    if not tuned.num_stages_kv:
        tuned = replace(tuned, num_stages_kv=3)
    if config.autotune:
        tuned = autotune_stage17_config(q, k, v, tuned)
    return _stage17_forward_impl(q, k, v, tuned)
