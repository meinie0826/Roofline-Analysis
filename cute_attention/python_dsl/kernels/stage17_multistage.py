"""
Stage17: dedicated entrypoint for the next SM90 warp-specialized multistage kernel.

This stage is intentionally introduced as a separate public surface so we can
iterate on deeper K/V staging without destabilizing the validated stage16
double-buffer baseline. Until the dedicated stage17 kernel lands, the forward
path conservatively reuses stage16's implementation while preserving stage17's
own config/defaults, tests, and benchmark hooks.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .common import AttentionConfig, HAS_CUTE, require_torch, torch, validate_qkv
from .stage16_multistage import _stage16_forward_impl, autotune_stage16_config


_STAGE17_AUTOTUNE_CACHE: dict = {}
_STAGE17_AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "stage17_autotune.json"


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


def _stage17_autotune_cache_key(config: AttentionConfig, q) -> str:
    device_name = torch.cuda.get_device_name(q.device)
    return "|".join(
        [
            device_name,
            str(tuple(q.shape)),
            str(q.dtype),
            str(config.block_m),
            str(config.block_n),
            str(config.num_stages_kv or 3),
        ]
    )


def _load_stage17_autotune_cache_from_disk() -> dict[str, dict[str, int]]:
    if not _STAGE17_AUTOTUNE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_STAGE17_AUTOTUNE_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_stage17_autotune_cache_to_disk(entries: dict[str, dict[str, int]]) -> None:
    _STAGE17_AUTOTUNE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STAGE17_AUTOTUNE_CACHE_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))


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

    cache_key = (tuple(q.shape), str(q.dtype), config.block_m, config.block_n, requested_stages)
    cached = _STAGE17_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache_key_disk = _stage17_autotune_cache_key(config, q)
    disk_cache = _load_stage17_autotune_cache_from_disk()
    cached_disk = disk_cache.get(cache_key_disk)
    if cached_disk is not None:
        tuned = _make_stage17_config(
            config,
            block_m=int(cached_disk["block_m"]),
            block_n=int(cached_disk["block_n"]),
            num_stages_kv=int(cached_disk["num_stages_kv"]),
        )
        _STAGE17_AUTOTUNE_CACHE[cache_key] = tuned
        return tuned

    tuned_stage16 = autotune_stage16_config(
        q,
        k,
        v,
        replace(config, num_threads=256, num_stages_kv=2, autotune=False),
        warmup=warmup,
        repeat=repeat,
    )
    tuned = _make_stage17_config(
        config,
        block_m=tuned_stage16.block_m,
        block_n=tuned_stage16.block_n,
        num_stages_kv=requested_stages,
    )
    _STAGE17_AUTOTUNE_CACHE[cache_key] = tuned
    disk_cache[cache_key_disk] = {
        "block_m": tuned.block_m,
        "block_n": tuned.block_n,
        "num_stages_kv": tuned.num_stages_kv,
    }
    _save_stage17_autotune_cache_to_disk(disk_cache)
    return tuned


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
