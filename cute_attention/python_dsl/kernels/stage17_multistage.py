"""
Stage17: autotuned family entrypoint for deeper multistage attention.

Stage16 is a validated 256-thread warp-specialized double-buffer kernel.
Stage13 already supports deeper K/V staging (`num_stages_kv > 2`) but does not
use the Stage15/16 warp-specialized schedule.  Stage17 bridges the two:

  - prefer the Stage16-style double-buffer backend when it is the best valid fit
  - fall back to the Stage13 multistage backend for deeper staging or smaller CTAs

This makes Stage17 a useful forward-looking public stage today while we keep
developing a dedicated warp-specialized multistage kernel behind the same API.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .common import AttentionConfig, HAS_CUTE, cutlass, require_torch, torch, validate_qkv
from .reference import causal_attention_reference
from .stage13_multistage import _stage13_forward_impl
from .stage16_multistage import _stage16_forward_impl


if HAS_CUTE:
    from .stage13_multistage import Stage13FlashAttentionAmpere
    from .stage16_multistage import Stage16FlashAttentionSm90DoubleBuffer


_STAGE17_AUTOTUNE_CACHE: dict = {}
_STAGE17_AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "stage17_autotune.json"


def _make_stage17_config(
    config: AttentionConfig,
    *,
    block_m: int,
    block_n: int,
    num_threads: int,
    num_stages_kv: int,
) -> AttentionConfig:
    return AttentionConfig(
        softmax_scale=config.softmax_scale,
        causal=config.causal,
        block_m=block_m,
        block_n=block_n,
        num_threads=num_threads,
        num_stages_kv=num_stages_kv,
        autotune=False,
    )


def _candidate_values(preferred: int, values: list[int], *, limit: int | None = None) -> list[int]:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value in ordered:
            continue
        if limit is not None and value > limit:
            continue
        ordered.append(value)
    return ordered


def _stage17_autotune_cache_key(config: AttentionConfig, q) -> str:
    device_name = torch.cuda.get_device_name(q.device)
    return "|".join(
        [
            device_name,
            str(tuple(q.shape)),
            str(q.dtype),
            str(config.block_m),
            str(config.block_n),
            str(config.num_threads),
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


def _stage17_backend_name(config: AttentionConfig, head_dim: int) -> str:
    if (
        config.num_threads == 256
        and config.num_stages_kv == 2
        and Stage16FlashAttentionSm90DoubleBuffer.can_implement(
            cutlass.Float16,
            head_dim,
            config.block_m,
            config.block_n,
            config.num_threads,
            True,
        )
    ):
        return "stage16_double_buffer"

    if config.num_stages_kv in {2, 3, 4, 5} and Stage13FlashAttentionAmpere.can_implement(
        cutlass.Float16,
        head_dim,
        config.block_m,
        config.block_n,
        config.num_threads,
        config.num_stages_kv,
        True,
    ):
        return "stage13_multistage"

    raise ValueError(
        "stage17 could not find a valid backend for "
        f"(block_m={config.block_m}, block_n={config.block_n}, "
        f"num_threads={config.num_threads}, num_stages_kv={config.num_stages_kv})."
    )
def _stage17_candidate_matches_reference(q, k, v, reference_out, tuned: AttentionConfig) -> bool:
    candidate_out = _stage17_forward_impl(q, k, v, tuned)
    torch.testing.assert_close(candidate_out, reference_out, rtol=4e-2, atol=4e-2)
    return True


def autotune_stage17_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    """Autotune stage17 across tile sizes, stage depth, and CTA thread count."""
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage17 autotune only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage17 autotune requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage17 autotune currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    preferred_stages = config.num_stages_kv or 3
    if preferred_stages < 2:
        raise ValueError(f"stage17 expects num_stages_kv >= 2, got {preferred_stages}.")

    cache_key = (
        tuple(q.shape),
        str(q.dtype),
        config.block_m,
        config.block_n,
        config.num_threads,
        preferred_stages,
    )
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
            num_threads=int(cached_disk["num_threads"]),
            num_stages_kv=int(cached_disk["num_stages_kv"]),
        )
        _STAGE17_AUTOTUNE_CACHE[cache_key] = tuned
        return tuned

    reference_out = causal_attention_reference(q, k, v, replace(config, autotune=False))
    block_m_values = _candidate_values(config.block_m, [128, 96, 64, 48, 32], limit=seq_len)
    block_n_values = _candidate_values(config.block_n, [256, 192, 128, 96, 64], limit=seq_len)
    thread_values = _candidate_values(config.num_threads or 256, [256, 128])
    stage_values = _candidate_values(preferred_stages, [5, 4, 3, 2], limit=5)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    best_config = None
    best_ms = None

    for num_threads in thread_values:
        for num_stages_kv in stage_values:
            for block_m in block_m_values:
                for block_n in block_n_values:
                    tuned = _make_stage17_config(
                        config,
                        block_m=block_m,
                        block_n=block_n,
                        num_threads=num_threads,
                        num_stages_kv=num_stages_kv,
                    )
                    try:
                        _stage17_candidate_matches_reference(q, k, v, reference_out, tuned)
                        for _ in range(warmup):
                            _stage17_forward_impl(q, k, v, tuned)
                        torch.cuda.synchronize()
                        elapsed = 0.0
                        for _ in range(repeat):
                            start_event.record()
                            _stage17_forward_impl(q, k, v, tuned)
                            end_event.record()
                            torch.cuda.synchronize()
                            elapsed += start_event.elapsed_time(end_event)
                        elapsed /= repeat
                    except (AssertionError, ValueError, RuntimeError):
                        continue
                    if best_ms is None or elapsed < best_ms:
                        best_ms = elapsed
                        best_config = tuned

    if best_config is None:
        raise ValueError("stage17 autotune failed to find a valid config.")

    _STAGE17_AUTOTUNE_CACHE[cache_key] = best_config
    disk_cache[cache_key_disk] = {
        "block_m": best_config.block_m,
        "block_n": best_config.block_n,
        "num_threads": best_config.num_threads,
        "num_stages_kv": best_config.num_stages_kv,
    }
    _save_stage17_autotune_cache_to_disk(disk_cache)
    return best_config


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

    head_dim = q.shape[-1]
    normalized = replace(config, autotune=False, num_stages_kv=requested_stages)
    backend = _stage17_backend_name(normalized, head_dim)
    if backend == "stage16_double_buffer":
        delegated = replace(normalized, num_threads=256, num_stages_kv=2)
        return _stage16_forward_impl(q, k, v, delegated)
    return _stage13_forward_impl(q, k, v, normalized)


def stage17_forward(q, k, v, config: AttentionConfig | None = None):
    """Stage17: autotuned family entrypoint for deeper multistage attention."""
    config = config or AttentionConfig(block_m=64, block_n=128, num_threads=256, num_stages_kv=3)
    tuned = replace(config, autotune=False)
    if not tuned.num_stages_kv:
        tuned = replace(tuned, num_stages_kv=3)
    if not tuned.num_threads:
        tuned = replace(tuned, num_threads=256)
    if config.autotune:
        tuned = autotune_stage17_config(q, k, v, tuned)
    return _stage17_forward_impl(q, k, v, tuned)
