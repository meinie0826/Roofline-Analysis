from __future__ import annotations

from dataclasses import replace

from .common import AttentionConfig, require_torch, validate_qkv, torch
from .stage10_hybrid import stage10_forward

MAX_SEQ_LEN_FOR_STAGE22_CUTE = 4096
_STAGE22_AUTOTUNE_CACHE = {}


def _make_stage22_config(
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


def _stage22_autotune_cache_key(config: AttentionConfig, q) -> tuple:
    return (
        tuple(q.shape),
        str(q.dtype),
        config.block_m,
        config.block_n,
        config.num_threads,
        config.num_stages_kv,
    )


def _stage22_candidate_values(preferred: int, values: list[int], *, limit: int) -> list[int]:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value > limit or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _stage22_find_safe_runtime_config(head_dim: int, seq_len: int, config: AttentionConfig) -> AttentionConfig:
    _ = head_dim
    m_candidates = _stage22_candidate_values(config.block_m or 128, [128, 64], limit=max(seq_len, 1))
    n_candidates = _stage22_candidate_values(config.block_n or 128, [128, 64], limit=max(seq_len, 1))
    for block_m in m_candidates:
        for block_n in n_candidates:
            if 256 % block_m == 0:
                return _make_stage22_config(config, block_m=block_m, block_n=block_n, num_stages_kv=3)
    # Hard fallback that always satisfies stage10 requirements.
    return _make_stage22_config(config, block_m=64, block_n=64, num_stages_kv=3)


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
    config = replace(config or AttentionConfig(), num_threads=256)
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage22 autotune only supports causal attention.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage22 autotune currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    cache_key = _stage22_autotune_cache_key(config, q)
    cached = _STAGE22_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tuned = _stage22_find_safe_runtime_config(head_dim, seq_len, config)
    _STAGE22_AUTOTUNE_CACHE[cache_key] = tuned
    return tuned


def _stage22_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage22 only supports causal attention.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage22 currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE22_CUTE:
        raise ValueError(f"stage22 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE22_CUTE}, got {seq_len}.")

    stable = replace(config, num_threads=256, num_stages_kv=(config.num_stages_kv or 3), autotune=False)
    stable = _stage22_find_safe_runtime_config(head_dim, seq_len, stable)
    return stage10_forward(q, k, v, stable)


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256)
    if config.autotune:
        tuned = autotune_stage22_config(q, k, v, tuned)
    elif tuned.num_stages_kv == 0:
        tuned = replace(tuned, num_stages_kv=3)
    return _stage22_forward_impl(q, k, v, tuned)
