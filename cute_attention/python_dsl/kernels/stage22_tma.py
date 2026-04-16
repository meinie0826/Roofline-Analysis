from __future__ import annotations

from dataclasses import replace

from .common import (
    AttentionConfig,
    HAS_CUTE,
    cute,
    from_dlpack,
    require_torch,
    torch,
    validate_qkv,
)
from .stage22_tma_backend import make_stage22_host


MAX_SEQ_LEN_FOR_STAGE22_CUTE = 4096
_STAGE22_COMPILED_CACHE = {}


def _normalize_stage22_config(config: AttentionConfig | None) -> AttentionConfig:
    cfg = replace(config or AttentionConfig(), autotune=False)
    num_threads = cfg.num_threads if cfg.num_threads > 0 else 256
    if num_threads % 32 != 0:
        raise ValueError("stage22 requires num_threads to be divisible by 32.")
    return replace(cfg, num_threads=num_threads)


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
    if not (q.dtype in [torch.float16, torch.bfloat16]):
        raise ValueError(f"stage22 currently only supports fp16/bf16 inputs, got {q.dtype}.")
    return _normalize_stage22_config(config)


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    validate_qkv(q, k, v)
    if not HAS_CUTE:
        raise RuntimeError("stage22 requires cutlass.cute.")

    cfg = _normalize_stage22_config(config)
    if not cfg.causal:
        raise ValueError("stage22 only supports causal attention.")
    if q.dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(f"stage22 currently only supports fp16/bf16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE22_CUTE:
        raise ValueError(f"stage22 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE22_CUTE}, got {seq_len}.")

    scale = cfg.resolve_scale(head_dim)
    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = q_flat.new_zeros(q_flat.shape)

    q_cute = from_dlpack(q_flat, assumed_align=16).mark_layout_dynamic()
    k_cute = from_dlpack(k_flat, assumed_align=16).mark_layout_dynamic()
    v_cute = from_dlpack(v_flat, assumed_align=16).mark_layout_dynamic()
    o_cute = from_dlpack(o_flat, assumed_align=16).mark_layout_dynamic()

    cache_key = (tuple(q_flat.shape), str(q_flat.dtype), cfg.num_threads)
    compiled = _stage22_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, cfg.num_threads)
    compiled(q_cute, k_cute, v_cute, o_cute, scale)
    return o_flat.reshape(batch, heads, seq_len, head_dim)


def _stage22_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, num_threads):
    compiled = _STAGE22_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[0][1]
        head_dim = cache_key[0][2]
        stage22_host = make_stage22_host(seq_len=seq_len, head_dim=head_dim, num_threads=num_threads)
        compiled = cute.compile(
            stage22_host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            scale,
        )
        _STAGE22_COMPILED_CACHE[cache_key] = compiled
    return compiled
