import json
from dataclasses import replace
from pathlib import Path

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
from .stage20_warpspec_backend import Stage20FlashAttentionWarpSpecExtreme

MAX_SEQ_LEN_FOR_STAGE20_CUTE = 4096
_STAGE20_COMPILED_CACHE = {}
_STAGE20_AUTOTUNE_CACHE = {}
_STAGE20_AUTOTUNE_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "stage20_autotune.json"


def _make_stage20_config(config: AttentionConfig, *, block_m: int, block_n: int, num_stages_kv: int) -> AttentionConfig:
    return AttentionConfig(
        softmax_scale=config.softmax_scale,
        causal=config.causal,
        block_m=block_m,
        block_n=block_n,
        num_threads=256,
        num_stages_kv=num_stages_kv,
        autotune=False,
    )


def _stage20_autotune_cache_key(config: AttentionConfig, q) -> str:
    device_name = torch.cuda.get_device_name(q.device)
    return "|".join(
        [
            device_name,
            str(tuple(q.shape)),
            str(q.dtype),
            str(config.block_m),
            str(config.block_n),
            str(config.num_stages_kv),
        ]
    )


def _load_stage20_autotune_cache_from_disk() -> dict[str, dict[str, int]]:
    if not _STAGE20_AUTOTUNE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_STAGE20_AUTOTUNE_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_stage20_autotune_cache_to_disk(entries: dict[str, dict[str, int]]) -> None:
    _STAGE20_AUTOTUNE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STAGE20_AUTOTUNE_CACHE_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))


def _stage20_candidate_values(preferred: int, values: list[int], *, limit: int) -> list[int]:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value > limit or value in ordered:
            continue
        ordered.append(value)
    return ordered


def _stage20_can_implement(head_dim: int, config: AttentionConfig) -> bool:
    return Stage20FlashAttentionWarpSpecExtreme.can_implement(
        cutlass.Float16,
        head_dim,
        config.block_m,
        config.block_n,
        config.num_threads,
        config.num_stages_kv,
        True,
    )


def autotune_stage20_config(
    q,
    k,
    v,
    config: AttentionConfig | None = None,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> AttentionConfig:
    require_torch()
    config = replace(config or AttentionConfig(), num_threads=256)
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage20 autotune only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage20 autotune requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage20 autotune currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    cache_key = (
        tuple(q.shape),
        str(q.dtype),
        config.block_m,
        config.block_n,
        config.num_stages_kv,
    )
    cached = _STAGE20_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        if _stage20_can_implement(head_dim, cached):
            return cached
        _STAGE20_AUTOTUNE_CACHE.pop(cache_key, None)

    cache_key_disk = _stage20_autotune_cache_key(config, q)
    disk_cache = _load_stage20_autotune_cache_from_disk()
    cached_disk = disk_cache.get(cache_key_disk)
    if cached_disk is not None:
        tuned = _make_stage20_config(
            config,
            block_m=int(cached_disk["block_m"]),
            block_n=int(cached_disk["block_n"]),
            num_stages_kv=int(cached_disk["num_stages_kv"]),
        )
        if _stage20_can_implement(head_dim, tuned):
            _STAGE20_AUTOTUNE_CACHE[cache_key] = tuned
            return tuned
        disk_cache.pop(cache_key_disk, None)
        _save_stage20_autotune_cache_to_disk(disk_cache)

    block_m_values = _stage20_candidate_values(config.block_m, [128, 96, 64], limit=seq_len)
    block_n_values = _stage20_candidate_values(config.block_n, [192, 128, 64], limit=seq_len)
    stage_values = _stage20_candidate_values(config.num_stages_kv or 3, [3, 2], limit=3)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    best_config = None
    best_ms = None

    for num_stages_kv in stage_values:
        for block_m in block_m_values:
            for block_n in block_n_values:
                tuned = _make_stage20_config(
                    config,
                    block_m=block_m,
                    block_n=block_n,
                    num_stages_kv=num_stages_kv,
                )
                if not _stage20_can_implement(head_dim, tuned):
                    continue
                try:
                    for _ in range(warmup):
                        _stage20_forward_impl(q, k, v, tuned)

                    torch.cuda.synchronize()
                    elapsed = 0.0
                    for _ in range(repeat):
                        start_event.record()
                        _stage20_forward_impl(q, k, v, tuned)
                        end_event.record()
                        torch.cuda.synchronize()
                        elapsed += start_event.elapsed_time(end_event)
                    elapsed /= repeat
                except ValueError:
                    continue

                if best_ms is None or elapsed < best_ms:
                    best_ms = elapsed
                    best_config = tuned

    if best_config is None:
        raise ValueError(
            f"stage20 autotune failed to find a valid config for shape={(batch, heads, seq_len, head_dim)}."
        )

    _STAGE20_AUTOTUNE_CACHE[cache_key] = best_config
    disk_cache[cache_key_disk] = {
        "block_m": best_config.block_m,
        "block_n": best_config.block_n,
        "num_stages_kv": best_config.num_stages_kv,
    }
    _save_stage20_autotune_cache_to_disk(disk_cache)
    return best_config


def _stage20_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage20 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage20 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage20 currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE20_CUTE:
        raise ValueError(f"stage20 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE20_CUTE}, got {seq_len}.")
    if config.num_stages_kv not in {0, 2, 3}:
        raise ValueError(f"stage20 currently supports num_stages_kv in {{2, 3}}, got {config.num_stages_kv}.")

    normalized = replace(config, num_threads=256, num_stages_kv=(config.num_stages_kv or 3), autotune=False)
    if not _stage20_can_implement(head_dim, normalized):
        raise ValueError("stage20 config is not supported by the extreme warpspec kernel constraints.")

    q_perm = q.permute(0, 2, 1, 3).contiguous()
    k_perm = k.permute(0, 2, 1, 3).contiguous()
    v_perm = v.permute(0, 2, 1, 3).contiguous()
    o_perm = torch.empty_like(q_perm)

    q_cute = from_dlpack(q_perm, assumed_align=16)
    k_cute = from_dlpack(k_perm, assumed_align=16)
    v_cute = from_dlpack(v_perm, assumed_align=16)
    o_cute = from_dlpack(o_perm, assumed_align=16)
    scale = normalized.resolve_scale(head_dim)
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    cache_key = (
        tuple(q_perm.shape),
        str(q_perm.dtype),
        normalized.block_m,
        normalized.block_n,
        normalized.num_threads,
        normalized.num_stages_kv,
    )
    compiled = _stage20_compile(
        cache_key,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        scale,
        current_stream,
        head_dim,
        normalized.block_m,
        normalized.block_n,
        normalized.num_threads,
        normalized.num_stages_kv,
    )
    compiled(q_cute, k_cute, v_cute, o_cute, scale, current_stream)
    return o_perm.permute(0, 2, 1, 3).contiguous()


def stage20_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256)
    if config.autotune:
        tuned = autotune_stage20_config(q, k, v, tuned)
    elif tuned.num_stages_kv == 0:
        tuned = replace(tuned, num_stages_kv=3)

    return _stage20_forward_impl(q, k, v, replace(tuned, autotune=False, num_threads=256))


def _stage20_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, current_stream, head_dim, block_m, block_n, num_threads, num_stages_kv):
    compiled = _STAGE20_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        kernel = Stage20FlashAttentionWarpSpecExtreme(
            head_dim=head_dim,
            m_block_size=block_m,
            n_block_size=block_n,
            num_threads=num_threads,
            num_stages_kv=num_stages_kv,
            is_causal=True,
        )
        compiled = cute.compile(kernel, q_cute, k_cute, v_cute, o_cute, scale, current_stream)
        _STAGE20_COMPILED_CACHE[cache_key] = compiled
    return compiled
