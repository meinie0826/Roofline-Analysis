from dataclasses import replace

import cuda.bindings.driver as cuda

from .common import (
    HAS_CUTE,
    AttentionConfig,
    cutlass,
    cute,
    from_dlpack,
    require_torch,
    torch,
    validate_qkv,
)
from .stage22_tma_backend import Stage22FlashAttentionTmaExperimental

MAX_SEQ_LEN_FOR_STAGE22_CUTE = 4096
_STAGE22_COMPILED_CACHE = {}


def _stage22_can_implement(head_dim: int, config: AttentionConfig) -> bool:
    return Stage22FlashAttentionTmaExperimental.can_implement(
        cutlass.Float16,
        head_dim,
        config.block_m,
        config.block_n,
        config.num_threads,
        config.num_stages_kv,
        True,
    )


def _stage22_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage22 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage22 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage22 currently only supports fp16 inputs, got {q.dtype}.")

    _, _, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE22_CUTE:
        raise ValueError(f"stage22 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE22_CUTE}, got {seq_len}.")
    if config.num_stages_kv not in {0, 2, 3}:
        raise ValueError(f"stage22 currently supports num_stages_kv in {{2, 3}}, got {config.num_stages_kv}.")

    normalized = replace(config, num_threads=256, num_stages_kv=(config.num_stages_kv or 3), autotune=False)
    if not _stage22_can_implement(head_dim, normalized):
        raise ValueError("stage22 config is not supported by the TMA experimental kernel constraints.")

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
    compiled = _stage22_compile(
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


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256)
    if config.autotune:
        tuned = autotune_stage22_config(q, k, v, tuned)
    elif tuned.num_stages_kv == 0:
        tuned = replace(tuned, num_stages_kv=3)

    return _stage22_forward_impl(q, k, v, replace(tuned, autotune=False, num_threads=256))


def autotune_stage22_config(q, k, v, config: AttentionConfig | None = None):
    raise NotImplementedError(
        "stage22_tma autotune will be added after the first runnable TMA load path lands."
    )


def _stage22_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, current_stream, head_dim, block_m, block_n, num_threads, num_stages_kv):
    compiled = _STAGE22_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        kernel = Stage22FlashAttentionTmaExperimental(
            head_dim=head_dim,
            m_block_size=block_m,
            n_block_size=block_n,
            num_threads=num_threads,
            num_stages_kv=num_stages_kv,
            is_causal=True,
        )
        compiled = cute.compile(kernel, q_cute, k_cute, v_cute, o_cute, scale, current_stream)
        _STAGE22_COMPILED_CACHE[cache_key] = compiled
    return compiled


__all__ = [
    "Stage22FlashAttentionTmaExperimental",
    "stage22_forward",
    "autotune_stage22_config",
]
