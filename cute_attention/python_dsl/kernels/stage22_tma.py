from dataclasses import replace
import math

import cuda.bindings.driver as cuda

from .common import (
    AttentionConfig,
    HAS_CUTE,
    require_torch,
    torch,
    validate_qkv,
)
from .stage22_blackwell_backend import (
    BlackwellFusedMultiHeadAttentionForward,
    Int32,
    cutlass,
    from_dlpack,
    cute,
    fmha_utils,
)

MAX_SEQ_LEN_FOR_STAGE22_CUTE = 4096
_STAGE22_COMPILED_CACHE = {}
_LOG2_E = math.log2(math.e)


def _stage22_cache_key(q, config: AttentionConfig):
    return (
        tuple(q.shape),
        str(q.dtype),
        config.block_m,
        config.block_n,
        config.causal,
        torch.cuda.get_device_name(q.device),
    )


def _normalize_stage22_config(config: AttentionConfig | None) -> AttentionConfig:
    return replace(
        config or AttentionConfig(),
        causal=True,
        block_m=128,
        block_n=128,
        num_threads=512,
        num_stages_kv=0,
        autotune=False,
    )


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
    return _normalize_stage22_config(config)


def _stage22_forward_impl(q, k, v, config: AttentionConfig):
    require_torch()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("stage22 only supports causal attention.")
    if not HAS_CUTE:
        raise RuntimeError("stage22 requires cutlass.cute.")
    if q.dtype != torch.float16:
        raise ValueError(f"stage22 currently only supports fp16 inputs, got {q.dtype}.")

    batch, heads, seq_len, head_dim = q.shape
    if seq_len > MAX_SEQ_LEN_FOR_STAGE22_CUTE:
        raise ValueError(f"stage22 currently supports seq_len <= {MAX_SEQ_LEN_FOR_STAGE22_CUTE}, got {seq_len}.")

    normalized = _normalize_stage22_config(config)
    if normalized.block_m != 128 or normalized.block_n != 128 or head_dim not in {32, 64, 128}:
        raise ValueError("stage22 only supports block_m=128, block_n=128, and head_dim in {32, 64, 128}.")

    q_bshd = q.permute(0, 2, 1, 3).contiguous()
    k_bshd = k.permute(0, 2, 1, 3).contiguous()
    v_bshd = v.permute(0, 2, 1, 3).contiguous()
    o_bshd = torch.empty_like(q_bshd)

    scale_softmax = normalized.resolve_scale(head_dim)
    scale_softmax_log2 = scale_softmax * _LOG2_E
    scale_output = 1.0
    problem_size = (batch, seq_len, seq_len, seq_len, heads, heads, head_dim)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    cache_key = _stage22_cache_key(q, normalized)
    compiled = _STAGE22_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        fmha = BlackwellFusedMultiHeadAttentionForward(
            cutlass.Float32,
            cutlass.Float32,
            (normalized.block_m, normalized.block_n, head_dim),
            False,
            fmha_utils.MaskEnum.WINDOW_MASK,
        )
        compiled = cute.compile(
            fmha,
            from_dlpack(q_bshd, assumed_align=16).iterator,
            from_dlpack(k_bshd, assumed_align=16).iterator,
            from_dlpack(v_bshd, assumed_align=16).iterator,
            from_dlpack(o_bshd, assumed_align=16).iterator,
            problem_size,
            None,
            None,
            None,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            None,
            Int32(0),
            current_stream,
        )
        _STAGE22_COMPILED_CACHE[cache_key] = compiled

    compiled(
        from_dlpack(q_bshd, assumed_align=16).iterator,
        from_dlpack(k_bshd, assumed_align=16).iterator,
        from_dlpack(v_bshd, assumed_align=16).iterator,
        from_dlpack(o_bshd, assumed_align=16).iterator,
        problem_size,
        None,
        None,
        None,
        scale_softmax_log2,
        scale_softmax,
        scale_output,
        None,
        Int32(0),
        current_stream,
    )
    return o_bshd.permute(0, 2, 1, 3).contiguous()


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=128, block_n=128, num_threads=512)
    tuned = _normalize_stage22_config(config)
    if config.autotune:
        tuned = autotune_stage22_config(q, k, v, tuned)
    return _stage22_forward_impl(q, k, v, tuned)
