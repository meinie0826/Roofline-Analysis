from __future__ import annotations

import sys

from .common import AttentionConfig, FLASH_ATTENTION_ROOT, require_torch, validate_qkv


def _import_fa4():
    if str(FLASH_ATTENTION_ROOT) not in sys.path:
        sys.path.insert(0, str(FLASH_ATTENTION_ROOT))
    from flash_attn.cute.interface import flash_attn_func

    return flash_attn_func


def baseline_fa4_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("baseline_fa4 only supports causal attention in this project.")

    flash_attn_func = _import_fa4()
    scale = config.resolve_scale(q.shape[-1])
    return flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
