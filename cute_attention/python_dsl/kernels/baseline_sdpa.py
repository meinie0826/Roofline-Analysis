from __future__ import annotations

from .common import AttentionConfig, require_torch, validate_qkv


def baseline_sdpa_forward(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    from torch.nn import functional as F

    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("baseline_sdpa only supports causal attention in this project.")

    scale = config.resolve_scale(q.shape[-1])
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    )
