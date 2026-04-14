from __future__ import annotations

from .common import AttentionConfig, HAS_CUTE
from .stage0_naive import stage0_forward


def attention_forward(q, k, v, scale=None):
    return stage0_forward(q, k, v, AttentionConfig(softmax_scale=scale))


__all__ = ["HAS_CUTE", "attention_forward", "stage0_forward"]
