from .common import AttentionConfig
from .stage22_tma_backend import Stage22FlashAttentionTmaExperimental


def stage22_forward(q, k, v, config: AttentionConfig | None = None):
    raise NotImplementedError(
        "stage22_tma is under construction and is not wired into the benchmark registry yet."
    )


def autotune_stage22_config(q, k, v, config: AttentionConfig | None = None):
    raise NotImplementedError(
        "stage22_tma autotune will be added after the first runnable TMA load path lands."
    )


__all__ = [
    "Stage22FlashAttentionTmaExperimental",
    "stage22_forward",
    "autotune_stage22_config",
]
