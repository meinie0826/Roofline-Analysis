from __future__ import annotations

from .stage4_mma import stage4_forward


def stage5_forward(q, k, v, config=None):
    """
    Stage5 (cp.async/pipeline path) scaffold.
    Current behavior intentionally reuses stage4 until pipeline kernel lands.
    """
    return stage4_forward(q, k, v, config)
