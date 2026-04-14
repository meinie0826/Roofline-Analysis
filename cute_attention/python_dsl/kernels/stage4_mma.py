from __future__ import annotations

from .stage1_fa2 import stage1_forward


def stage4_forward(q, k, v, config=None):
    """
    Stage4 (MMA path) scaffold.
    Current behavior intentionally reuses stage1 kernel until tiled_mma path lands.
    """
    return stage1_forward(q, k, v, config)
