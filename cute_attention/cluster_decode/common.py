from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTE = True
except ImportError:  # pragma: no cover - depends on local env
    cutlass = None  # type: ignore[assignment]
    cute = None  # type: ignore[assignment]
    from_dlpack = None  # type: ignore[assignment]
    HAS_CUTE = False


@dataclass(frozen=True)
class ClusterDecodeConfig:
    softmax_scale: float | None = None
    num_threads: int = 128
    cluster_size: int = 2

    def resolve_scale(self, head_dim: int) -> float:
        return self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(head_dim)


def require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required but is not installed in the current environment.")


def available_backends() -> dict[str, bool]:
    return {
        "torch": HAS_TORCH,
        "cute": HAS_CUTE,
    }


def validate_decode_qkv(q: Any, k: Any, v: Any, config: ClusterDecodeConfig) -> None:
    require_torch()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to have shape (batch, heads, seqlen, headdim).")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, v must have the same batch size.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("cluster_decode v0 only supports MHA with matching q/k/v heads.")
    if q.shape[2] != 1:
        raise ValueError("cluster_decode v0 only supports decode q_len=1.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("cluster_decode v0 requires q, k, v to have the same head_dim.")
    if q.shape[-1] != 128:
        raise ValueError("cluster_decode v0 is scoped to head_dim=128.")
    if config.cluster_size not in (2, 4):
        raise ValueError("cluster_decode v0 supports cluster_size 2 or 4.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("cluster_decode targets CUDA tensors only.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, v must be contiguous.")
