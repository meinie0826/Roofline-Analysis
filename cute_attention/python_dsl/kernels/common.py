from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
    from cutlass._mlir import ir
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTE = True
except ImportError:  # pragma: no cover - depends on local env
    cutlass = None  # type: ignore[assignment]
    cute = None  # type: ignore[assignment]
    ir = None  # type: ignore[assignment]
    from_dlpack = None  # type: ignore[assignment]
    HAS_CUTE = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLASH_ATTENTION_ROOT = PROJECT_ROOT / "flash-attention"


@dataclass(frozen=True)
class AttentionConfig:
    softmax_scale: float | None = None
    causal: bool = True
    block_m: int = 1
    block_n: int = 128
    num_threads: int = 128
    producer_warps: int = 0
    num_stages_kv: int = 0
    autotune: bool = False

    def resolve_scale(self, head_dim: int) -> float:
        return self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(head_dim)


def require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required but is not installed in the current environment.")


def ensure_cute_ir_context() -> None:
    if not HAS_CUTE:
        return
    if ir.Context.current is None:
        ctx = ir.Context()
        ctx.__enter__()


def validate_qkv(q: Any, k: Any, v: Any) -> None:
    require_torch()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to have shape (batch, heads, seqlen, headdim).")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes for this simplified causal path.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("This project targets CUDA tensors only.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, v must be contiguous.")


def available_backends() -> dict[str, bool]:
    return {
        "torch": HAS_TORCH,
        "cute": HAS_CUTE,
        "fa4_repo": FLASH_ATTENTION_ROOT.exists(),
    }
