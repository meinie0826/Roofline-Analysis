from __future__ import annotations

import math
from dataclasses import dataclass, field
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
class MegakernelConfig:
    """Configuration for the ClusterFusion-style decode megakernel.

    Maps directly to the ClusterFusion design:
      - One cluster per attention head.
      - Each CTA in the cluster owns a DIM_PER_BLOCK = hidden_dim // cluster_size
        slice of the hidden dimension for the W_qkv / W_o GEMMs.
      - Each CTA also owns a KV_DIM_PER_BLOCK = ceil(seq_len / cluster_size) slice
        of the KV cache for flash-decoding.
    """
    # Model dimensions (Llama-2-7B defaults)
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128

    # Cluster / threading
    cluster_size: int = 4       # 2 or 4 CTAs per cluster
    num_threads: int = 128      # BLOCK_SIZE = 4 warps * 32

    # Softmax scale (None → 1/sqrt(head_dim))
    softmax_scale: float | None = None

    # TMA tile granularity along the hidden-dim axis (hidden rows loaded at once)
    tma_load_once: int = 64     # TMA_LOAD_ONCE from ClusterFusion config.h

    def resolve_scale(self) -> float:
        return self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)

    @property
    def dim_per_block(self) -> int:
        """Hidden-dim slice owned by each CTA (DIM_PER_BLOCK)."""
        return self.hidden_dim // self.cluster_size

    @property
    def kv_per_cta(self, seq_len: int = 0) -> int:
        """KV-cache rows owned by each CTA (rounded up, padded to tma_load_once/2)."""
        if seq_len == 0:
            return 0
        raw = (seq_len + self.cluster_size - 1) // self.cluster_size
        tile = self.tma_load_once // 2
        return (raw + tile - 1) & ~(tile - 1)

    def validate(self) -> None:
        if self.hidden_dim % self.cluster_size != 0:
            raise ValueError("hidden_dim must be divisible by cluster_size.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.hidden_dim // self.num_heads != self.head_dim:
            raise ValueError("head_dim must equal hidden_dim // num_heads.")
        if self.cluster_size not in (2, 4):
            raise ValueError("cluster_size must be 2 or 4.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")


@dataclass(frozen=True)
class ClusterDecodeConfig:
    """Lightweight config for the standalone attention-only stages (cluster_decode*.py).

    Kept for backward compatibility with the existing test suite.
    """
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


def validate_megakernel_inputs(
    hidden_states: Any,
    w_qkv: Any,
    w_o: Any,
    k_cache: Any,
    v_cache: Any,
    rms_weight: Any,
    config: MegakernelConfig,
) -> None:
    """Validate inputs for the full ClusterFusion megakernel."""
    require_torch()
    config.validate()

    # hidden_states: (1, hidden_dim)
    if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
        raise ValueError("hidden_states must be (1, hidden_dim) for single-token decode.")
    if hidden_states.shape[1] != config.hidden_dim:
        raise ValueError(f"hidden_states.shape[1]={hidden_states.shape[1]} != hidden_dim={config.hidden_dim}.")

    # w_qkv: (3 * hidden_dim, hidden_dim) – row-major, Q/K/V stacked
    if w_qkv.shape != (3 * config.hidden_dim, config.hidden_dim):
        raise ValueError(
            f"w_qkv must be (3*hidden_dim, hidden_dim)=({3*config.hidden_dim},{config.hidden_dim}), "
            f"got {tuple(w_qkv.shape)}."
        )

    # w_o: (hidden_dim, hidden_dim)
    if w_o.shape != (config.hidden_dim, config.hidden_dim):
        raise ValueError(
            f"w_o must be (hidden_dim, hidden_dim)=({config.hidden_dim},{config.hidden_dim}), "
            f"got {tuple(w_o.shape)}."
        )

    # k_cache / v_cache: (seq_len, num_heads, head_dim)
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("k_cache and v_cache must be (seq_len, num_heads, head_dim).")
    if k_cache.shape[1:] != (config.num_heads, config.head_dim):
        raise ValueError(
            f"k_cache.shape[1:]={tuple(k_cache.shape[1:])} != (num_heads={config.num_heads}, head_dim={config.head_dim})."
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape.")

    # rms_weight: (hidden_dim,)
    if rms_weight.shape != (config.hidden_dim,):
        raise ValueError(f"rms_weight must be (hidden_dim={config.hidden_dim},), got {tuple(rms_weight.shape)}.")

    for name, t in [
        ("hidden_states", hidden_states),
        ("w_qkv", w_qkv),
        ("w_o", w_o),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("rms_weight", rms_weight),
    ]:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
