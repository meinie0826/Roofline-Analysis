"""Optional SGLang integration probes for reference and benchmark work.

The core megakernel tests deliberately do not require SGLang.  Its full Llama
attention path depends on runtime metadata, paged KV cache managers, backend
selection, and optional compiled kernels.  This module provides lightweight
gates so SGLang-reference tests can fail closed with a useful reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

from .common import MegakernelConfig, require_torch
from .megakernel_reference import rms_norm


@dataclass(frozen=True)
class FrameworkStatus:
    """Import status for an optional external framework."""

    name: str
    available: bool
    version: str | None = None
    error: str | None = None


def probe_framework_import(name: str) -> FrameworkStatus:
    """Try importing an optional framework without making it a hard dependency."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - depends on local env
        return FrameworkStatus(name=name, available=False, error=f"{type(exc).__name__}: {exc}")

    return FrameworkStatus(
        name=name,
        available=True,
        version=getattr(module, "__version__", None),
    )


def probe_sglang_import() -> FrameworkStatus:
    """Try importing SGLang without making it a hard dependency."""
    return probe_framework_import("sglang")


def validate_supported_external_config(
    config: MegakernelConfig,
    *,
    batch_size: int = 1,
    q_len: int = 1,
    num_kv_heads: int | None = None,
    rope_style: str = "gptj",
    paged_kv: bool = False,
    sliding_window: bool = False,
    attention_sinks: bool = False,
    quantized_kv: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    """Reject SGLang branches that the current megakernel does not model.

    This is intended for optional external reference/benchmark tests.  It keeps
    comparisons honest by failing before a framework path silently exercises a
    feature outside the dense single-token MHA kernel we currently implement.
    """
    config.validate()
    kv_heads = config.num_heads if num_kv_heads is None else num_kv_heads

    unsupported: list[str] = []
    if batch_size != 1:
        unsupported.append("batch_size != 1")
    if q_len != 1:
        unsupported.append("q_len != 1")
    if kv_heads != config.num_heads:
        unsupported.append("GQA/MQA num_kv_heads != num_heads")
    if rope_style.lower() != "gptj":
        unsupported.append("non-GPT-J RoPE")
    if paged_kv:
        unsupported.append("paged KV cache")
    if sliding_window:
        unsupported.append("sliding-window attention")
    if attention_sinks:
        unsupported.append("attention sinks")
    if quantized_kv:
        unsupported.append("quantized KV cache")
    if tensor_parallel_size != 1:
        unsupported.append("tensor parallel size != 1")

    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"Unsupported external-reference config: {joined}.")


def _rope_half_tables(cos_rope, sin_rope):
    """Convert full interleaved GPT-J cos/sin tables to framework shape."""
    return cos_rope.to(cos_rope.device, dtype=cos_rope.dtype)[0::2].unsqueeze(0), sin_rope.to(
        sin_rope.device, dtype=sin_rope.dtype
    )[0::2].unsqueeze(0)


def _apply_sglang_gptj_rope(q, k, cos_rope, sin_rope):
    """Apply GPT-J/interleaved RoPE using SGLang's implementation."""
    cos_half, sin_half = _rope_half_tables(cos_rope, sin_rope)
    rotary = importlib.import_module("sglang.srt.layers.rotary_embedding.utils")
    apply_rotary_emb = rotary.apply_rotary_emb
    return (
        apply_rotary_emb(q, cos_half, sin_half, is_neox_style=False),
        apply_rotary_emb(k, cos_half, sin_half, is_neox_style=False),
    )


def sglang_megakernel_reference_forward(
    hidden_states,
    w_qkv,
    w_o,
    k_cache,
    v_cache,
    rms_weight,
    cos_rope,
    sin_rope,
    config: MegakernelConfig | None = None,
    eps: float = 1e-6,
):
    """Reference forward using SGLang RoPE for the supported dense path.

    This intentionally does not instantiate SGLang's full attention layer yet:
    that path requires paged-cache/runtime metadata. Instead, SGLang owns the
    RoPE semantics, while the dense single-token MHA decode math mirrors
    `megakernel_reference_forward`.
    """
    require_torch()
    config = config or MegakernelConfig()
    validate_supported_external_config(config, rope_style="gptj")

    import torch

    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    head_dim = config.head_dim
    scale = config.resolve_scale()

    h = hidden_states.to(torch.float32)
    w_qkv_f = w_qkv.to(torch.float32)
    w_o_f = w_o.to(torch.float32)

    h_norm = rms_norm(h, rms_weight, eps=eps)
    qkv = h_norm @ w_qkv_f.T
    q = qkv[:, :hidden_dim].reshape(1, num_heads, head_dim)
    k = qkv[:, hidden_dim : 2 * hidden_dim].reshape(1, num_heads, head_dim)
    v = qkv[:, 2 * hidden_dim :].reshape(1, num_heads, head_dim)

    q_rot, k_rot = _apply_sglang_gptj_rope(q, k, cos_rope, sin_rope)

    k_new = k_rot.to(hidden_states.dtype)
    v_new = v.to(hidden_states.dtype)

    k_f = torch.cat([k_cache.to(torch.float32), k_new.to(torch.float32)], dim=0)
    v_f = torch.cat([v_cache.to(torch.float32), v_new.to(torch.float32)], dim=0)
    scores = torch.bmm(q_rot[0].unsqueeze(1), k_f.permute(1, 2, 0)) * scale
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.bmm(probs, v_f.permute(1, 0, 2)).squeeze(1)

    scratch_wo = torch.empty(
        (num_heads, hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    for h_idx in range(num_heads):
        cols = slice(h_idx * head_dim, (h_idx + 1) * head_dim)
        scratch_wo[h_idx] = torch.sum(
            attn_out[h_idx].unsqueeze(0) * w_o_f[:, cols],
            dim=1,
        )
    output = scratch_wo.sum(dim=0).unsqueeze(0).to(hidden_states.dtype)

    return output, k_new, v_new


def external_reference_status(config: MegakernelConfig | None = None) -> str:
    """Human-readable status for CLI/debug use."""
    config = config or MegakernelConfig()
    try:
        validate_supported_external_config(config)
        config_status = "config supported"
    except ValueError as exc:
        config_status = str(exc)

    parts = [config_status]
    status = probe_sglang_import()
    if status.available:
        suffix = f" {status.version}" if status.version else ""
        parts.append(f"sglang: available{suffix}")
    else:
        parts.append(f"sglang: unavailable ({status.error})")
    return "\n".join(parts)
