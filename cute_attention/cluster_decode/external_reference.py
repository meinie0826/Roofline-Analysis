"""Optional SGLang integration probes for reference and benchmark work.

The core megakernel tests deliberately do not require SGLang.  Its full Llama
attention path depends on runtime metadata, paged KV cache managers, backend
selection, and optional compiled kernels.  This module provides lightweight
gates so SGLang-reference tests can fail closed with a useful reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace

from .common import MegakernelConfig, require_torch


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


def _sglang_rms_norm(hidden_states, rms_weight, eps):
    """Run SGLang's RMSNorm module for the pre-attention norm boundary."""
    layernorm = importlib.import_module("sglang.srt.layers.layernorm")
    RMSNorm = layernorm.RMSNorm
    norm = RMSNorm(
        hidden_states.shape[-1],
        eps=eps,
        weight_dtype=rms_weight.dtype,
    ).to(device=hidden_states.device)
    norm.weight.data.copy_(rms_weight)
    return norm(hidden_states)


class _DenseReqToTokenPool:
    """Minimal request-to-token map needed by SGLang's torch-native backend."""

    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class _DenseTokenToKVPool:
    """Dense KV pool adapter for SGLang RadixAttention decode forward."""

    def __init__(self, k_cache, v_cache, k_new, v_new):
        import torch

        self.k_buffer = torch.cat([k_cache, torch.zeros_like(k_new)], dim=0).contiguous()
        self.v_buffer = torch.cat([v_cache, torch.zeros_like(v_new)], dim=0).contiguous()

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        self.k_buffer[loc] = cache_k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
        self.v_buffer[loc] = cache_v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer


def _sglang_radix_decode_forward(q, k_new, v_new, k_cache, v_cache, config):
    """Run SGLang RadixAttention.forward on a dense single-token decode batch."""
    import torch

    forward_batch_info = importlib.import_module(
        "sglang.srt.model_executor.forward_batch_info"
    )
    radix_attention = importlib.import_module("sglang.srt.layers.radix_attention")
    torch_native_backend = importlib.import_module(
        "sglang.srt.layers.attention.torch_native_backend"
    )

    ForwardBatch = forward_batch_info.ForwardBatch
    ForwardMode = forward_batch_info.ForwardMode
    RadixAttention = radix_attention.RadixAttention
    TorchNativeAttnBackend = torch_native_backend.TorchNativeAttnBackend

    seq_len = k_cache.shape[0]
    total_len = seq_len + 1
    device = q.device

    layer = RadixAttention(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        scaling=config.resolve_scale(),
        num_kv_heads=config.num_heads,
        layer_id=0,
    )
    backend = TorchNativeAttnBackend(SimpleNamespace(device=device))

    req_to_token = torch.arange(total_len, device=device, dtype=torch.int64).unsqueeze(0)
    out_cache_loc = torch.tensor([seq_len], device=device, dtype=torch.int64)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int64)

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.zeros(1, device=device, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, device=device, dtype=torch.int64),
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_len,
        seq_lens_cpu=seq_lens.cpu(),
        req_to_token_pool=_DenseReqToTokenPool(req_to_token),
        token_to_kv_pool=_DenseTokenToKVPool(k_cache, v_cache, k_new, v_new),
        attn_backend=backend,
        positions=torch.tensor([seq_len], device=device, dtype=torch.int64),
        num_token_non_padded_cpu=1,
    )

    out = layer.forward(
        q.reshape(1, config.hidden_dim),
        k_new.reshape(1, config.hidden_dim),
        v_new.reshape(1, config.hidden_dim),
        forward_batch,
        save_kv_cache=True,
    )
    return out.reshape(config.num_heads, config.head_dim)


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
    """Reference forward using SGLang RoPE and RadixAttention for dense decode.

    This instantiates SGLang RMSNorm, GPT-J RoPE, and
    `sglang.srt.layers.radix_attention.RadixAttention` with a minimal dense
    `ForwardBatch` adapter. Dense QKV/W_o are evaluated with the same packed
    tensor layout that SGLang's Llama attention uses after weight loading:
    `[all Q; all K; all V]` rows for QKV and row-major output projection.
    """
    require_torch()
    config = config or MegakernelConfig()
    validate_supported_external_config(config, rope_style="gptj")

    import torch

    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    head_dim = config.head_dim

    import torch.nn.functional as F

    h_norm = _sglang_rms_norm(hidden_states, rms_weight, eps=eps)
    qkv = F.linear(h_norm, w_qkv)
    q = qkv[:, :hidden_dim].reshape(1, num_heads, head_dim)
    k = qkv[:, hidden_dim : 2 * hidden_dim].reshape(1, num_heads, head_dim)
    v = qkv[:, 2 * hidden_dim :].reshape(1, num_heads, head_dim)

    q_rot, k_rot = _apply_sglang_gptj_rope(q, k, cos_rope, sin_rope)

    k_new = k_rot.to(hidden_states.dtype)
    v_new = v.to(hidden_states.dtype)

    attn_out = _sglang_radix_decode_forward(
        q_rot,
        k_new,
        v_new,
        k_cache,
        v_cache,
        config,
    )

    output = F.linear(attn_out.reshape(1, hidden_dim), w_o).to(hidden_states.dtype)

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
