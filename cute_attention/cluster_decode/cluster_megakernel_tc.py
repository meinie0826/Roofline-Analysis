"""Experimental tensor-core decode layer path.

This path keeps the same public contract as ``cluster_megakernel_forward`` but
uses framework tensor-core matmuls for QKV and WO.  It is intentionally separate
from the CuTeDSL cluster kernel: use it as a fast target line while the TC/UMMA
projection tiles are being ported into the fused kernel.
"""

from __future__ import annotations

from .common import MegakernelConfig, require_torch, validate_megakernel_inputs
from .megakernel_reference import rms_norm


def _apply_rope_gptj_batched(q, k, cos, sin):
    q0 = q[..., 0::2]
    q1 = q[..., 1::2]
    k0 = k[..., 0::2]
    k1 = k[..., 1::2]
    c0 = cos[0::2]
    c1 = cos[1::2]
    s0 = sin[0::2]
    s1 = sin[1::2]

    q_rot = q.new_empty(q.shape)
    k_rot = k.new_empty(k.shape)
    q_rot[..., 0::2] = q0 * c0 - q1 * s0
    q_rot[..., 1::2] = q1 * c1 + q0 * s1
    k_rot[..., 0::2] = k0 * c0 - k1 * s0
    k_rot[..., 1::2] = k1 * c1 + k0 * s1
    return q_rot, k_rot


def cluster_megakernel_tc_forward(
    hidden_states,
    w_qkv,
    w_o,
    k_cache,
    v_cache,
    rms_weight,
    cos_rope,
    sin_rope,
    config: MegakernelConfig | None = None,
):
    """Run an experimental tensor-core version of the decode layer.

    QKV and WO are expressed as dense matmuls so CUDA routes them to tensor
    cores for fp16/bf16.  Attention is a dense PyTorch decode reference.
    """
    require_torch()
    config = config or MegakernelConfig()
    validate_megakernel_inputs(hidden_states, w_qkv, w_o, k_cache, v_cache, rms_weight, config)

    import torch

    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    head_dim = config.head_dim
    scale = config.resolve_scale()
    dtype = hidden_states.dtype

    # Match the activation-dtype boundary used by SGLang/HF Llama RMSNorm.
    h_norm = rms_norm(hidden_states.float(), rms_weight, eps=1e-6).to(dtype)

    # Tensor-core QKV projection: (1, D) x (D, 3D) -> (1, 3D)
    qkv = h_norm @ w_qkv.t()
    qkv = qkv.reshape(1, 3, num_heads, head_dim)
    q = qkv[:, 0]
    k = qkv[:, 1]
    v = qkv[:, 2]

    cos = cos_rope.to(dtype)
    sin = sin_rope.to(dtype)
    q_rot, k_rot = _apply_rope_gptj_batched(q, k, cos, sin)

    k_new = k_rot.to(dtype)
    v_new = v.to(dtype)

    k_f = torch.cat([k_cache.to(torch.float32), k_new.to(torch.float32)], dim=0)
    v_f = torch.cat([v_cache.to(torch.float32), v_new.to(torch.float32)], dim=0)
    q_bh = q_rot[0].to(torch.float32).unsqueeze(1)
    k_bh = k_f.permute(1, 2, 0)
    v_bh = v_f.permute(1, 0, 2)

    scores = torch.bmm(q_bh, k_bh) * scale
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.bmm(probs, v_bh).squeeze(1).to(dtype)

    # Tensor-core WO projection: (1, D) x (D, D) -> (1, D)
    output = attn_out.reshape(1, hidden_dim) @ w_o.t()
    return output.to(dtype), k_new, v_new
