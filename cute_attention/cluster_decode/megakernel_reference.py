"""Pure-PyTorch reference implementation of the ClusterFusion decode megakernel.

Implements the same pipeline as cluster_megakernel.py but in eager PyTorch so
that correctness of the CuTeDSL kernel can be verified:

  RMSNorm → W_qkv projection → RoPE → current-token K/V output
  → Flash-decode attention over provided KV cache plus current token
  → W_o projection

This is a single-token (q_len=1) decode step. The caller provides the previous
KV cache; the current-token K/V are produced here and included in attention.

Usage::

    from cluster_decode.megakernel_reference import megakernel_reference_forward
    output, k_new, v_new = megakernel_reference_forward(
        hidden_states, w_qkv, w_o, k_cache, v_cache, rms_weight, cos_rope, sin_rope, config
    )
"""

from __future__ import annotations

import math
import torch

from .common import MegakernelConfig, require_torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x_norm = x / rms(x) * weight.

    Args:
        x:      (..., hidden_dim) – input in any dtype.
        weight: (hidden_dim,) fp16 or fp32 learnable scale.
    """
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(torch.float32)).to(x.dtype)


def apply_rope_gptj(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT-J style RoPE (interleaved pairs).

    Args:
        q, k:    (head_dim,) fp32 – single-head vectors.
        cos, sin: (head_dim,) fp32.

    Returns: rotated q, k with same shape.
    """
    def rotate(v):
        v0 = v[0::2]   # even indices
        v1 = v[1::2]   # odd indices
        rotated = torch.empty_like(v)
        rotated[0::2] = v0 * cos[0::2] - v1 * sin[0::2]
        rotated[1::2] = v1 * cos[1::2] + v0 * sin[1::2]
        return rotated

    return rotate(q), rotate(k)


def megakernel_reference_forward(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    rms_weight: torch.Tensor,
    cos_rope: torch.Tensor,
    sin_rope: torch.Tensor,
    config: MegakernelConfig | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference decode megakernel.

    Args:
        hidden_states: (1, hidden_dim) fp16.
        w_qkv:         (3*hidden_dim, hidden_dim) fp16 – [W_q; W_k; W_v] row-major.
        w_o:           (hidden_dim, hidden_dim) fp16.
        k_cache:       (seq_len, num_heads, head_dim) fp16.
        v_cache:       (seq_len, num_heads, head_dim) fp16.
        rms_weight:    (hidden_dim,) fp16.
        cos_rope:      (head_dim,) fp32.
        sin_rope:      (head_dim,) fp32.
        config:        MegakernelConfig (defaults to Llama-2-7B).
        eps:           RMSNorm epsilon.

    Returns:
        output:  (1, hidden_dim) fp16 – attn @ W_o.
        k_new:   (1, num_heads, head_dim) fp16 – current-token K after RoPE.
        v_new:   (1, num_heads, head_dim) fp16 – current-token V (no RoPE).
    """
    require_torch()
    config = config or MegakernelConfig()
    config.validate()

    hidden_dim  = config.hidden_dim
    num_heads   = config.num_heads
    head_dim    = config.head_dim
    scale       = config.resolve_scale()

    # Work in float32 internally
    h = hidden_states.to(torch.float32)                 # (1, D)
    w_qkv_f = w_qkv.to(torch.float32)                  # (3D, D)
    w_o_f   = w_o.to(torch.float32)                    # (D, D)

    # ------------------------------------------------------------------ #
    # Stage 0 – RMSNorm                                                   #
    # ------------------------------------------------------------------ #
    h_norm = rms_norm(h, rms_weight, eps=eps)           # (1, D) fp32

    # ------------------------------------------------------------------ #
    # Stage 1 – W_qkv projection                                          #
    # ------------------------------------------------------------------ #
    # w_qkv layout: [W_q (D×D); W_k (D×D); W_v (D×D)] stacked on dim-0
    qkv = h_norm @ w_qkv_f.T                            # (1, 3D)
    Q = qkv[:, :hidden_dim]                             # (1, D)
    K = qkv[:, hidden_dim:2*hidden_dim]                 # (1, D)
    V = qkv[:, 2*hidden_dim:]                           # (1, D)

    # Reshape to per-head: (1, num_heads, head_dim)
    Q = Q.reshape(1, num_heads, head_dim)
    K = K.reshape(1, num_heads, head_dim)
    V = V.reshape(1, num_heads, head_dim)

    # ------------------------------------------------------------------ #
    # Stage 2 – RoPE                                                      #
    # ------------------------------------------------------------------ #
    cos = cos_rope.to(torch.float32)
    sin = sin_rope.to(torch.float32)

    Q_rot = torch.empty_like(Q)
    K_rot = torch.empty_like(K)
    for h_idx in range(num_heads):
        q_h = Q[0, h_idx]   # (head_dim,)
        k_h = K[0, h_idx]
        q_r, k_r = apply_rope_gptj(q_h, k_h, cos, sin)
        Q_rot[0, h_idx] = q_r
        K_rot[0, h_idx] = k_r

    # Current-token outputs
    k_new = K_rot.to(hidden_states.dtype)               # (1, num_heads, head_dim)
    v_new = V.to(hidden_states.dtype)                   # (1, num_heads, head_dim)

    # ------------------------------------------------------------------ #
    # Stage 3 – Flash-decode attention over previous cache + current KV   #
    # ------------------------------------------------------------------ #
    # The fused kernel exposes current K/V as fp16/bf16 outputs before the
    # caller stores them in cache. Use the same values for attention so the
    # reference models the framework-visible decode state.
    k_f = torch.cat([k_cache.to(torch.float32), k_new.to(torch.float32)], dim=0)
    v_f = torch.cat([v_cache.to(torch.float32), v_new.to(torch.float32)], dim=0)

    # scores: (1, num_heads, seq_len)
    # Q_rot: (1, num_heads, head_dim) → (num_heads, 1, head_dim)
    # k_f:   (seq_len, num_heads, head_dim) → (num_heads, head_dim, seq_len)
    Q_bh  = Q_rot[0].unsqueeze(1)                      # (num_heads, 1, head_dim)
    K_bh  = k_f.permute(1, 2, 0)                       # (num_heads, head_dim, seq_len)
    V_bh  = v_f.permute(1, 0, 2)                       # (num_heads, seq_len, head_dim)

    scores = torch.bmm(Q_bh, K_bh) * scale             # (num_heads, 1, seq_len)
    probs  = torch.softmax(scores, dim=-1)              # (num_heads, 1, seq_len)
    attn_out = torch.bmm(probs, V_bh)                  # (num_heads, 1, head_dim)
    attn_out = attn_out.squeeze(1)                      # (num_heads, head_dim)
    attn_vec = attn_out.reshape(1, hidden_dim)          # (1, D)

    # ------------------------------------------------------------------ #
    # Stage 5 – W_o projection                                            #
    # ------------------------------------------------------------------ #
    output = (attn_vec @ w_o_f.T).to(hidden_states.dtype)   # (1, D)

    return output, k_new, v_new


def make_random_megakernel_inputs(
    config: MegakernelConfig,
    seq_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> dict:
    """Construct random inputs for the megakernel (reference and CUDA kernel).

    Returns a dict with all required tensors already on `device`.
    """
    torch.manual_seed(seed)
    D  = config.hidden_dim
    NH = config.num_heads
    HD = config.head_dim

    return {
        "hidden_states": torch.randn(1,  D,        device=device, dtype=dtype),
        "w_qkv":         torch.randn(3*D, D,       device=device, dtype=dtype),
        "w_o":           torch.randn(D,   D,       device=device, dtype=dtype),
        "k_cache":       torch.randn(seq_len, NH, HD, device=device, dtype=dtype),
        "v_cache":       torch.randn(seq_len, NH, HD, device=device, dtype=dtype),
        "rms_weight":    torch.ones(D,            device=device, dtype=dtype),
        "cos_rope":      torch.ones(HD,           device=device, dtype=torch.float32),
        "sin_rope":      torch.zeros(HD,          device=device, dtype=torch.float32),
    }
