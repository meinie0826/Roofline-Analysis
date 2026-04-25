from __future__ import annotations

from .common import require_torch, torch


def leader_reduce_payload_floats(head_dim: int) -> int:
    """Per CTA payload for leader-only split-KV decode reduction.

    Each CTA contributes one local max, one local softmax denominator, and
    one unnormalized output vector. This is the fine-grained primitive we want
    before considering ClusterFusion's broader all-reduce style primitive.
    """
    return head_dim + 2


def split_kv_decode_partials(q, k, v, cluster_size: int, softmax_scale: float):
    """Build split-KV softmax partials that mirror the planned DSM contract.

    Shapes are flattened over batch and head:
      q: (batch_heads, 1, head_dim)
      k: (batch_heads, seq_len, head_dim)
      v: (batch_heads, seq_len, head_dim)

    Returns:
      partial_max: (batch_heads, cluster_size)
      partial_sum: (batch_heads, cluster_size)
      partial_out: (batch_heads, cluster_size, head_dim)

    The kernel-side contract is that CTA rank r owns the same KV slice as the
    r-th column here, writes these three values to DSM, and only rank 0 reads
    the cluster payload and writes the final output row.
    """
    require_torch()
    _validate_split_inputs(q, k, v, cluster_size)

    batch_heads, seq_len, head_dim = k.shape
    partial_max = torch.full(
        (batch_heads, cluster_size),
        -torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    partial_sum = torch.zeros((batch_heads, cluster_size), device=q.device, dtype=torch.float32)
    partial_out = torch.zeros((batch_heads, cluster_size, head_dim), device=q.device, dtype=torch.float32)

    q_f = q[:, 0, :].to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    kv_per_cta = (seq_len + cluster_size - 1) // cluster_size

    for cta_rank in range(cluster_size):
        kv_start = cta_rank * kv_per_cta
        kv_stop = min(kv_start + kv_per_cta, seq_len)
        if kv_start >= kv_stop:
            continue

        scores = torch.sum(q_f[:, None, :] * k_f[:, kv_start:kv_stop, :], dim=-1) * softmax_scale
        slice_max = torch.max(scores, dim=-1).values
        exp_scores = torch.exp(scores - slice_max[:, None])
        partial_max[:, cta_rank] = slice_max
        partial_sum[:, cta_rank] = torch.sum(exp_scores, dim=-1)
        partial_out[:, cta_rank, :] = torch.sum(exp_scores[:, :, None] * v_f[:, kv_start:kv_stop, :], dim=1)

    return partial_max, partial_sum, partial_out


def merge_split_kv_decode_partials(partial_max, partial_sum, partial_out):
    """Merge split-KV partials into final decode attention output.

    This is the numerically stable cross-CTA combine:

      M = max_i(m_i)
      L = sum_i(l_i * exp(m_i - M))
      O = sum_i(O_i * exp(m_i - M)) / L
    """
    require_torch()
    if partial_max.ndim != 2 or partial_sum.shape != partial_max.shape:
        raise ValueError("partial_max and partial_sum must have shape (batch_heads, cluster_size).")
    if partial_out.ndim != 3 or partial_out.shape[:2] != partial_max.shape:
        raise ValueError("partial_out must have shape (batch_heads, cluster_size, head_dim).")

    global_max = torch.max(partial_max, dim=1).values
    safe_delta = torch.where(
        torch.isfinite(partial_max),
        partial_max - global_max[:, None],
        torch.full_like(partial_max, -torch.inf),
    )
    renorm = torch.exp(safe_delta)
    global_sum = torch.sum(partial_sum * renorm, dim=1)
    numerator = torch.sum(partial_out * renorm[:, :, None], dim=1)
    return numerator / global_sum[:, None]


def split_kv_decode_reference(q, k, v, cluster_size: int, softmax_scale: float):
    """Reference output for the planned fine-grained cluster reduction."""
    partials = split_kv_decode_partials(q, k, v, cluster_size, softmax_scale)
    return merge_split_kv_decode_partials(*partials).to(dtype=q.dtype).unsqueeze(1)


def _validate_split_inputs(q, k, v, cluster_size: int) -> None:
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive.")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected flattened q/k/v tensors with rank 3.")
    if q.shape[1] != 1:
        raise ValueError("split-KV decode reduction expects q_len=1.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, v must have the same batch_heads dimension.")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical flattened shapes.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k/v must have the same head_dim.")
    if k.shape[1] == 0:
        raise ValueError("seq_len must be nonzero.")
