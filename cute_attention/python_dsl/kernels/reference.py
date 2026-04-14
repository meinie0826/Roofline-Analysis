from __future__ import annotations

from .common import torch
from .common import AttentionConfig, require_torch, validate_qkv


def causal_attention_reference(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("This project only supports causal attention.")

    scale = config.resolve_scale(q.shape[-1])
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale

    seq_q = q.shape[-2]
    seq_k = k.shape[-2]
    causal_mask = q.new_ones((seq_q, seq_k), dtype=q.dtype).triu(1).bool()
    scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = scores.softmax(dim=-1)
    return probs @ v


def causal_attention_online_reference(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("This project only supports causal attention.")

    scale = config.resolve_scale(q.shape[-1])
    out = q.new_empty(q.shape)

    batch, heads, seqlen, _ = q.shape
    for b in range(batch):
        for h in range(heads):
            for qi in range(seqlen):
                q_row = q[b, h, qi]
                row_max = None
                row_sum = None
                row_out = None

                for kj in range(qi + 1):
                    score = (q_row * k[b, h, kj]).sum() * scale
                    value = v[b, h, kj]
                    if row_max is None:
                        row_max = score
                        row_sum = score.new_tensor(1.0, dtype=torch.float32)
                        row_out = value.to(torch.float32)
                        continue

                    new_max = torch.maximum(row_max, score)
                    old_scale = torch.exp(row_max - new_max)
                    new_scale = torch.exp(score - new_max)
                    row_sum = row_sum * old_scale + new_scale
                    row_out = row_out * old_scale + value.to(torch.float32) * new_scale
                    row_max = new_max

                out[b, h, qi] = (row_out / row_sum).to(dtype=q.dtype)

    return out


def causal_attention_blocked_reference(q, k, v, config: AttentionConfig | None = None):
    require_torch()
    config = config or AttentionConfig()
    validate_qkv(q, k, v)
    if not config.causal:
        raise ValueError("This project only supports causal attention.")

    scale = config.resolve_scale(q.shape[-1])
    out = q.new_empty(q.shape)
    block_n = config.block_n

    batch, heads, seqlen, _ = q.shape
    for b in range(batch):
        for h in range(heads):
            for q_start in range(0, seqlen, config.block_m):
                q_stop = min(q_start + config.block_m, seqlen)
                q_tile = q[b, h, q_start:q_stop].to(torch.float32)
                tile_out = torch.zeros_like(q_tile)
                tile_max = torch.full((q_stop - q_start,), float("-inf"), device=q.device)
                tile_sum = torch.zeros((q_stop - q_start,), device=q.device, dtype=torch.float32)

                max_k = q_stop
                for k_start in range(0, max_k, block_n):
                    k_stop = min(k_start + block_n, max_k)
                    k_tile = k[b, h, k_start:k_stop].to(torch.float32)
                    v_tile = v[b, h, k_start:k_stop].to(torch.float32)
                    scores = q_tile @ k_tile.transpose(-2, -1)
                    scores = scores * scale

                    q_idx = torch.arange(q_start, q_stop, device=q.device)[:, None]
                    k_idx = torch.arange(k_start, k_stop, device=q.device)[None, :]
                    scores = scores.masked_fill(k_idx > q_idx, float("-inf"))

                    block_max = scores.max(dim=-1).values
                    new_max = torch.maximum(tile_max, block_max)
                    old_scale = torch.exp(tile_max - new_max)
                    probs = torch.exp(scores - new_max[:, None])
                    tile_sum = tile_sum * old_scale + probs.sum(dim=-1)
                    tile_out = tile_out * old_scale[:, None] + probs @ v_tile
                    tile_max = new_max

                out[b, h, q_start:q_stop] = (tile_out / tile_sum[:, None]).to(dtype=q.dtype)

    return out
