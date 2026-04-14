#!/usr/bin/env python3
"""
Stage 0: Naive Attention Kernel
每个 CTA 处理一行 query (baseline)
"""

import torch


def attention_forward(Q, K, V, scale=None):
    """
    Naive attention forward
    
    Args:
        Q, K, V: (B, H, N, d) tensors
        scale: softmax scale (default: 1/sqrt(d))
    
    Returns:
        O: (B, H, N, d) output tensor
    """
    B, H, N, d = Q.shape
    
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    
    # Compute attention scores: Q @ K^T
    # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax
    weights = torch.softmax(scores, dim=-1)
    
    # Compute output: weights @ V
    # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
    output = torch.matmul(weights, V)
    
    return output


class Stage0Attention(torch.autograd.Function):
    """Autograd wrapper for naive attention"""
    
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        O = attention_forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, O, torch.tensor(scale if scale else 1.0 / (Q.shape[-1] ** 0.5)))
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implemented for Stage 0
        raise NotImplementedError("Backward not implemented for Stage 0")


def stage0_attention(Q, K, V, scale=None):
    """Stage 0 attention interface"""
    return Stage0Attention.apply(Q, K, V, scale)


# ============================================================
# Performance metrics
# ============================================================

def compute_tflops(Q, time_ms):
    """Compute achieved TFLOPs"""
    B, H, N, d = Q.shape
    
    # Attention FLOPs: 2 * B * H * N * N * d (Q@K^T + scores@V)
    # For simplicity, assuming non-causal (multiply by 0.5 for causal)
    flops = 2 * B * H * N * N * d
    
    tflops = flops / time_ms / 1e9
    return tflops


def compute_tc_utilization(tflops, peak_tflops=2250):
    """Compute TC utilization percentage"""
    return tflops / peak_tflops * 100
