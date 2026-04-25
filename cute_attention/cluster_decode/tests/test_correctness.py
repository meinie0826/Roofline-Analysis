#!/usr/bin/env python3
from __future__ import annotations

import pytest

from cluster_decode import (
    ClusterDecodeConfig,
    available_backends,
    cluster_decode_forward,
    cluster_decode_split_forward,
    leader_reduce_payload_floats,
    split_kv_decode_reference,
)


backends = available_backends()
torch = pytest.importorskip("torch") if backends["torch"] else None


def make_decode_inputs(batch, heads, seqlen_k, head_dim, dtype):
    q = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def decode_reference(q, k, v, scale):
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.to(torch.float32)).to(dtype=q.dtype)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.parametrize("cluster_size", [2, 4])
def test_cluster_decode_fine_grained_reduce_contract_matches_reference(cluster_size):
    batch_heads, seq_len, head_dim = 3, 129, 128
    q = torch.randn(batch_heads, 1, head_dim, dtype=torch.float32)
    k = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    config = ClusterDecodeConfig(cluster_size=cluster_size)

    ref = decode_reference(q, k, v, config.resolve_scale(head_dim))
    out = split_kv_decode_reference(q, k, v, cluster_size, config.resolve_scale(head_dim))

    assert leader_reduce_payload_floats(head_dim) == 130
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_cluster_decode_v0_matches_decode_reference():
    q, k, v = make_decode_inputs(batch=1, heads=2, seqlen_k=128, head_dim=128, dtype=torch.float16)
    config = ClusterDecodeConfig(num_threads=128, cluster_size=2)
    ref = decode_reference(q, k, v, config.resolve_scale(q.shape[-1]))
    out = cluster_decode_forward(q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
@pytest.mark.parametrize("cluster_size", [2, 4])
def test_cluster_decode_split_skeleton_matches_decode_reference(cluster_size):
    q, k, v = make_decode_inputs(batch=1, heads=2, seqlen_k=128, head_dim=128, dtype=torch.float16)
    config = ClusterDecodeConfig(num_threads=128, cluster_size=cluster_size)
    ref = decode_reference(q, k, v, config.resolve_scale(q.shape[-1]))
    out = cluster_decode_split_forward(q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
