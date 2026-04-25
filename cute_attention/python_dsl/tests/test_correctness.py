#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

import pytest

from kernels import AttentionConfig, available_backends, run_stage
from kernels.cluster_decode_reduce import (
    leader_reduce_payload_floats,
    split_kv_decode_reference,
)
from kernels.reference import (
    causal_attention_reference,
)


backends = available_backends()
torch = pytest.importorskip("torch") if backends["torch"] else None


def make_inputs(shape, dtype):
    q = torch.randn(*shape, device="cuda", dtype=dtype)
    k = torch.randn(*shape, device="cuda", dtype=dtype)
    v = torch.randn(*shape, device="cuda", dtype=dtype)
    return q, k, v


def make_decode_inputs(batch, heads, seqlen_k, head_dim, dtype):
    q = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_cluster_decode_v0_matches_decode_reference():
    q, k, v = make_decode_inputs(batch=1, heads=2, seqlen_k=128, head_dim=128, dtype=torch.float16)
    config = AttentionConfig(causal=False, block_n=64, num_threads=128, cluster_size=2)
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1))
    probs = torch.softmax(scores * config.resolve_scale(q.shape[-1]), dim=-1)
    ref = torch.matmul(probs, v.to(torch.float32)).to(dtype=q.dtype)
    out = run_stage("cluster_decode", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
@pytest.mark.parametrize("cluster_size", [2, 4])
def test_cluster_decode_split_skeleton_matches_decode_reference(cluster_size):
    q, k, v = make_decode_inputs(batch=1, heads=2, seqlen_k=128, head_dim=128, dtype=torch.float16)
    config = AttentionConfig(causal=False, block_n=64, num_threads=128, cluster_size=cluster_size)
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1))
    probs = torch.softmax(scores * config.resolve_scale(q.shape[-1]), dim=-1)
    ref = torch.matmul(probs, v.to(torch.float32)).to(dtype=q.dtype)
    out = run_stage("cluster_decode_split", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.parametrize("cluster_size", [2, 4])
def test_cluster_decode_fine_grained_reduce_contract_matches_reference(cluster_size):
    batch_heads, seq_len, head_dim = 3, 129, 128
    q = torch.randn(batch_heads, 1, head_dim, dtype=torch.float32)
    k = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    config = AttentionConfig(cluster_size=cluster_size)

    scores = torch.matmul(q, k.transpose(-2, -1)) * config.resolve_scale(head_dim)
    ref = torch.matmul(torch.softmax(scores, dim=-1), v)
    out = split_kv_decode_reference(q, k, v, cluster_size, config.resolve_scale(head_dim))

    assert leader_reduce_payload_floats(head_dim) == 130
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (1, 2, 128, 64), (2, 4, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_stage0_matches_reference(shape, dtype):
    q, k, v = make_inputs(shape, dtype)
    config = AttentionConfig(block_n=64, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage0", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage1_fa2_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_n=32, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage1", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (1, 2, 128, 64), (2, 4, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_stage2_matches_reference(shape, dtype):
    q, k, v = make_inputs(shape, dtype)
    head_dim = shape[-1]
    config = AttentionConfig(block_m=head_dim, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage2", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage3_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_n=32, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage3", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
@pytest.mark.parametrize("stage_name", ["stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10"])
def test_stage4_stage5_stage6_stage7_stage8_stage9_stage10_match_reference(stage_name):
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=32, block_n=32, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage(stage_name, q, k, v, config)
    rtol, atol = (3e-2, 3e-2) if stage_name == "stage7" else (2e-2, 2e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage11_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage11", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage12_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage12", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage12_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage12", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage13_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage13", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage14_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage14", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage15_matches_reference_small():
    q, k, v = make_inputs((1, 1, 64, 64), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage15", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage16_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=128, num_threads=256)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage16", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage16_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=128, num_threads=256, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage16", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage16_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=128, num_threads=256)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage16", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage16_autotune_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=128, num_threads=256, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage16", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_autotune_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_matches_reference_warpspec_twostage_backend():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=128, num_threads=256, num_stages_kv=2)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage18_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage18", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage18_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage18", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage19_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage19", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage19_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage19", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage20_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage20", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage20_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage20", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage21_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage21", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage21_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage21", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage22_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=128, block_n=128, num_threads=256, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage22", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage22_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=128, block_n=128, num_threads=256, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage22", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)
