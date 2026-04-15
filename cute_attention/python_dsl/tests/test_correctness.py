#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

import pytest

from kernels import AttentionConfig, available_backends, run_stage
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
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_autotune_matches_reference_small():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_autotune_matches_reference_multiblock():
    q, k, v = make_inputs((1, 1, 256, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, num_stages_kv=3, autotune=True)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL is not installed")
def test_stage17_matches_reference_ampere_multistage_backend():
    q, k, v = make_inputs((1, 1, 128, 128), torch.float16)
    config = AttentionConfig(block_m=64, block_n=64, num_threads=128, num_stages_kv=3)
    ref = causal_attention_reference(q, k, v, config)
    out = run_stage("stage17", q, k, v, config)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)
