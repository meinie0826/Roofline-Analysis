#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

import pytest

from kernels import AttentionConfig, available_backends, run_stage


backends = available_backends()
torch = pytest.importorskip("torch") if backends["torch"] else None


def make_inputs(shape, dtype):
    q = torch.randn(*shape, device="cuda", dtype=dtype)
    k = torch.randn(*shape, device="cuda", dtype=dtype)
    v = torch.randn(*shape, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (1, 2, 128, 64), (2, 4, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_stage0_matches_reference(shape, dtype):
    q, k, v = make_inputs(shape, dtype)
    config = AttentionConfig(block_n=64, num_threads=128)
    ref = run_stage("reference", q, k, v, config)
    out = run_stage("stage0", q, k, v, config)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not backends["torch"], reason="PyTorch is not installed")
@pytest.mark.parametrize("stage_name", ["stage1", "stage2"])
def test_intermediate_reference_stages_match_reference(stage_name):
    q, k, v = make_inputs((1, 2, 96, 64), torch.float32)
    config = AttentionConfig(block_m=32, block_n=64)
    ref = run_stage("reference", q, k, v, config)
    out = run_stage(stage_name, q, k, v, config)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
