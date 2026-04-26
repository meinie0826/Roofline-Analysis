#!/usr/bin/env python3
"""Optional SGLang reference checks.

These tests are skipped unless SGLang is installed. They compare the supported
dense GPT-J RoPE path against the local PyTorch reference.
"""
from __future__ import annotations

import pytest

from cluster_decode import (
    MegakernelConfig,
    available_backends,
    cluster_megakernel_forward,
    make_random_megakernel_inputs,
    megakernel_reference_forward,
    probe_sglang_import,
    sglang_megakernel_reference_forward,
    validate_supported_external_config,
)


torch = pytest.importorskip("torch")
backends = available_backends()


def test_sglang_reference_matches_local_reference():
    status = probe_sglang_import()
    if not status.available:
        pytest.skip(f"SGLang unavailable: {status.error}")

    config = MegakernelConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        cluster_size=2,
    )
    inputs = make_random_megakernel_inputs(
        config,
        seq_len=32,
        device="cpu",
        dtype=torch.float32,
    )

    local = megakernel_reference_forward(**inputs, config=config)
    external = sglang_megakernel_reference_forward(
        **inputs,
        config=config,
    )

    for actual, expected in zip(external, local, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not backends["cute"], reason="CuTe DSL not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_megakernel_matches_sglang_reference():
    status = probe_sglang_import()
    if not status.available:
        pytest.skip(f"SGLang unavailable: {status.error}")

    config = MegakernelConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        cluster_size=2,
        num_threads=128,
    )
    inputs = make_random_megakernel_inputs(
        config,
        seq_len=128,
        device="cuda",
        dtype=torch.float16,
    )

    sglang_out, sglang_k, sglang_v = sglang_megakernel_reference_forward(
        **inputs,
        config=config,
    )
    cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
    torch.cuda.synchronize()

    torch.testing.assert_close(cuda_out, sglang_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cuda_k, sglang_k, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cuda_v, sglang_v, rtol=2e-2, atol=2e-2)


def test_external_config_rejects_unsupported_branches():
    config = MegakernelConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        cluster_size=2,
    )

    validate_supported_external_config(config)
    with pytest.raises(ValueError, match="GQA/MQA"):
        validate_supported_external_config(config, num_kv_heads=2)
    with pytest.raises(ValueError, match="non-GPT-J RoPE"):
        validate_supported_external_config(config, rope_style="neox")
    with pytest.raises(ValueError, match="paged KV cache"):
        validate_supported_external_config(config, paged_kv=True)
