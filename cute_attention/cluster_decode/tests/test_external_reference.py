#!/usr/bin/env python3
"""Optional SGLang/vLLM reference checks.

These tests are skipped unless the frameworks are installed. They compare the
supported dense GPT-J RoPE path against the local PyTorch reference.
"""
from __future__ import annotations

import pytest

from cluster_decode import (
    MegakernelConfig,
    external_megakernel_reference_forward,
    make_random_megakernel_inputs,
    megakernel_reference_forward,
    probe_framework_import,
    validate_supported_external_config,
)


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_external_reference_matches_local_reference(framework):
    status = probe_framework_import(framework)
    if not status.available:
        pytest.skip(f"{framework} unavailable: {status.error}")

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
    external = external_megakernel_reference_forward(
        framework,
        **inputs,
        config=config,
    )

    for actual, expected in zip(external, local, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


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
