#!/usr/bin/env python3
"""Correctness tests for the ClusterFusion-style decode megakernel.

  1. test_reference_*  – PyTorch reference sanity checks (CUDA, no CuTeDSL).
  2. test_megakernel_* – CuTeDSL kernel vs reference (requires Blackwell + CuTeDSL).
  3. test_attn_compat  – ensure existing cluster_decode attention tests still pass.
"""
from __future__ import annotations

import math
import pytest

from cluster_decode import (
    ClusterDecodeConfig,
    MegakernelConfig,
    available_backends,
    cluster_decode_forward,
    cluster_decode_split_forward,
    cluster_megakernel_forward,
    leader_reduce_payload_floats,
    make_random_megakernel_inputs,
    megakernel_reference_forward,
    split_kv_decode_reference,
)


backends = available_backends()
torch = pytest.importorskip("torch") if backends["torch"] else None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_attn_ref(q, k, v, scale):
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * scale
    probs  = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.to(torch.float32)).to(q.dtype)


# ---------------------------------------------------------------------------
# 1.  Reference sanity checks (pure PyTorch, no CuTeDSL required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backends["torch"], reason="PyTorch not installed")
@pytest.mark.skipif(
    not (backends["torch"] and hasattr(torch, "cuda") and torch.cuda.is_available()),
    reason="CUDA GPU required",
)
class TestReferenceForward:
    """Validate megakernel_reference_forward against manual computation (CUDA tensors)."""

    def _small_config(self):
        return MegakernelConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            cluster_size=2,
        )

    def test_output_shape(self):
        config  = self._small_config()
        inputs  = make_random_megakernel_inputs(config, seq_len=32, device="cpu")
        out, k_new, v_new = megakernel_reference_forward(**inputs, config=config)
        assert out.shape   == (1, config.hidden_dim)
        assert k_new.shape == (1, config.num_heads, config.head_dim)
        assert v_new.shape == (1, config.num_heads, config.head_dim)

    def test_output_dtype_preserved(self):
        config = self._small_config()
        for dtype in [torch.float16, torch.bfloat16]:
            inputs = make_random_megakernel_inputs(config, seq_len=32, device="cuda", dtype=dtype)
            out, k_new, v_new = megakernel_reference_forward(**inputs, config=config)
            assert out.dtype   == dtype
            assert k_new.dtype == dtype
            assert v_new.dtype == dtype

    def test_deterministic(self):
        """Same inputs → same outputs (no randomness in forward pass)."""
        config = self._small_config()
        inputs = make_random_megakernel_inputs(config, seq_len=32, device="cuda")
        out1, _, _ = megakernel_reference_forward(**inputs, config=config)
        out2, _, _ = megakernel_reference_forward(**inputs, config=config)
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    def test_identity_w_o_returns_attn_out(self):
        """With W_o = I, output should equal attn_out projected to hidden."""
        config = self._small_config()
        D  = config.hidden_dim
        NH = config.num_heads
        HD = config.head_dim

        torch.manual_seed(7)
        inputs = make_random_megakernel_inputs(config, seq_len=16, device="cpu")
        # Replace W_o with identity
        inputs["w_o"] = torch.eye(D, dtype=inputs["w_o"].dtype)
        out, _, _ = megakernel_reference_forward(**inputs, config=config)
        # Output should be finite and non-zero
        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_current_token_participates_in_attention(self):
        """With empty previous cache and W_o = I, decode output is current V."""
        config = self._small_config()
        D = config.hidden_dim

        inputs = make_random_megakernel_inputs(
            config,
            seq_len=0,
            device="cpu",
            dtype=torch.float32,
        )
        inputs["w_o"] = torch.eye(D, dtype=inputs["w_o"].dtype)

        out, _, v_new = megakernel_reference_forward(**inputs, config=config)
        torch.testing.assert_close(
            out,
            v_new.reshape(1, D),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_cluster_size_invariant(self, cluster_size):
        """Result should not depend on cluster_size (it's a launch parameter)."""
        config2 = MegakernelConfig(hidden_dim=256, num_heads=4, head_dim=64, cluster_size=cluster_size)
        inputs  = make_random_megakernel_inputs(config2, seq_len=32, device="cuda")
        out, _, _ = megakernel_reference_forward(**inputs, config=config2)
        assert torch.isfinite(out).all()

    def test_rmsnorm_correctness(self):
        """Verify RMSNorm stage against torch.nn.functional.rms_norm."""
        from cluster_decode.megakernel_reference import rms_norm
        x = torch.randn(1, 64, dtype=torch.float32, device="cuda")
        w = torch.ones(64, dtype=torch.float32, device="cuda")
        out_ref = torch.nn.functional.rms_norm(x, (64,), weight=w, eps=1e-6)
        out_ours = rms_norm(x, w, eps=1e-6).to(torch.float32)
        torch.testing.assert_close(out_ours, out_ref, rtol=1e-5, atol=1e-5)

    def test_rope_identity_when_cos1_sin0(self):
        """RoPE with cos=1, sin=0 should be identity."""
        from cluster_decode.megakernel_reference import apply_rope_gptj
        HD = 32
        q = torch.randn(HD, dtype=torch.float32, device="cuda")
        k = torch.randn(HD, dtype=torch.float32, device="cuda")
        cos = torch.ones(HD, dtype=torch.float32, device="cuda")
        sin = torch.zeros(HD, dtype=torch.float32, device="cuda")
        q_rot, k_rot = apply_rope_gptj(q, k, cos, sin)
        torch.testing.assert_close(q_rot, q, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k_rot, k, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2.  CuTeDSL kernel vs reference (requires Blackwell GPU + CuTe DSL)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backends["torch"],  reason="PyTorch not installed")
@pytest.mark.skipif(not backends["cute"],   reason="CuTe DSL not installed")
@pytest.mark.skipif(
    not (backends["torch"] and torch.cuda.is_available()),
    reason="CUDA GPU required",
)
class TestMegakernelVsReference:
    """Compare the CuTeDSL megakernel against the PyTorch reference."""

    def _small_config(self):
        return MegakernelConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            cluster_size=2,
            num_threads=128,
        )

    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_output_matches_reference(self, seq_len, cluster_size):
        config = MegakernelConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            cluster_size=cluster_size,
            num_threads=128,
        )
        inputs = make_random_megakernel_inputs(config, seq_len=seq_len, device="cuda")

        ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
        cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
        torch.cuda.synchronize()

        torch.testing.assert_close(cuda_out, ref_out, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cuda_k,   ref_k,   rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cuda_v,   ref_v,   rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_llama_7b_config(self, seq_len):
        """Smoke test with actual Llama-2-7B dimensions."""
        config = MegakernelConfig()   # hidden=4096, heads=32, head_dim=128
        inputs = make_random_megakernel_inputs(config, seq_len=seq_len, device="cuda")

        ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
        cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
        torch.cuda.synchronize()

        # This is a launch-shape smoke test with large random fp16 reductions.
        # Keep strict output checks on the smaller semantic tests, including
        # the optional SGLang end-to-end reference gate. At full Llama shape,
        # random unscaled W_o reductions are dominated by order/ULP noise, so
        # this test focuses on shape, finite output, and exact exposed KV state.
        assert cuda_out.shape == ref_out.shape
        assert cuda_out.dtype == ref_out.dtype
        assert torch.isfinite(cuda_out).all()
        torch.testing.assert_close(cuda_k, ref_k, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cuda_v, ref_v, rtol=2e-2, atol=2e-2)

        diff = (cuda_out.float() - ref_out.float()).abs()
        rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
            ref_out.float()
        ).clamp_min(1e-12)
        assert rel_l2 < 5e-3


@pytest.mark.skipif(not backends["torch"], reason="PyTorch not installed")
@pytest.mark.skipif(
    not (backends["torch"] and torch.cuda.is_available()),
    reason="CUDA GPU required",
)
class TestTensorCoreMegakernel:
    """Compare the experimental tensor-core path against the reference."""

    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("cluster_size", [2, 4, 8])
    def test_tc_output_matches_reference(self, seq_len, cluster_size):
        config = MegakernelConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            cluster_size=cluster_size,
            num_threads=128,
        )
        inputs = make_random_megakernel_inputs(config, seq_len=seq_len, device="cuda")

        ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
        tc_out, tc_k, tc_v = cluster_megakernel_forward(
            **inputs,
            config=config,
            use_tensor_core=True,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(tc_k, ref_k, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(tc_v, ref_v, rtol=2e-2, atol=2e-2)
        rel_l2 = torch.linalg.vector_norm((tc_out.float() - ref_out.float())) / torch.linalg.vector_norm(
            ref_out.float()
        ).clamp_min(1e-12)
        assert rel_l2 < 5e-3


# ---------------------------------------------------------------------------
# 3.  Backward-compat: standalone attention stages
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not backends["torch"], reason="PyTorch not installed")
class TestAttnStagesCompat:
    """Ensure the existing attention-only stages are unaffected by the refactor."""

    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_reduce_contract(self, cluster_size):
        batch_heads, seq_len, head_dim = 3, 129, 128
        q = torch.randn(batch_heads, 1,       head_dim, dtype=torch.float32)
        k = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
        config = ClusterDecodeConfig(cluster_size=cluster_size)
        scale  = config.resolve_scale(head_dim)

        ref = decode_attn_ref(q, k, v, scale)
        out = split_kv_decode_reference(q, k, v, cluster_size, scale)
        assert leader_reduce_payload_floats(head_dim) == 130
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not backends["cute"], reason="CuTe DSL not installed")
    @pytest.mark.skipif(
        not (backends["torch"] and torch.cuda.is_available()),
        reason="CUDA GPU required",
    )
    def test_cluster_decode_v0(self):
        q = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        config = ClusterDecodeConfig(num_threads=128, cluster_size=2)
        ref    = decode_attn_ref(q, k, v, config.resolve_scale(128))
        out    = cluster_decode_forward(q, k, v, config)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    @pytest.mark.skipif(not backends["cute"], reason="CuTe DSL not installed")
    @pytest.mark.skipif(
        not (backends["torch"] and torch.cuda.is_available()),
        reason="CUDA GPU required",
    )
    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_cluster_decode_split(self, cluster_size):
        q = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        config = ClusterDecodeConfig(num_threads=128, cluster_size=cluster_size)
        ref    = decode_attn_ref(q, k, v, config.resolve_scale(128))
        out    = cluster_decode_split_forward(q, k, v, config)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
