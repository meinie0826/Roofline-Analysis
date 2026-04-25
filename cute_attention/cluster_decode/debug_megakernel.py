"""Debug script to print intermediate values from megakernel vs reference.

Run on server:
  PYTHONPATH=cute_attention python3 -m cluster_decode.debug_megakernel
"""

import torch
from cluster_decode.common import MegakernelConfig
from cluster_decode.megakernel_reference import megakernel_reference_forward, make_random_megakernel_inputs
from cluster_decode.cluster_megakernel import cluster_megakernel_forward


def debug_megakernel():
    config = MegakernelConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        cluster_size=2,
        num_threads=128,
    )
    seq_len = 32
    inputs = make_random_megakernel_inputs(config, seq_len=seq_len, device="cuda")

    # ---- Reference ----
    ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
    print("=== REFERENCE ===")
    print(f"output shape: {ref_out.shape}, dtype: {ref_out.dtype}")
    print(f"output stats: mean={ref_out.float().mean():.4f}, std={ref_out.float().std():.4f}")
    print(f"output sample [0, 0:8]: {ref_out[0, 0:8].float()}")
    print(f"k_new stats: mean={ref_k.float().mean():.4f}")
    print(f"v_new stats: mean={ref_v.float().mean():.4f}")

    # ---- CUDA Kernel ----
    # Need to expose scratch buffers for debugging
    from cluster_decode.cluster_megakernel import _make_cluster_megakernel_host, _MEGAKERNEL_COMPILED_CACHE
    from cluster_decode.common import HAS_CUTE, cute, from_dlpack, cutlass

    if not HAS_CUTE:
        print("No CuTeDSL available")
        return

    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    head_dim = config.head_dim
    cluster_size = config.cluster_size
    scale = config.resolve_scale()

    output_buf = torch.zeros((1, hidden_dim), device="cuda", dtype=torch.float16)
    k_new = torch.zeros((1, num_heads, head_dim), device="cuda", dtype=torch.float16)
    v_new = torch.zeros((1, num_heads, head_dim), device="cuda", dtype=torch.float16)

    scratch_l2 = torch.zeros((num_heads, cluster_size), device="cuda", dtype=torch.float32)
    scratch_max = torch.zeros((num_heads, cluster_size), device="cuda", dtype=torch.float32)
    scratch_sum = torch.zeros((num_heads, cluster_size), device="cuda", dtype=torch.float32)
    scratch_out = torch.zeros((num_heads, cluster_size, head_dim), device="cuda", dtype=torch.float16)
    scratch_wo = torch.zeros((num_heads, hidden_dim), device="cuda", dtype=torch.float16)

    def _wrap(t):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic()

    tensors = {
        "hidden_states": _wrap(inputs["hidden_states"]),
        "w_qkv": _wrap(inputs["w_qkv"]),
        "w_o": _wrap(inputs["w_o"]),
        "k_cache": _wrap(inputs["k_cache"]),
        "v_cache": _wrap(inputs["v_cache"]),
        "rms_weight": _wrap(inputs["rms_weight"]),
        "cos_rope": _wrap(inputs["cos_rope"]),
        "sin_rope": _wrap(inputs["sin_rope"]),
        "output": _wrap(output_buf),
        "k_out": _wrap(k_new),
        "v_out": _wrap(v_new),
        "scratch_l2": _wrap(scratch_l2),
        "scratch_max": _wrap(scratch_max),
        "scratch_sum": _wrap(scratch_sum),
        "scratch_out": _wrap(scratch_out),
        "scratch_wo": _wrap(scratch_wo),
    }

    cache_key = (seq_len, config)
    compiled = _MEGAKERNEL_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host = _make_cluster_megakernel_host(seq_len=seq_len, config=config)
        compiled = cute.compile(
            host,
            tensors["hidden_states"], tensors["w_qkv"], tensors["w_o"],
            tensors["k_cache"], tensors["v_cache"],
            tensors["rms_weight"], tensors["cos_rope"], tensors["sin_rope"],
            tensors["output"], tensors["k_out"], tensors["v_out"],
            tensors["scratch_l2"], tensors["scratch_max"], tensors["scratch_sum"],
            tensors["scratch_out"], tensors["scratch_wo"],
            cutlass.Float32(scale),
        )
        _MEGAKERNEL_COMPILED_CACHE[cache_key] = compiled

    compiled(
        tensors["hidden_states"], tensors["w_qkv"], tensors["w_o"],
        tensors["k_cache"], tensors["v_cache"],
        tensors["rms_weight"], tensors["cos_rope"], tensors["sin_rope"],
        tensors["output"], tensors["k_out"], tensors["v_out"],
        tensors["scratch_l2"], tensors["scratch_max"], tensors["scratch_sum"],
        tensors["scratch_out"], tensors["scratch_wo"],
        cutlass.Float32(scale),
    )
    torch.cuda.synchronize()

    cuda_out = scratch_wo.sum(dim=0).unsqueeze(0).to(torch.float16)

    print("\n=== CUDA KERNEL ===")
    print(f"scratch_l2 (RMSNorm partial l2 per head per CTA):")
    print(scratch_l2)
    print(f"\nscratch_max (attn partial max per head per CTA):")
    print(scratch_max)
    print(f"\nscratch_sum (attn partial sum per head per CTA):")
    print(scratch_sum)
    print(f"\nscratch_out[0, 0, 0:8] (attn partial output, head 0, CTA 0):")
    print(scratch_out[0, 0, 0:8].float())
    print(f"\nscratch_wo[0, 0:8] (W_o output, head 0):")
    print(scratch_wo[0, 0:8].float())
    print(f"\nscratch_wo[1, 0:8] (W_o output, head 1):")
    print(scratch_wo[1, 0:8].float())
    print(f"\ncuda_out stats: mean={cuda_out.float().mean():.4f}, std={cuda_out.float().std():.4f}")
    print(f"cuda_out sample [0, 0:8]: {cuda_out[0, 0:8].float()}")

    print("\n=== COMPARISON ===")
    diff = (cuda_out.float() - ref_out.float()).abs()
    print(f"max abs diff: {diff.max():.4f}")
    print(f"mean abs diff: {diff.mean():.4f}")
    rel_diff = diff / (ref_out.float().abs() + 1e-6)
    print(f"max rel diff: {rel_diff.max():.4f}")

    # Also compare k_new, v_new
    k_diff = (k_new.float() - ref_k.float()).abs()
    v_diff = (v_new.float() - ref_v.float()).abs()
    print(f"\nk_new max abs diff: {k_diff.max():.4f}")
    print(f"v_new max abs diff: {v_diff.max():.4f}")

    # Reference intermediate: compute RMSNorm result manually
    h_f = inputs["hidden_states"].float()
    l2_ref = h_f.pow(2).sum(dim=-1)
    print(f"\nReference total l2: {l2_ref.item():.4f}")
    print(f"Kernel total l2 (from scratch_l2 sum): {scratch_l2.sum(dim=1)}")

    # Reference: individual head QKV
    h_norm = torch.nn.functional.rms_norm(h_f, (hidden_dim,), weight=inputs["rms_weight"].float(), eps=1e-6)
    qkv_ref = h_norm @ inputs["w_qkv"].float().T
    Q_ref = qkv_ref[:, :hidden_dim].reshape(1, num_heads, head_dim)
    print(f"\nReference Q[0, 0, 0:4]: {Q_ref[0, 0, 0:4]}")
    # The kernel's Q should be in scratch_out indirectly, but let's check scratch_wo directly

    # Per-head W_o reference
    for h_idx in range(num_heads):
        attn_ref_h = ref_out.float().reshape(num_heads, head_dim)  # wrong, this is output not attn
    # Better: compute attn_out per head from reference
    # Actually we can't easily extract per-head attn from reference output
    # Let's just compare the final outputs
    print(f"\nRef output[0, 0:8]: {ref_out[0, 0:8].float()}")
    print(f"Cuda output[0, 0:8]: {cuda_out[0, 0:8].float()}")


if __name__ == "__main__":
    debug_megakernel()