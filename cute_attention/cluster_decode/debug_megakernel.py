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

    hidden_dim = config.hidden_dim
    num_heads = config.num_heads
    head_dim = config.head_dim
    scale = config.resolve_scale()

    # ---- CUDA Kernel ----
    cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
    torch.cuda.synchronize()

    print("\n=== CUDA KERNEL ===")
    print(f"cuda_out stats: mean={cuda_out.float().mean():.4f}, std={cuda_out.float().std():.4f}")
    print(f"cuda_out sample [0, 0:8]: {cuda_out[0, 0:8].float()}")

    # ---- Manual reference computation for comparison ----
    h_f = inputs["hidden_states"].float()                # (1, D)
    w_qkv_f = inputs["w_qkv"].float()                   # (3D, D)
    w_o_f = inputs["w_o"].float()                       # (D, D)

    # RMSNorm
    rms_weight_f = inputs["rms_weight"].float()
    total_l2_ref = h_f.pow(2).sum(dim=-1).item()
    mean_l2_ref = total_l2_ref / hidden_dim
    rms_rcp_ref = 1.0 / (mean_l2_ref ** 0.5 + 1e-6) ** 0.5  # wrong
    rms_rcp_ref = (mean_l2_ref + 1e-6) ** (-0.5)
    h_norm = h_f * rms_rcp_ref * rms_weight_f

    print(f"\n=== DETAILED REFERENCE ===")
    print(f"total l2 (sum of x^2): {total_l2_ref:.4f}")
    print(f"mean l2: {mean_l2_ref:.4f}")
    print(f"rms_rcp: {rms_rcp_ref:.4f}")
    print(f"h_norm[0, 0:4]: {h_norm[0, 0:4]}")

    # QKV
    qkv_ref = h_norm @ w_qkv_f.T
    Q_ref = qkv_ref[:, :hidden_dim].reshape(1, num_heads, head_dim)
    V_ref = qkv_ref[:, 2*hidden_dim:].reshape(1, num_heads, head_dim)
    print(f"Q head 0 [0:4]: {Q_ref[0, 0, 0:4]}")

    # The random debug inputs use cos=1 and sin=0, so RoPE is identity.
    Q_rot = Q_ref
    print(f"Q_rot head 0 [0:4] (after RoPE, cos=1 sin=0): {Q_rot[0, 0, 0:4]}")

    # Attention scores for head 0
    k_f = inputs["k_cache"].float()
    v_f = inputs["v_cache"].float()
    q0 = Q_rot[0, 0]  # (head_dim,)
    scores0 = (q0 @ k_f[:, 0, :].T) * scale
    print(f"attn scores head 0 max: {scores0.max():.4f}, mean: {scores0.mean():.4f}")
    probs0 = torch.softmax(scores0, dim=-1)
    attn_out0 = probs0 @ v_f[:, 0, :]
    print(f"attn_out head 0 [0:4]: {attn_out0[0:4]}")

    # W_o contribution for head 0: output_col += attn[d] * w_o[output_col, d].
    wo_head0_out = attn_out0 @ w_o_f[:, 0:head_dim].T
    print(f"W_o output head 0 [0:4]: {wo_head0_out[0:4]}")

    print("\n=== COMPARISON ===")
    diff = (cuda_out.float() - ref_out.float()).abs()
    print(f"max abs diff: {diff.max():.4f}")
    print(f"mean abs diff: {diff.mean():.4f}")
    rel_diff = diff / (ref_out.float().abs() + 1e-6)
    print(f"max rel diff: {rel_diff.max():.4f}")

    # Also compare k_new, v_new
    k_diff = (cuda_k.float() - ref_k.float()).abs()
    v_diff = (cuda_v.float() - ref_v.float()).abs()
    print(f"\nk_new max abs diff: {k_diff.max():.4f}")
    print(f"v_new max abs diff: {v_diff.max():.4f}")

    v_new_diff = (cuda_v.float() - V_ref.float()).abs()
    print(f"v_new vs QKV V max abs diff: {v_new_diff.max():.4f}")
    print(f"\nRef output[0, 0:8]: {ref_out[0, 0:8].float()}")
    print(f"Cuda output[0, 0:8]: {cuda_out[0, 0:8].float()}")


if __name__ == "__main__":
    debug_megakernel()
