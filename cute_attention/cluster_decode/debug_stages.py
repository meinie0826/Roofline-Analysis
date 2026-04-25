"""Minimal stage-by-stage debug for the single-CTA megakernel.

Compares each stage independently to isolate where the bug is.
Run: PYTHONPATH=cute_attention python3 -m cluster_decode.debug_stages
"""

import torch
from cluster_decode.common import MegakernelConfig
from cluster_decode.megakernel_reference import megakernel_reference_forward, make_random_megakernel_inputs


def debug_stages():
    config = MegakernelConfig(
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        cluster_size=2,  # doesn't matter for single-CTA
        num_threads=128,
    )
    seq_len = 32
    inputs = make_random_megakernel_inputs(config, seq_len=seq_len, device="cuda")

    D = config.hidden_dim
    NH = config.num_heads
    HD = config.head_dim
    scale = config.resolve_scale()
    eps = 1e-6

    h = inputs["hidden_states"].float()  # (1, D)
    w_qkv = inputs["w_qkv"].float()     # (3D, D)
    w_o = inputs["w_o"].float()          # (D, D)
    rms_w = inputs["rms_weight"].float() # (D,)
    k_cache = inputs["k_cache"].float()  # (S, NH, HD)
    v_cache = inputs["v_cache"].float()  # (S, NH, HD)

    print(f"=== Stage 0: RMSNorm ===")
    total_l2 = h.pow(2).sum(dim=-1).item()
    mean_l2 = total_l2 / D
    rms_rcp = (mean_l2 + eps) ** (-0.5)
    h_norm = h * rms_rcp * rms_w
    print(f"  total_l2={total_l2:.4f}, mean_l2={mean_l2:.4f}, rms_rcp={rms_rcp:.6f}")
    print(f"  h_norm[0, 0:4] = {h_norm[0, 0:4]}")

    print(f"\n=== Stage 1: QKV GEMM ===")
    qkv = h_norm @ w_qkv.T  # (1, 3D)
    print(f"  qkv shape: {qkv.shape}")
    print(f"  qkv[0, 0:4] = {qkv[0, 0:4]}")  # head 0 Q element 0-3

    # Per-head Q/K/V
    Q = qkv[:, :D].reshape(1, NH, HD)
    K = qkv[:, D:2*D].reshape(1, NH, HD)
    V = qkv[:, 2*D:].reshape(1, NH, HD)
    print(f"  Q[0, 0, 0:4] = {Q[0, 0, 0:4]}")  # head 0 Q

    print(f"\n=== Stage 2: RoPE (cos=1, sin=0 → identity) ===")
    cos = inputs["cos_rope"].float()
    sin = inputs["sin_rope"].float()
    Q_rot = Q.clone()  # identity since cos=1, sin=0
    K_rot = K.clone()
    print(f"  Q_rot[0, 0, 0:4] = {Q_rot[0, 0, 0:4]}")

    print(f"\n=== Stage 3: Attention ===")
    # Head 0 attention scores
    q0 = Q_rot[0, 0]  # (HD,)
    scores0 = (q0.unsqueeze(0) @ k_cache[:, 0, :].T).squeeze(0) * scale  # (S,)
    print(f"  scores0 max={scores0.max():.4f}, min={scores0.min():.4f}")
    probs0 = torch.softmax(scores0, dim=-1)
    attn0 = probs0 @ v_cache[:, 0, :]  # (HD,)
    print(f"  attn_out[0] (head 0) [0:4] = {attn0[0:4]}")

    # All heads
    attn_all = torch.zeros(1, D, device="cuda")
    for h_idx in range(NH):
        q_h = Q_rot[0, h_idx]
        scores_h = (q_h.unsqueeze(0) @ k_cache[:, h_idx, :].T).squeeze(0) * scale
        probs_h = torch.softmax(scores_h, dim=-1)
        attn_h = probs_h @ v_cache[:, h_idx, :]
        attn_all[0, h_idx*HD:(h_idx+1)*HD] = attn_h
    print(f"  attn_vec[0, 0:8] = {attn_all[0, 0:8]}")

    print(f"\n=== Stage 4: W_o GEMM ===")
    # output = attn_vec @ w_o.T
    output_ref = attn_all @ w_o.T
    print(f"  output_ref[0, 0:8] = {output_ref[0, 0:8]}")

    # Per-head W_o contributions
    for h_idx in range(NH):
        attn_h = attn_all[0, h_idx*HD:(h_idx+1)*HD]
        wo_h = attn_h @ w_o[h_idx*HD:(h_idx+1)*HD, :].T
        print(f"  W_o head {h_idx} [0:4] = {wo_h[0:4]}")

    # Now run the kernel
    from cluster_decode.cluster_megakernel import cluster_megakernel_forward
    ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
    cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
    torch.cuda.synchronize()

    print(f"\n=== KERNEL OUTPUT ===")
    print(f"  cuda_out[0, 0:8] = {cuda_out[0, 0:8].float()}")
    print(f"  ref_out[0, 0:8]  = {ref_out[0, 0:8].float()}")
    print(f"  diff[0, 0:8] = {(cuda_out.float() - ref_out.float())[0, 0:8]}")
    print(f"  k_new diff max: {(cuda_k.float() - ref_k.float()).abs().max():.6f}")
    print(f"  v_new diff max: {(cuda_v.float() - ref_v.float()).abs().max():.6f}")

    # If k/v match but output doesn't, the bug is in attention or W_o
    # If k/v also mismatch, the bug is in RMSNorm or QKV or RoPE


if __name__ == "__main__":
    debug_stages()