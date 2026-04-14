#!/usr/bin/env python
"""
FA4 Roofline Analysis — Theory Only (No Speculation)
=====================================================

This script computes ONLY what can be derived from public hardware specs and
the FA4 algorithm definition. No measured performance numbers are hardcoded.

What IS derived here (provable from first principles):
  - Per-resource cycle counts from hardware specs (TC ops/cycle, MUFU ops/cycle, SMEM BW)
  - Theoretical upper bounds (ceilings) for each resource
  - Arithmetic intensity
  - The asymmetry problem: why B200 creates new bottlenecks

What is NOT here (must come from benchmark_ablation_sm100.py):
  - Actual speedup of ping-pong
  - Actual effect of conditional rescaling
  - Actual effect of exp2 emulation
  - LPT improvement
  - Any measured TFLOPs/s

Hardware specs used (public, from NVIDIA datasheets):
  H100 SXM: 989 TFLOPs BF16, 132 SMs, 1.98 GHz boost
             TC: 4096 BF16 MACs/cycle/SM, MUFU: 16 ops/cycle/SM, SMEM: 128 B/cycle/SM
  B200 SXM: 2250 TFLOPs BF16, 148 SMs, 1.85 GHz boost
             TC: 8192 BF16 MACs/cycle/SM, MUFU: 16 ops/cycle/SM, SMEM: 128 B/cycle/SM

Usage:
    python fa4/fa4_roofline_theory.py                          # D=128, save to ./plots
    python fa4/fa4_roofline_theory.py --hdim 64,96,128,192    # multiple hdims
    python fa4/fa4_roofline_theory.py --out-dir ./my_plots
"""

import argparse
import math
import os
from typing import Dict, List

# ── Hardware specs (from NVIDIA public datasheets) ────────────────────────────

class HW:
    def __init__(self, name, tc_bf16_per_sm, mufu_per_sm, smem_bw_per_sm,
                 num_sms, clock_ghz, peak_bf16_tflops):
        self.name = name
        self.tc   = tc_bf16_per_sm     # BF16 MACs/cycle/SM  (from datasheet)
        self.mufu = mufu_per_sm        # exp2 ops/cycle/SM   (from datasheet)
        self.smem = smem_bw_per_sm     # bytes/cycle/SM read (from datasheet)
        self.nsms = num_sms
        self.clk  = clock_ghz
        self.peak = peak_bf16_tflops   # TFLOPs/s full chip

# Source: NVIDIA H100 and B200 Technical Overview / GPU Spec Sheets
H100 = HW("H100 SXM5", tc_bf16_per_sm=4096, mufu_per_sm=16, smem_bw_per_sm=128,
           num_sms=132, clock_ghz=1.98, peak_bf16_tflops=989)
B200 = HW("B200 SXM",  tc_bf16_per_sm=8192, mufu_per_sm=16, smem_bw_per_sm=128,
           num_sms=148, clock_ghz=1.85, peak_bf16_tflops=2250)


# ── Provable cycle counts ─────────────────────────────────────────────────────

def tile_cycles(hw: HW, M: int, N: int, D: int) -> Dict[str, float]:
    """
    Compute per-tile clock cycle lower bounds for each hardware resource.

    All formulas are derived from the FA forward pass algorithm:
      1. QK^T MMA: M×D matrix × D×N matrix → M×N scores  [2*M*N*D MACs]
      2. PV  MMA : M×N probs × N×D values  → M×D output  [2*M*N*D MACs]
      Softmax: M*N exp2 operations
      SMEM reads (SS-mode MMA reads both operands from SMEM):
        QK^T: Q tile (M×D) + K tile (D×N) from SMEM
        PV:   V tile (N×D) from SMEM  [P comes from TMEM on SM100 → no SMEM read]

    Returns
    -------
    dict with keys: tc, exp, smem, bottleneck, bottleneck_name
    """
    # Tensor Core cycles
    # Each MMA: 2*M*N*D MACs. Two MMAs total. TC throughput = hw.tc MACs/cycle/SM.
    tc_cyc = (2 * 2 * M * N * D) / hw.tc  # = 4*M*N*D / hw.tc

    # MUFU exp2 cycles (MUFU.EX2 instruction)
    # Softmax row: M rows × N exp2 per row = M*N total exp2 ops
    # MUFU throughput = hw.mufu ops/cycle/SM
    exp_cyc = (M * N) / hw.mufu

    # SMEM read cycles (BF16 = 2 bytes/element)
    # QK^T (SS-mode): reads Q tile + K tile from SMEM
    #   Q tile: M×D elements, K tile: D×N elements → (M+N)*D*2 bytes per sub-MMA
    #   Number of 128×128 sub-MMAs: ceil(M/128) × ceil(N/128)
    qk_sub = math.ceil(M / 128) * math.ceil(N / 128)
    qk_bytes = qk_sub * (128 + 128) * D * 2  # each sub-MMA: 128*D Q + D*128 K

    # PV (TS-mode on SM100): P is in TMEM (free), reads only V from SMEM
    #   V tile: N×D elements per sub-MMA, ceil(M/128)*ceil(D/128) sub-MMAs
    pv_sub = math.ceil(M / 128) * math.ceil(D / 128)
    pv_bytes = pv_sub * N * 128 * 2  # each sub-MMA reads N*128 V elements

    smem_cyc = (qk_bytes + pv_bytes) / hw.smem

    cycs = {"tc": tc_cyc, "exp": exp_cyc, "smem": smem_cyc}
    bn = max(cycs, key=cycs.get)
    return {**cycs, "bottleneck": cycs[bn], "bottleneck_name": bn.upper()}


def attainable_tflops(hw: HW, bottleneck_cycles: float,
                      tc_cycles: float) -> float:
    """
    Attainable TFLOPs/s given a bottleneck.

    attainable = peak_tflops × (TC_cycles / bottleneck_cycles)

    Derivation:
      peak_tflops = hw.tc × hw.clk × hw.nsms × 2 / 1e12  (× 2 for FMA)
      actual_util = TC_cycles / bottleneck_cycles
      attainable  = peak_tflops × actual_util
    """
    return hw.peak * (tc_cycles / bottleneck_cycles)


# ── Chart 1: Hardware asymmetry ────────────────────────────────────────────────

def plot_asymmetry(out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M, N, D = 128, 128, 128
    h = tile_cycles(H100, M, N, D)
    b = tile_cycles(B200, M, N, D)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "B200 Hardware Asymmetry: Why FA4 Needs New Optimizations\n"
        f"Per-tile cycle counts (M={M}, N={N}, D={D}), derived from NVIDIA datasheets",
        fontsize=11, fontweight="bold",
    )

    resources = ["TC\n(BF16 MMA)", "EXP\n(MUFU.EX2)", "SMEM\nBandwidth"]
    colors = ["#e74c3c", "#e67e22", "#2980b9"]

    # ── Left: absolute cycles ──
    ax = axes[0]
    x = np.arange(3)
    w = 0.35
    h_vals = [h["tc"], h["exp"], h["smem"]]
    b_vals = [b["tc"], b["exp"], b["smem"]]
    bars_h = ax.bar(x - w/2, h_vals, w, label="H100 SXM5", color=[c + "aa" for c in ["#e74c3c","#e67e22","#2980b9"]],
                    edgecolor="white", alpha=0.75)
    bars_b = ax.bar(x + w/2, b_vals, w, label="B200 SXM",  color=colors,
                    edgecolor="white", alpha=0.95)
    for bar, v in [*zip(bars_h, h_vals), *zip(bars_b, b_vals)]:
        ax.text(bar.get_x() + bar.get_width()/2, v + 15,
                f"{v:.0f}c", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(resources, fontsize=10)
    ax.set_ylabel("Cycles per tile", fontsize=10)
    ax.set_title("Absolute cycles (lower = faster)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: ratio vs TC ──
    ax = axes[1]
    h_ratios = [1.0, h["exp"]/h["tc"], h["smem"]/h["tc"]]
    b_ratios = [1.0, b["exp"]/b["tc"], b["smem"]/b["tc"]]
    bars_h2 = ax.bar(x - w/2, h_ratios, w, label="H100 SXM5",
                     color=["#e74c3caa","#e67e22aa","#2980b9aa"], edgecolor="white", alpha=0.75)
    bars_b2 = ax.bar(x + w/2, b_ratios, w, label="B200 SXM",
                     color=colors, edgecolor="white", alpha=0.95)
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="TC = 1.0 (balanced)")
    for bar, v in [*zip(bars_h2, h_ratios), *zip(bars_b2, b_ratios)]:
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(resources, fontsize=10)
    ax.set_ylabel("Resource cycles / TC cycles\n(>1.0 means this resource is SLOWER than TC)", fontsize=9)
    ax.set_title("Bottleneck ratio vs TC\n(>1.0 = performance bottleneck)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: EXP became a co-bottleneck
    ax.annotate(
        "H100: EXP/TC = 0.50×\n(EXP is 2× faster than TC\n → not a bottleneck)",
        xy=(1 - w/2, h_ratios[1]), xytext=(0.4, 0.6),
        fontsize=8, color="#e67e22",
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1),
    )
    ax.annotate(
        "B200: EXP/TC = 1.00×\n(EXP exactly ties TC\n → CO-BOTTLENECK!)",
        xy=(1 + w/2, b_ratios[1]), xytext=(1.25, 1.05),
        fontsize=8.5, color="#d35400", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2),
    )

    plt.tight_layout()
    p = os.path.join(out_dir, "01_hardware_asymmetry.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  {p}")
    return p


# ── Chart 2: Per-tile ceilings vs head dimension ──────────────────────────────

def plot_hdim_ceilings(out_dir: str, hdims: List[int] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if hdims is None:
        hdims = [64, 96, 128, 192, 256]

    # Default tile sizes used by FA4 for each hdim (from flash_fwd_sm100.py defaults)
    tile_for_hdim = {
        64:  (192, 128),
        96:  (192, 128),
        128: (128, 128),
        192: (128, 128),
        256: (128, 128),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Theoretical Per-Resource Ceilings by Head Dimension — B200 SXM\n"
        "Derived from hardware specs + FA4 tile sizes. No measured numbers.",
        fontsize=11, fontweight="bold",
    )

    x = np.arange(len(hdims))
    w = 0.28
    colors = {"tc": "#e74c3c", "exp": "#e67e22", "smem": "#2980b9"}
    resource_labels = {"tc": "TC (BF16 MMA)", "exp": "EXP (MUFU.EX2)", "smem": "SMEM BW"}

    # Panel 1: absolute cycles
    ax = axes[0]
    for ri, res in enumerate(["tc", "exp", "smem"]):
        vals = []
        for D in hdims:
            M, N = tile_for_hdim.get(D, (128, 128))
            r = tile_cycles(B200, M, N, D)
            vals.append(r[res])
        bars = ax.bar(x + (ri-1)*w, vals, w, label=resource_labels[res],
                      color=colors[res], edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+10,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}\n({tile_for_hdim.get(d,(128,128))[0]}×{tile_for_hdim.get(d,(128,128))[1]})"
                        for d in hdims], fontsize=8.5)
    ax.set_ylabel("Cycles per tile", fontsize=10)
    ax.set_title("Per-Resource Cycles (absolute)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: attainable TFLOPs/s if only that resource were the bottleneck
    ax = axes[1]
    for ri, res in enumerate(["tc", "exp", "smem"]):
        atts = []
        for D in hdims:
            M, N = tile_for_hdim.get(D, (128, 128))
            r = tile_cycles(B200, M, N, D)
            atts.append(attainable_tflops(B200, r[res], r["tc"]))
        bars = ax.bar(x + (ri-1)*w, atts, w, label=f"Ceiling if {res.upper()}-bound",
                      color=colors[res], edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, atts):
            ax.text(bar.get_x()+bar.get_width()/2, v+10,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(B200.peak, color="black", ls="--", lw=1.5, alpha=0.6,
               label=f"B200 peak ({B200.peak:.0f} TFLOPs/s)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}\n({tile_for_hdim.get(d,(128,128))[0]}×{tile_for_hdim.get(d,(128,128))[1]})"
                        for d in hdims], fontsize=8.5)
    ax.set_ylabel("Attainable TFLOPs/s (if only this resource limited)", fontsize=9)
    ax.set_title("Per-Resource Ceilings (attainable TFLOPs/s)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "02_per_resource_ceilings_by_hdim.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  {p}")
    return p


# ── Chart 3: Which optimizations target which ceilings ────────────────────────

def plot_optimization_ceiling_map(out_dir: str, D: int = 128):
    """
    Show ONLY what is provable:
    - The ceiling for each resource (derived from HW specs)
    - Which FA4 optimization is designed to raise each ceiling
    - The theoretical maximum if that ceiling were removed

    We do NOT show measured performance — use benchmark_ablation_sm100.py for that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    M, N = 128, 128
    r = tile_cycles(B200, M, N, D)

    # Theoretical ceilings
    tc_ceil   = attainable_tflops(B200, r["tc"],   r["tc"])   # = peak
    exp_ceil  = attainable_tflops(B200, r["exp"],  r["tc"])   # limited by MUFU
    smem_ceil = attainable_tflops(B200, r["smem"], r["tc"])   # limited by SMEM

    # What each optimization changes (provable claims only):
    # ping-pong: removes pipeline serialization. Without it, MMA+softmax run serially.
    #   Serial cycles >= TC + EXP (must do both in sequence before next tile).
    #   With ping-pong: overlapped → limited by max(TC, EXP, SMEM).
    serial_ceil = attainable_tflops(B200, r["tc"] + r["exp"], r["tc"])

    # exp2 FMA emulation: upper bound if EXP bottleneck completely removed
    #   (i.e., FMA units provided unlimited exp throughput):
    exp_removed_ceil = attainable_tflops(B200, max(r["tc"], r["smem"]), r["tc"])

    # 2-CTA MMA: halves K SMEM reads (each CTA reads only its half)
    # Provable: qk_smem is halved → new smem_cycles = pv_smem + qk_smem/2
    pv_bytes = math.ceil(M/128) * math.ceil(D/128) * N * 128 * 2
    qk_bytes = math.ceil(M/128) * math.ceil(N/128) * (128+128) * D * 2
    smem_2cta_cyc = (pv_bytes + qk_bytes * 0.5) / B200.smem
    smem_2cta_ceil = attainable_tflops(B200, smem_2cta_cyc, r["tc"])

    fig, ax = plt.subplots(figsize=(13, 7))

    # Draw the ceiling lines
    ceilings = [
        (serial_ceil,      "#c0392b", "--", 2.2,
         f"Serial pipeline ceiling  = {serial_ceil:.0f} T  "
         f"[TC({r['tc']:.0f}c) + EXP({r['exp']:.0f}c) = {r['tc']+r['exp']:.0f}c serial]"),
        (exp_ceil,         "#e67e22", "-.", 2.0,
         f"EXP (MUFU) ceiling       = {exp_ceil:.0f} T  "
         f"[{r['exp']:.0f} cycles for {M*N} exp2 at {B200.mufu} ops/c]"),
        (smem_ceil,        "#2980b9", ":",  2.0,
         f"SMEM ceiling             = {smem_ceil:.0f} T  "
         f"[{r['smem']:.0f} cycles at {B200.smem} B/c]"),
        (exp_removed_ceil, "#27ae60", "-.", 1.5,
         f"EXP ceiling if fully removed = {exp_removed_ceil:.0f} T  "
         f"[theoretical upper bound for exp2 optimization]"),
        (smem_2cta_ceil,   "#9b59b6", ":",  1.5,
         f"SMEM ceiling with 2-CTA  = {smem_2cta_ceil:.0f} T  "
         f"[K reads halved: {smem_2cta_cyc:.0f}c]"),
        (tc_ceil,          "#e74c3c", "--", 1.5,
         f"TC ceiling (peak)        = {tc_ceil:.0f} T  "
         f"[= B200 BF16 theoretical peak]"),
    ]

    for val, color, ls, lw, label in ceilings:
        ax.axhline(val, color=color, ls=ls, lw=lw, label=label, alpha=0.85)

    # Add optimization arrows showing which ceiling each optimization targets
    arrows = [
        # (from_y, to_y, x_pos, label, color)
        (serial_ceil, exp_ceil,        1.5, "ping-pong\n(q_stage=2)\nremoves\nserial stall", "#e67e22"),
        (exp_ceil,    exp_removed_ceil, 3.5, "exp2 FMA emu\nraises EXP ceiling\n(theoretical max)", "#27ae60"),
        (smem_ceil,   smem_2cta_ceil,  5.5, "2-CTA MMA\nhalves K reads\n(non-causal)", "#9b59b6"),
    ]

    for (y0, y1, xp, label, color) in arrows:
        ax.annotate("", xy=(xp, y1 - 20), xytext=(xp, y0 + 20),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
        mid = (y0 + y1) / 2
        ax.text(xp + 0.1, mid, label, ha="left", va="center",
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

    # Shade the "gap" regions
    ax.axhspan(0, serial_ceil, alpha=0.04, color="#c0392b")
    ax.axhspan(serial_ceil, exp_ceil, alpha=0.04, color="#e67e22")
    ax.axhspan(exp_ceil, tc_ceil, alpha=0.04, color="#27ae60")

    ax.set_xlim(0, 7)
    ax.set_ylim(0, tc_ceil * 1.15)
    ax.set_xticks([])
    ax.set_ylabel("Attainable TFLOPs/s", fontsize=11)
    ax.set_title(
        f"FA4 Optimization → Hardware Ceiling Mapping  (B200, D={D}, M=N={M})\n"
        "ALL numbers derived from hardware specs. No measured performance.",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Summary table
    table_lines = [
        f"Provable ceiling values for B200, D={D}, M=N={M}:",
        f"  TC cycles:           {r['tc']:.0f} c   (4×{M}×{N}×{D} / {B200.tc} ops/c)",
        f"  EXP cycles:          {r['exp']:.0f} c   ({M}×{N} / {B200.mufu} ops/c)",
        f"  SMEM cycles:         {r['smem']:.0f} c   ({int(math.ceil(M/128)*math.ceil(N/128)*256*D*2 + math.ceil(M/128)*math.ceil(D/128)*128*N*2)} B / {B200.smem} B/c)",
        f"  Serial baseline:     ≤ {serial_ceil:.0f} T  (TC+EXP serially)",
        f"  MUFU ceiling:        ≤ {exp_ceil:.0f} T  (EXP limit)",
        f"  SMEM ceiling:        ≤ {smem_ceil:.0f} T  (SMEM limit)",
        f"  EXP fully removed:   ≤ {exp_removed_ceil:.0f} T  (theoretical)",
        f"  SMEM with 2-CTA:     ≤ {smem_2cta_ceil:.0f} T  (K halved)",
        f"  TC peak:             = {tc_ceil:.0f} T  (B200 peak)",
    ]
    ax.text(0.01, 0.02, "\n".join(table_lines),
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#ccc", alpha=0.95))

    plt.tight_layout()
    p = os.path.join(out_dir, f"03_ceiling_map_D{D}.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  {p}")
    return p


# ── Print theory table to stdout ──────────────────────────────────────────────

def print_theory_table(hdims: List[int] = None):
    if hdims is None:
        hdims = [64, 96, 128, 192, 256]

    tile_map = {64: (192,128), 96: (192,128), 128: (128,128), 192: (128,128), 256: (128,128)}

    print("\n" + "═"*90)
    print("  FA4 Theoretical Ceilings — B200 SXM (all derived from hardware specs, no guessing)")
    print("═"*90)
    print(f"  {'D':>4}  {'M×N':>8}  {'TC_c':>6}  {'EXP_c':>6}  {'SMEM_c':>7}  "
          f"{'TC_ceil_T':>10}  {'EXP_ceil_T':>11}  {'SMEM_ceil_T':>12}  {'Serial_ceil_T':>14}  {'Bottleneck'}")
    print("  " + "─"*86)
    for D in hdims:
        M, N = tile_map.get(D, (128, 128))
        r = tile_cycles(B200, M, N, D)
        tc_c   = attainable_tflops(B200, r["tc"],        r["tc"])
        exp_c  = attainable_tflops(B200, r["exp"],       r["tc"])
        smem_c = attainable_tflops(B200, r["smem"],      r["tc"])
        ser_c  = attainable_tflops(B200, r["tc"]+r["exp"], r["tc"])
        print(f"  {D:>4}  {M}×{N}  {r['tc']:>6.0f}  {r['exp']:>6.0f}  {r['smem']:>7.0f}  "
              f"{tc_c:>10.0f}  {exp_c:>11.0f}  {smem_c:>12.0f}  {ser_c:>14.0f}  {r['bottleneck_name']}")
    print()
    print("  Legend:")
    print("    TC_c / EXP_c / SMEM_c : clock cycles per tile for that resource")
    print("    *_ceil_T               : attainable TFLOPs/s if only that resource limited")
    print("    Serial_ceil_T          : attainable if TC+EXP run serially (no ping-pong)")
    print("    Bottleneck             : which resource has the most cycles (worst case)")
    print()
    print("  Key numbers for D=128 (M=N=128) on B200:")
    r = tile_cycles(B200, 128, 128, 128)
    print(f"    TC  cycles = {r['tc']:.0f}c  (4×128×128×128 / 8192 ops/c/SM)")
    print(f"    EXP cycles = {r['exp']:.0f}c  (128×128 / 16 ops/c/SM)")
    print(f"    → TC:EXP = {r['tc']/r['exp']:.2f}× — EXACT co-bottleneck on B200")
    print(f"    → Serial pipeline wastes {(r['tc']+r['exp']-r['tc'])/(r['tc']+r['exp'])*100:.0f}% of TC capacity")
    print()
    print("  Compare with H100 for D=128 (M=N=128):")
    rh = tile_cycles(H100, 128, 128, 128)
    print(f"    TC  cycles = {rh['tc']:.0f}c  (4×128×128×128 / 4096 ops/c/SM)")
    print(f"    EXP cycles = {rh['exp']:.0f}c  (same MUFU)")
    print(f"    → TC:EXP = {rh['tc']/rh['exp']:.2f}× — EXP is 2× faster than TC on H100 → not a bottleneck")
    print("═"*90)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FA4 Roofline Theory (no speculation, hardware specs only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir", default="./plots")
    parser.add_argument("--hdim", type=lambda s: [int(x) for x in s.split(",")],
                        default=[64, 96, 128, 192, 256])
    parser.add_argument("--no-plots", action="store_true",
                        help="Print table only, skip matplotlib")
    args = parser.parse_args()

    print_theory_table(args.hdim)

    if not args.no_plots:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Generating theory charts → {args.out_dir}/")
        plot_asymmetry(args.out_dir)
        plot_hdim_ceilings(args.out_dir, args.hdim)
        for D in [d for d in args.hdim if d in [64, 96, 128, 192]]:
            plot_optimization_ceiling_map(args.out_dir, D=D)
        print("Done.")


if __name__ == "__main__":
    main()
