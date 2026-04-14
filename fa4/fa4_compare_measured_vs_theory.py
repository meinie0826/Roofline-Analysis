#!/usr/bin/env python
"""
FA4 Measured vs Theory Comparison
===================================

Reads the CSV output from benchmark_ablation_sm100.py and overlays measured
TFLOPs/s against the theoretical hardware ceilings from fa4_roofline_theory.py.

No numbers are hardcoded. Everything comes either from:
  (a) hardware specs (theory ceilings)
  (b) your benchmark_ablation_sm100.py CSV (measured TFLOPs/s)

Usage:
    python fa4/fa4_compare_measured_vs_theory.py \\
        --csv fa4_results/ablation_noncausal_D128.csv \\
        --out-dir fa4_results

    python fa4/fa4_compare_measured_vs_theory.py \\
        --csv fa4_results/ablation_causal_D128.csv \\
        --causal \\
        --out-dir fa4_results

Optional (with NSight CSV):
    python fa4/fa4_compare_measured_vs_theory.py \\
        --csv fa4_results/ablation_noncausal_D128.csv \\
        --ncu fa4_results/ncu_counters.csv \\
        --out-dir fa4_results
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from fa4_roofline_theory import B200, H100, tile_cycles, attainable_tflops


# ── Load benchmark CSV ────────────────────────────────────────────────────────

def load_benchmark_csv(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "stage_idx":   int(row["stage_idx"]),
                    "stage_label": row["stage_label"],
                    "seqlen":      int(row["seqlen"]),
                    "hdim":        int(row["hdim"]),
                    "causal":      row["causal"].strip().lower() in ("true", "1"),
                    "tflops":      float(row["tflops"]) if row["tflops"] and row["tflops"] != "None" else None,
                    "ms":          float(row["ms"])     if row["ms"]     and row["ms"]     != "None" else None,
                    "max_diff":    float(row["max_diff"]) if row["max_diff"] and row["max_diff"] != "None" else None,
                    "error":       row.get("error", ""),
                })
            except (ValueError, KeyError):
                continue
    return rows


# ── Plot: measured bars + theory ceiling lines ────────────────────────────────

def plot_measured_vs_theory(rows: List[Dict], out_dir: str, causal: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by seqlen
    seqlens = sorted(set(r["seqlen"] for r in rows))
    stage_labels = {r["stage_idx"]: r["stage_label"] for r in rows}
    n_stages = max(stage_labels) + 1

    # Infer hdim (should be uniform)
    hdims = set(r["hdim"] for r in rows)
    if len(hdims) != 1:
        print(f"WARNING: multiple hdim values in CSV: {hdims}, using first.")
    D = hdims.pop()

    # Compute theory ceilings
    tile_map = {64: (192,128), 96: (192,128), 128: (128,128), 192: (128,128), 256: (128,128)}
    M, N = tile_map.get(D, (128, 128))
    r_hw = tile_cycles(B200, M, N, D)
    tc_ceil   = attainable_tflops(B200, r_hw["tc"],            r_hw["tc"])
    exp_ceil  = attainable_tflops(B200, r_hw["exp"],           r_hw["tc"])
    smem_ceil = attainable_tflops(B200, r_hw["smem"],          r_hw["tc"])
    ser_ceil  = attainable_tflops(B200, r_hw["tc"]+r_hw["exp"], r_hw["tc"])

    pv_bytes = math.ceil(M/128)*math.ceil(D/128)*N*128*2
    qk_bytes = math.ceil(M/128)*math.ceil(N/128)*(128+128)*D*2
    smem_2cta_cyc = (pv_bytes + qk_bytes*0.5) / B200.smem
    smem_2cta_ceil = attainable_tflops(B200, smem_2cta_cyc, r_hw["tc"])

    # One subplot per seqlen
    ncols = min(len(seqlens), 3)
    nrows = math.ceil(len(seqlens) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5.5*nrows))
    if len(seqlens) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    c_str = "causal" if causal else "non-causal"
    fig.suptitle(
        f"FA4 Ablation: Measured TFLOPs/s vs Theory Ceilings\n"
        f"B200, hdim={D}, {c_str} | Measured from benchmark_ablation_sm100.py",
        fontsize=12, fontweight="bold",
    )

    stage_colors = plt.cm.tab10(np.linspace(0, 0.7, n_stages))

    for ax, seqlen in zip(axes_flat, seqlens):
        stage_data = {}
        for r in rows:
            if r["seqlen"] == seqlen:
                stage_data[r["stage_idx"]] = r

        x = np.arange(n_stages)
        measured = [stage_data.get(i, {}).get("tflops") or 0 for i in range(n_stages)]
        bars = ax.bar(x, measured, 0.6, color=stage_colors, edgecolor="white", zorder=3)

        # Theory ceilings
        ax.axhline(ser_ceil,  color="#c0392b", ls="--",  lw=1.8, alpha=0.85,
                   label=f"Serial ceil {ser_ceil:.0f}T")
        ax.axhline(exp_ceil,  color="#e67e22", ls="-.",  lw=1.8, alpha=0.85,
                   label=f"MUFU ceil {exp_ceil:.0f}T")
        ax.axhline(smem_ceil, color="#2980b9", ls=":",   lw=1.8, alpha=0.85,
                   label=f"SMEM ceil {smem_ceil:.0f}T")
        ax.axhline(tc_ceil,   color="#e74c3c", ls="--",  lw=1.2, alpha=0.5,
                   label=f"TC peak {tc_ceil:.0f}T")

        # Value labels
        for bar, v in zip(bars, measured):
            if v > 0:
                pct = v / tc_ceil * 100
                ax.text(bar.get_x()+bar.get_width()/2, v+10,
                        f"{v:.0f}T\n({pct:.0f}%)",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Speedup vs stage 0
        base = measured[0]
        if base > 0:
            for i in range(1, n_stages):
                if measured[i] > 0:
                    ax.text(x[i], -120, f"{measured[i]/base:.2f}×\nvs S0",
                            ha="center", va="top", fontsize=7, color="#444", style="italic")

        # x-axis: stage labels
        ax.set_xticks(x)
        ax.set_xticklabels([stage_labels.get(i, f"S{i}") for i in range(n_stages)],
                           fontsize=7.5, rotation=15, ha="right")
        ax.set_ylabel("TFLOPs/s", fontsize=9)
        ax.set_title(f"seqlen={seqlen:,}", fontsize=10)
        ax.set_ylim(0, tc_ceil * 1.2)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax in axes_flat[len(seqlens):]:
        ax.set_visible(False)

    plt.tight_layout()
    fname = f"measured_vs_theory_D{D}_{'causal' if causal else 'noncausal'}.png"
    p = os.path.join(out_dir, fname)
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  {p}")
    return p


# ── Print summary table ────────────────────────────────────────────────────────

def print_summary(rows: List[Dict]):
    seqlens = sorted(set(r["seqlen"] for r in rows))
    n_stages = max(r["stage_idx"] for r in rows) + 1
    stage_labels = {r["stage_idx"]: r["stage_label"] for r in rows}

    hdims = set(r["hdim"] for r in rows)
    D = next(iter(hdims))
    tile_map = {64: (192,128), 96: (192,128), 128: (128,128), 192: (128,128), 256: (128,128)}
    M, N = tile_map.get(D, (128, 128))
    r_hw = tile_cycles(B200, M, N, D)
    ser_ceil  = attainable_tflops(B200, r_hw["tc"]+r_hw["exp"], r_hw["tc"])
    exp_ceil  = attainable_tflops(B200, r_hw["exp"],            r_hw["tc"])
    tc_ceil   = attainable_tflops(B200, r_hw["tc"],             r_hw["tc"])

    W = 14
    print("\n" + "═"*80)
    print(f"  Measured vs Theory  (D={D}, M=N={M}, B200)")
    print(f"  Theory ceilings: serial={ser_ceil:.0f}T  MUFU={exp_ceil:.0f}T  TC={tc_ceil:.0f}T")
    print("═"*80)
    hdr = f"  {'Stage':<28}" + "".join(f" {'s='+str(s):>{W}}" for s in seqlens)
    print(hdr)
    print("  " + "─"*len(hdr))

    for i in range(n_stages):
        label = stage_labels.get(i, f"S{i}")
        row_str = f"  {label:<28}"
        for seqlen in seqlens:
            r = next((x for x in rows if x["stage_idx"]==i and x["seqlen"]==seqlen), None)
            if r and r["tflops"]:
                pct = r["tflops"] / tc_ceil * 100
                cell = f"{r['tflops']:.0f}T({pct:.0f}%)"
            else:
                cell = "N/A"
            row_str += f" {cell:>{W}}"
        print(row_str)

        if i > 0:
            spd_row = f"  {'  speedup vs S0':<28}"
            for seqlen in seqlens:
                base = next((x for x in rows if x["stage_idx"]==0 and x["seqlen"]==seqlen), None)
                cur  = next((x for x in rows if x["stage_idx"]==i and x["seqlen"]==seqlen), None)
                if base and cur and base["tflops"] and cur["tflops"]:
                    spd_row += f" {f'{cur[\"tflops\"]/base[\"tflops\"]:.2f}×':>{W}}"
                else:
                    spd_row += f" {'N/A':>{W}}"
            print(spd_row)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare measured TFLOPs/s against theoretical ceilings",
    )
    parser.add_argument("--csv", required=True, help="CSV from benchmark_ablation_sm100.py")
    parser.add_argument("--ncu", default=None, help="NSight CSV (optional)")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    rows = load_benchmark_csv(args.csv)
    if not rows:
        print(f"ERROR: no rows loaded from {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print_summary(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_measured_vs_theory(rows, args.out_dir, causal=args.causal)

    if args.ncu:
        print(f"\nNSight CSV: {args.ncu}")
        print("(NSight parsing not yet implemented — open the CSV in NSight Compute UI)")

    print("Done.")


if __name__ == "__main__":
    main()
