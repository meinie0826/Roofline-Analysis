#!/usr/bin/env python
"""
FA4 (SM100/Blackwell) Ablation Study: Decomposing per-optimization performance contributions.

Ablation ladder (forward pass, B200, BF16, hdim=128):
  Stage 0: Baseline  — q_stage=1 (no ping-pong), always rescale, all-MUFU exp, linear sched
  Stage 1: +pingpong — q_stage=2 (Blackwell async TMEM pipeline)
  Stage 2: +condrescale — conditional rescaling (threshold=8.0, skip small-jump rescales)
  Stage 3: +ex2emu  — FMA software exp2 emulation mixed with MUFU
  Stage 4: +LPT     — LPT tile scheduler (meaningful only for causal)

For each stage × seqlen, we measure TFLOPs/s and TC utilization %.
We also print the multi-resource roofline theory (TC/EXP/SMEM cycles).

Usage:
    # Dry-run: check roofline theory without benchmarking
    python benchmarks/benchmark_ablation_sm100.py --roofline-only

    # Full ablation (default: non-causal + causal, hdim=128, seqlen=1024..8192)
    python benchmarks/benchmark_ablation_sm100.py

    # Causal only, more seqlens
    python benchmarks/benchmark_ablation_sm100.py --seqlen 2048,4096,8192,16384 --causal-only

    # Save CSV
    python benchmarks/benchmark_ablation_sm100.py --csv ablation.csv

    # Plot roofline ladder (requires matplotlib)
    python benchmarks/benchmark_ablation_sm100.py --plot
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F

# ─── B200 Hardware Constants ──────────────────────────────────────────────────
B200_NUM_SMS           = 148
B200_TC_OPS_PER_CYCLE  = 8192   # BF16 MMA ops/cycle/SM
B200_MUFU_OPS_PER_CYCLE = 16    # exp2 ops/cycle/SM
B200_SMEM_BW_PER_CYCLE = 128    # bytes/cycle/SM (shared mem read bandwidth)
B200_PEAK_TFLOPS_BF16  = 2250.0 # theoretical peak BF16 TFLOPs (full chip)


# ─── Roofline math ────────────────────────────────────────────────────────────

def roofline_fwd(M=128, N=128, D=128) -> Dict[str, float]:
    """Compute per-tile clock cycle estimates for each bottleneck resource."""
    # Two MMAs per tile: QK^T (M×D×N) and PV (M×N×D), each = 2*M*N*D FLOPs
    tc_cycles   = 4.0 * M * N * D / B200_TC_OPS_PER_CYCLE
    # Softmax: M*N exp2 operations
    exp_cycles  = float(M * N) / B200_MUFU_OPS_PER_CYCLE
    # SMEM reads (BF16 = 2 bytes/elem):
    #   QK^T: each sub-tile reads 128×D Q and D×128 K from SMEM
    qk_smem = math.ceil(M/128) * math.ceil(N/128) * 256 * D * 2
    #   PV: each sub-tile reads N×128 V from SMEM
    pv_smem = math.ceil(M/128) * math.ceil(D/128) * 128 * N * 2
    smem_cycles = (qk_smem + pv_smem) / B200_SMEM_BW_PER_CYCLE
    all_cyc = [tc_cycles, exp_cycles, smem_cycles]
    bottleneck = max(all_cyc)
    bn_name = ["TC", "EXP", "SMEM"][all_cyc.index(bottleneck)]
    return dict(
        tc=tc_cycles, exp=exp_cycles, smem=smem_cycles,
        bottleneck=bottleneck, bottleneck_name=bn_name,
        tc_exp_ratio=tc_cycles / exp_cycles,
        tc_smem_ratio=tc_cycles / smem_cycles,
    )


def print_roofline_table(D=128, M=128, N=128):
    r = roofline_fwd(M, N, D)
    print(f"\n{'─'*66}")
    print(f"  Multi-Resource Roofline  (tile M={M}×N={N}×D={D}, B200 per SM)")
    print(f"{'─'*66}")
    print(f"  TC  (BF16 MMA):       {r['tc']:6.0f} cyc  (8192 ops/cycle, 2 MMAs)")
    print(f"  EXP (MUFU.EX2):       {r['exp']:6.0f} cyc  (16 ops/cycle, M×N exps)")
    print(f"  SMEM read bandwidth:  {r['smem']:6.0f} cyc  (128 bytes/cycle)")
    print(f"  Primary bottleneck:   {r['bottleneck_name']} ({r['bottleneck']:.0f} cyc)")
    print(f"  TC:EXP  = {r['tc_exp_ratio']:.2f}x  → {'EXP co-bottleneck' if r['tc_exp_ratio'] > 1 else 'TC bound'}")
    print(f"  TC:SMEM = {r['tc_smem_ratio']:.2f}x  → {'SMEM co-bottleneck' if r['tc_smem_ratio'] > 1 else 'TC bound'}")
    print(f"{'─'*66}")
    print()


def fwd_flops(batch, nheads, seqlen, hdim, hdim_v=None, causal=False):
    if hdim_v is None:
        hdim_v = hdim
    avg_seqlen = seqlen / 2 if causal else seqlen
    return batch * nheads * 2 * seqlen * avg_seqlen * (hdim + hdim_v)


# ─── Ablation Stage Definitions ───────────────────────────────────────────────

@dataclass
class AblationStage:
    label: str          # filesystem-friendly id
    description: str    # human description of what changed
    # These map directly to _ablation_* kwargs in _flash_attn_fwd
    q_stage: int        # 1=no ping-pong, 2=ping-pong
    ex2_emu_freq: int   # 0=MUFU-only, >0=mixed FMA+MUFU (default tuning if -1)
    rescale_threshold: float   # 0.0=always rescale, 8.0=conditional
    no_lpt: bool        # True=disable LPT scheduler, False=use default


ABLATION_STAGES = [
    AblationStage(
        label="0-Baseline",
        description="q_stage=1, always rescale, MUFU-only exp, linear scheduler",
        q_stage=1, ex2_emu_freq=0, rescale_threshold=0.0, no_lpt=True,
    ),
    AblationStage(
        label="1+PingPong",
        description="+ping-pong (q_stage=2): async TMEM MMA↔softmax overlap",
        q_stage=2, ex2_emu_freq=0, rescale_threshold=0.0, no_lpt=True,
    ),
    AblationStage(
        label="2+CondRescale",
        description="+conditional rescaling (threshold=8.0): skip ~10x rescale work",
        q_stage=2, ex2_emu_freq=0, rescale_threshold=8.0, no_lpt=True,
    ),
    AblationStage(
        label="3+Ex2Emu",
        description="+FMA exp2 emulation: ~2x effective exp throughput via polynomial",
        q_stage=2, ex2_emu_freq=-1, rescale_threshold=8.0, no_lpt=True,
    ),
    AblationStage(
        label="4+LPT",
        description="+LPT scheduler: load-balance SMs (causal: +4-14%, non-causal: ~0%)",
        q_stage=2, ex2_emu_freq=-1, rescale_threshold=8.0, no_lpt=False,
    ),
]

STAGE_SHORT_LABELS = [
    "Baseline\n(no Blackwell opts)",
    "+ping-pong\n(q_stage=2)",
    "+cond-rescale\n(thresh=8)",
    "+exp2 emu\n(FMA)",
    "+LPT sched\n(causal↑)",
]


# ─── Benchmark function ───────────────────────────────────────────────────────

def bench_stage(
    stage: AblationStage,
    batch: int,
    seqlen: int,
    nheads: int,
    hdim: int,
    hdim_v: int,
    causal: bool,
    warmup: int = 5,
    rep: int = 30,
    check_correctness: bool = True,
) -> Dict[str, Any]:
    """Benchmark one ablation stage using _flash_attn_fwd with ablation kwargs."""
    result = dict(
        stage_label=stage.label,
        description=stage.description,
        batch=batch, seqlen=seqlen, nheads=nheads, hdim=hdim, hdim_v=hdim_v, causal=causal,
        ms=None, tflops=None, tc_util_pct=None, max_diff=None, error=None,
    )

    try:
        from flash_attn.cute.interface import _flash_attn_fwd

        softmax_scale = hdim ** -0.5
        q = torch.randn(batch, seqlen, nheads, hdim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, nheads, hdim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, nheads, hdim_v, dtype=torch.bfloat16, device="cuda")

        # Build ablation kwargs
        kwargs = dict(
            softmax_scale=softmax_scale,
            causal=causal,
            _ablation_q_stage=stage.q_stage,
            _ablation_no_lpt=stage.no_lpt,
            _ablation_rescale_threshold=stage.rescale_threshold,
        )
        # ex2_emu_freq: -1 means "use tuning config default" → pass None to let FA4 decide
        if stage.ex2_emu_freq >= 0:
            kwargs["_ablation_ex2_emu_freq"] = stage.ex2_emu_freq
        else:
            kwargs["_ablation_ex2_emu_freq"] = None  # use tuning config

        # Compile (first call triggers JIT)
        out, _ = _flash_attn_fwd(q, k, v, **kwargs)

        # Correctness check vs PyTorch reference
        if check_correctness:
            ref = F.scaled_dot_product_attention(
                q.transpose(1, 2).float(),
                k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
                scale=softmax_scale,
                is_causal=causal,
            ).transpose(1, 2).to(torch.bfloat16)
            result["max_diff"] = (out.float() - ref.float()).abs().max().item()

        # Warmup
        for _ in range(warmup):
            _flash_attn_fwd(q, k, v, **kwargs)
        torch.cuda.synchronize()

        # Timed runs
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(rep):
            _flash_attn_fwd(q, k, v, **kwargs)
        t1.record()
        torch.cuda.synchronize()

        ms = t0.elapsed_time(t1) / rep
        flops = fwd_flops(batch, nheads, seqlen, hdim, hdim_v, causal)
        tflops = flops / ms / 1e9
        result.update(ms=ms, tflops=tflops, tc_util_pct=tflops / B200_PEAK_TFLOPS_BF16 * 100)

    except Exception as e:
        import traceback
        result["error"] = traceback.format_exc()[-400:]

    return result


# ─── Run ablation + reporting ─────────────────────────────────────────────────

def run_ablation(
    seqlens: List[int],
    hdim: int = 128,
    hdim_v: Optional[int] = None,
    nheads: int = 16,
    causal: bool = False,
    warmup: int = 5,
    rep: int = 30,
    check_correctness: bool = True,
) -> List[Dict[str, Any]]:
    if hdim_v is None:
        hdim_v = hdim
    total_tokens = 32768

    print(f"\n{'═'*80}")
    print(f"  FA4 SM100 Ablation │ hdim={hdim} hdim_v={hdim_v} causal={causal} nheads={nheads}")
    print(f"{'═'*80}")
    print_roofline_table(D=hdim)

    W = 14
    header = f"  {'Stage':<38}" + "".join(f" {'s='+str(s):>{W}}" for s in seqlens)
    print(header)
    print("  " + "─" * len(header))

    all_results = []

    for i, stage in enumerate(ABLATION_STAGES):
        print(f"\n  [{stage.label}] {stage.description}")

        row = f"  {STAGE_SHORT_LABELS[i].replace(chr(10),' '):<38}"
        stage_results = []

        for seqlen in seqlens:
            batch = max(1, total_tokens // seqlen)
            r = bench_stage(
                stage, batch, seqlen, nheads, hdim, hdim_v, causal,
                warmup=warmup, rep=rep, check_correctness=check_correctness,
            )
            r["stage_idx"] = i
            all_results.append(r)
            stage_results.append(r)

            if r["error"]:
                cell = "ERR"
                # Print first 120 chars of error
                print(f"    ERROR at seqlen={seqlen}: {r['error'][-120:]}", file=sys.stderr)
            elif r["tflops"] is not None:
                diff_str = f" Δ={r['max_diff']:.5f}" if r["max_diff"] is not None else ""
                cell = f"{r['tflops']:.0f}T({r['tc_util_pct']:.0f}%)"
                if r["max_diff"] is not None and r["max_diff"] > 0.01:
                    cell += "!"  # flag suspicious diff
            else:
                cell = "NONE"
            row += f" {cell:>{W}}"

        print(row)

        # Speedup vs stage 0
        if i > 0:
            base_results = [r for r in all_results if r["stage_idx"] == 0]
            spd_row = f"  {'  speedup vs baseline':<38}"
            for j, seqlen in enumerate(seqlens):
                cur = stage_results[j]
                base = next((r for r in base_results if r["seqlen"] == seqlen), None)
                if base and cur["tflops"] and base["tflops"]:
                    spd_row += f" {f'{cur[\"tflops\"]/base[\"tflops\"]:.2f}x':>{W}}"
                else:
                    spd_row += f" {'N/A':>{W}}"
            print(spd_row)

    return all_results


def print_summary(all_results: List[Dict[str, Any]], seqlens: List[int]):
    print(f"\n{'═'*80}")
    print("  Per-Optimization Marginal Contribution")
    print(f"{'═'*80}")
    W = 16
    header = f"  {'Optimization':<42}" + "".join(f" {'s='+str(s):>{W}}" for s in seqlens)
    print(header)
    print("  " + "─" * len(header))

    opt_names = [
        "Baseline (all opts disabled)",
        "→ +ping-pong (q_stage=2)",
        "→ +conditional rescaling",
        "→ +FMA exp2 emulation",
        "→ +LPT scheduler",
    ]

    for i, name in enumerate(opt_names):
        row = f"  {name:<42}"
        for seqlen in seqlens:
            cur = next((r for r in all_results if r["stage_idx"]==i and r["seqlen"]==seqlen), None)
            if not cur or cur["error"] or not cur["tflops"]:
                row += f" {'FAIL':>{W}}"
                continue
            if i == 0:
                row += f" {cur['tflops']:>{W-1}.1f}T"
            else:
                prev = next((r for r in all_results if r["stage_idx"]==i-1 and r["seqlen"]==seqlen), None)
                if prev and prev["tflops"]:
                    delta = cur["tflops"] - prev["tflops"]
                    pct = delta / prev["tflops"] * 100
                    sign = "+" if delta >= 0 else ""
                    row += f" {f'{sign}{delta:.0f}T({sign}{pct:.0f}%)':>{W}}"
                else:
                    row += f" {cur['tflops']:>{W-1}.1f}T"
        print(row)

    print("  " + "─" * len(header))
    n_last = len(ABLATION_STAGES) - 1
    total_row = f"  {'Full FA4 vs Baseline (total speedup)':<42}"
    for seqlen in seqlens:
        base = next((r for r in all_results if r["stage_idx"]==0 and r["seqlen"]==seqlen), None)
        full = next((r for r in all_results if r["stage_idx"]==n_last and r["seqlen"]==seqlen), None)
        if base and full and base["tflops"] and full["tflops"]:
            total_row += f" {f'{full[\"tflops\"]/base[\"tflops\"]:.2f}x':>{W}}"
        else:
            total_row += f" {'N/A':>{W}}"
    print(total_row)
    print()


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_csv(results: List[Dict], path: str):
    import csv
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["error"] = str(row.get("error") or "")[:100]
            writer.writerow(row)
    print(f"CSV saved: {path}")


# ─── Matplotlib roofline plot ─────────────────────────────────────────────────

def plot_ladder(results: List[Dict], seqlens: List[int], hdim: int, causal: bool, save_dir="."):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    rf = roofline_fwd(128, 128, hdim)
    # Attainable TFLOPs/s for each resource ceiling:
    #   peak_TC  = B200_PEAK_TFLOPS_BF16
    #   attain_X = peak_TC * (TC_cycles / X_cycles)
    tc_roof   = B200_PEAK_TFLOPS_BF16
    exp_roof  = B200_PEAK_TFLOPS_BF16 * rf["tc"] / rf["exp"]
    smem_roof = B200_PEAK_TFLOPS_BF16 * rf["tc"] / rf["smem"]

    for seqlen in seqlens:
        stage_vals = []
        for i in range(len(ABLATION_STAGES)):
            r = next((x for x in results if x["stage_idx"]==i and x["seqlen"]==seqlen), None)
            stage_vals.append(r["tflops"] if r and r["tflops"] else 0.0)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(ABLATION_STAGES))
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(ABLATION_STAGES)))
        bars = ax.bar(x, stage_vals, 0.55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

        ax.axhline(tc_roof,   color="#e74c3c", ls="--", lw=1.8,
                   label=f"TC roofline ({tc_roof:.0f} TFLOPs/s)", zorder=4)
        ax.axhline(exp_roof,  color="#e67e22", ls="-.", lw=1.8,
                   label=f"EXP roofline ({exp_roof:.0f} TFLOPs/s)", zorder=4)
        ax.axhline(smem_roof, color="#2980b9", ls=":",  lw=1.8,
                   label=f"SMEM roofline ({smem_roof:.0f} TFLOPs/s)", zorder=4)

        for bar, val in zip(bars, stage_vals):
            if val > 0:
                util = val / B200_PEAK_TFLOPS_BF16 * 100
                ax.text(bar.get_x() + bar.get_width()/2, val + 15,
                        f"{val:.0f}\n({util:.0f}%)",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Speedup annotations below x-axis
        if stage_vals[0] > 0:
            for i in range(1, len(ABLATION_STAGES)):
                if stage_vals[i] > 0:
                    spd = stage_vals[i] / stage_vals[0]
                    ax.text(x[i], -80, f"{spd:.2f}x",
                            ha="center", va="top", fontsize=7.5, color="#555", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("\n", " ") for s in STAGE_SHORT_LABELS], fontsize=8.5)
        ax.set_ylabel("TFLOPs/s", fontsize=11)
        ax.set_title(
            f"FA4 Ablation (forward pass, SM100 B200)\n"
            f"hdim={hdim}, seqlen={seqlen:,}, causal={causal}, BF16",
            fontsize=12,
        )
        ymax = max(max(stage_vals), exp_roof, smem_roof) * 1.28 + 50
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Roofline theory inset
        ax.text(0.99, 0.03,
                f"Roofline (M=N=D={hdim}):\n"
                f"  TC:   {rf['tc']:.0f} cyc  (ratio 1.0x)\n"
                f"  EXP:  {rf['exp']:.0f} cyc  (TC÷EXP = {rf['tc_exp_ratio']:.1f}x → EXP bottleneck)\n"
                f"  SMEM: {rf['smem']:.0f} cyc  (TC÷SMEM = {rf['tc_smem_ratio']:.1f}x)\n"
                f"  Bottleneck: {rf['bottleneck_name']}",
                transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="#bbb", alpha=0.92))

        plt.tight_layout()
        fname = f"{save_dir}/fa4_ablation_fwd_hdim{hdim}_sl{seqlen}_causal{int(causal)}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {fname}")
        plt.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FA4 SM100 Ablation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seqlen", type=lambda s: [int(x) for x in s.split(",")],
                        default=[1024, 2048, 4096, 8192],
                        help="Comma-separated seqlens (default: 1024,2048,4096,8192)")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--nheads", type=int, default=None)
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--non-causal-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="Generate roofline plots (requires matplotlib)")
    parser.add_argument("--plot-dir", type=str, default=".", help="Directory for plot outputs")
    parser.add_argument("--roofline-only", action="store_true",
                        help="Print roofline theory and exit (no GPU needed)")
    args = parser.parse_args()

    if args.roofline_only:
        print_roofline_table(D=args.hdim)
        # Show expected bottleneck explanation
        rf = roofline_fwd(128, 128, args.hdim)
        print(f"Key insight: TC:EXP = {rf['tc_exp_ratio']:.1f}x")
        print("  → exp2 is a co-bottleneck with TC. Software FMA emulation can effectively")
        print("    double exp throughput by executing on FP32 units while MUFU handles other rows.")
        print(f"  → smem:TC = {1/rf['tc_smem_ratio']:.2f}x: SMEM is {rf['smem']:.0f}c vs TC {rf['tc']:.0f}c")
        print("  → 2-CTA MMA mode (use_2cta_instrs) can halve SMEM traffic (not ablated here)")
        return

    # Check device
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.", file=sys.stderr)
        sys.exit(1)
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 100:
        print(f"WARNING: Device is SM{sm}. FA4 SM100 kernels require B200/B100 (SM100+).",
              file=sys.stderr)
        print("  Results may fail or show incorrect performance.", file=sys.stderr)

    hdim = args.hdim
    nheads = args.nheads or (32 if hdim <= 64 else (16 if hdim <= 192 else 8))

    if args.causal_only:
        causals = [True]
    elif args.non_causal_only:
        causals = [False]
    else:
        causals = [False, True]

    all_results = []
    for causal in causals:
        results = run_ablation(
            seqlens=args.seqlen,
            hdim=hdim,
            nheads=nheads,
            causal=causal,
            warmup=args.warmup,
            rep=args.rep,
            check_correctness=not args.no_correctness,
        )
        all_results.extend(results)
        print_summary(results, args.seqlen)

        if args.plot:
            import os
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_ladder(results, args.seqlen, hdim, causal, save_dir=args.plot_dir)

    if args.csv:
        save_csv(all_results, args.csv)

    print("\n── NSight profiling command for per-stage metrics ──")
    print("  ncu --metrics sm__sass_inst_executed_op_mufu_ex2.sum,\\")
    print("          sm__sass_inst_executed_op_fadd.sum,\\")
    print("          sm__sass_inst_executed_op_ffma.sum,\\")
    print("          l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\\")
    print("          sm__cycles_elapsed.avg \\")
    print("      --target-processes all \\")
    print("      python benchmarks/benchmark_ablation_sm100.py --seqlen 4096 --no-correctness --rep 3")


if __name__ == "__main__":
    main()
