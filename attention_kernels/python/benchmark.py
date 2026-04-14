#!/usr/bin/env python3
"""
FlashAttention Kernel Benchmark Suite

Benchmarks all 5 stages of the kernel and generates:
- Performance comparison (TFLOPs/s)
- Roofline analysis plots
- CSV data for further analysis

Usage:
    python benchmark.py --seqlen 1024,2048,4096,8192 --csv results.csv
    python benchmark.py --stage 4 --seqlen 4096 --profile
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F

# Import our wrapper
try:
    from attention_wrapper import FlashAttentionKernel, create_test_inputs, AttentionConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from attention_wrapper import FlashAttentionKernel, create_test_inputs, AttentionConfig


# ─── Hardware Constants ─────────────────────────────────────────────────────

# A100 constants
A100_PEAK_TFLOPS_BF16 = 312.0
A100_PEAK_BANDWIDTH_GBPS = 2039.0
A100_SM_COUNT = 108

# B200 constants (if available)
B200_PEAK_TFLOPS_BF16 = 2250.0
B200_PEAK_BANDWIDTH_GBPS = 8000.0
B200_SM_COUNT = 148


def get_hardware_constants() -> Dict[str, float]:
    """Get hardware constants for current GPU."""
    if not torch.cuda.is_available():
        return {}
    
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    if sm >= 100:  # B200/B100
        return {
            "peak_tflops_bf16": B200_PEAK_TFLOPS_BF16,
            "peak_bandwidth_gbps": B200_PEAK_BANDWIDTH_GBPS,
            "sm_count": B200_SM_COUNT,
            "compute_capability": sm
        }
    elif sm >= 80:  # A100
        return {
            "peak_tflops_bf16": A100_PEAK_TFLOPS_BF16,
            "peak_bandwidth_gbps": A100_PEAK_BANDWIDTH_GBPS,
            "sm_count": A100_SM_COUNT,
            "compute_capability": sm
        }
    else:
        # Conservative defaults
        return {
            "peak_tflops_bf16": 100.0,
            "peak_bandwidth_gbps": 900.0,
            "sm_count": 84,
            "compute_capability": sm
        }


# ─── Benchmark Functions ───────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    stage: int
    stage_name: str
    batch_size: int
    seq_len: int
    n_heads: int
    head_dim: int
    causal: bool
    warmup_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    tflops: float
    bandwidth_gbps: float
    tc_utilization_pct: float
    max_diff: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benchmark_single(
    stage: int,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    causal: bool,
    warmup: int = 10,
    repeat: int = 50,
    check_correctness: bool = True
) -> BenchmarkResult:
    """Benchmark a single configuration."""
    
    stage_names = {
        0: "naive",
        1: "tiled",
        2: "shared_mem",
        3: "tensor_core",
        4: "final"
    }
    
    result = BenchmarkResult(
        stage=stage,
        stage_name=stage_names[stage],
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        causal=causal,
        warmup_ms=0,
        avg_ms=0,
        min_ms=0,
        max_ms=0,
        std_ms=0,
        tflops=0,
        bandwidth_gbps=0,
        tc_utilization_pct=0
    )
    
    try:
        # Create kernel and inputs
        kernel = FlashAttentionKernel(stage)
        q, k, v = create_test_inputs(batch_size, seq_len, n_heads, head_dim)
        
        # Compute reference
        if check_correctness:
            ref_out = FlashAttentionKernel.reference(q, k, v, causal)
        
        # Warmup
        for _ in range(warmup):
            _ = kernel(q, k, v, causal)
        torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = kernel(q, k, v, causal)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        import numpy as np
        times = np.array(times)
        
        result.warmup_ms = warmup * times.mean() if times.any() else 0
        result.avg_ms = float(times.mean())
        result.min_ms = float(times.min())
        result.max_ms = float(times.max())
        result.std_ms = float(times.std())
        
        # Compute metrics
        config = AttentionConfig(batch_size, seq_len, n_heads, head_dim, causal=causal)
        total_flops = config.total_flops()
        
        result.tflops = total_flops / result.avg_ms / 1e9
        
        # Memory bandwidth (approximate)
        # Q: batch * seq * n_heads * head_dim * 2 bytes
        # K, V: same
        # O: same
        total_bytes = 4 * batch_size * seq_len * n_heads * head_dim * 2
        result.bandwidth_gbps = total_bytes / result.avg_ms / 1e6
        
        # TC utilization
        hw = get_hardware_constants()
        peak_tflops = hw.get("peak_tflops_bf16", 100.0)
        result.tc_utilization_pct = result.tflops / peak_tflops * 100
        
        # Correctness check
        if check_correctness and out is not None:
            result.max_diff = (out.float() - ref_out.float()).abs().max().item()
        
    except Exception as e:
        import traceback
        result.error = str(e)[-200:]
        print(f"  ERROR: {result.error}", file=sys.stderr)
    
    return result


def run_full_benchmark(
    seqlens: List[int] = [1024, 2048, 4096, 8192],
    hdim: int = 128,
    nheads: int = 16,
    causal_modes: List[bool] = [False, True],
    stages: Optional[List[int]] = None,
    total_tokens: int = 32768,
    warmup: int = 10,
    repeat: int = 50,
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    
    if stages is None:
        stages = list(range(5))
    
    all_results = []
    
    hw = get_hardware_constants()
    print("\n" + "=" * 80)
    print("  FlashAttention Kernel Benchmark")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Peak BF16: {hw.get('peak_tflops_bf16', 0):.0f} TFLOPs/s")
    print(f"  Peak BW: {hw.get('peak_bandwidth_gbps', 0):.0f} GB/s")
    print("=" * 80)
    
    for causal in causal_modes:
        print(f"\n  Causal={causal}")
        print("  " + "-" * 76)
        
        header = f"  {'Stage':<18}" + "".join(f" {f's={s}':>12}" for s in seqlens)
        print(header)
        
        for stage in stages:
            stage_name = {
                0: "0-naive",
                1: "1-tiled",
                2: "2-smem",
                3: "3-tc",
                4: "4-final"
            }.get(stage, f"{stage}")
            
            row = f"  {stage_name:<18}"
            stage_results = []
            
            for seq_len in seqlens:
                batch = max(1, total_tokens // seq_len)
                
                result = benchmark_single(
                    stage, batch, seq_len, nheads, hdim, causal,
                    warmup=warmup, repeat=repeat
                )
                
                stage_results.append(result)
                all_results.append(result)
                
                if result.error:
                    cell = "ERR"
                elif result.tflops > 0:
                    cell = f"{result.tflops:.0f}T"
                else:
                    cell = "FAIL"
                row += f" {cell:>12}"
            
            print(row)
            
            # Speedup vs baseline
            if stage > 0 and not any(r.error for r in stage_results):
                baseline_results = [r for r in all_results 
                                   if r.stage == 0 and r.causal == causal]
                spd_row = f"  {'  vs baseline':<18}"
                for i, seq_len in enumerate(seqlens):
                    cur = stage_results[i]
                    base = next((r for r in baseline_results if r.seq_len == seq_len), None)
                    if base and cur.tflops and base.tflops and not cur.error:
                        spd = cur.tflops / base.tflops
                        spd_row += f" {f'{spd:.2f}x':>12}"
                    else:
                        spd_row += f" {'N/A':>12}"
                print(spd_row)
    
    return all_results


# ─── CSV Export ────────────────────────────────────────────────────────────

def save_csv(results: List[BenchmarkResult], path: str):
    """Save results to CSV file."""
    if not results:
        return
    
    keys = list(results[0].to_dict().keys())
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    
    print(f"CSV saved: {path}")


# ─── Roofline Plotting ──────────────────────────────────────────────────────

def plot_roofline(results: List[BenchmarkResult], save_dir: str = "."):
    """Generate roofline plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    Path(save_dir).mkdir(exist_ok=True)
    
    hw = get_hardware_constants()
    peak_tflops = hw.get("peak_tflops_bf16", 100.0)
    
    # Group by causal mode
    for causal in [False, True]:
        causal_results = [r for r in results if r.causal == causal and not r.error]
        
        if not causal_results:
            continue
        
        # Group by seq_len
        seqlens = sorted(set(r.seq_len for r in causal_results))
        
        for seq_len in seqlens:
            seq_results = [r for r in causal_results if r.seq_len == seq_len]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            stages = [r.stage for r in seq_results]
            tflops = [r.tflops for r in seq_results]
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
            bars = ax.bar(stages, tflops, color=colors, edgecolor="white")
            
            # Peak roofline
            ax.axhline(peak_tflops, color="red", linestyle="--", 
                      label=f"Peak TC ({peak_tflops:.0f} TFLOPs/s)")
            
            # Labels
            for bar, val, util in zip(bars, tflops, [r.tc_utilization_pct for r in seq_results]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                           f"{val:.0f}\n({util:.0f}%)",
                           ha="center", va="bottom", fontsize=8)
            
            ax.set_xlabel("Stage")
            ax.set_ylabel("TFLOPs/s")
            ax.set_title(f"FlashAttention Kernel Evolution\n"
                        f"seq_len={seq_len}, causal={causal}, BF16")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            fname = f"{save_dir}/roofline_sl{seq_len}_causal{int(causal)}.png"
            plt.savefig(fname, dpi=150)
            print(f"  Plot saved: {fname}")
            plt.close()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FlashAttention Kernel Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seqlen", type=lambda s: [int(x) for x in s.split(",")],
                        default=[1024, 2048, 4096, 8192],
                        help="Comma-separated sequence lengths")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--nheads", type=int, default=None)
    parser.add_argument("--stage", type=lambda s: [int(x) for x in s.split(",")],
                        default=None, help="Specific stages to benchmark")
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--non-causal-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=str, default=".")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required", file=sys.stderr)
        sys.exit(1)
    
    hdim = args.hdim
    nheads = args.nheads or (32 if hdim <= 64 else (16 if hdim <= 192 else 8))
    
    if args.causal_only:
        causals = [True]
    elif args.non_causal_only:
        causals = [False]
    else:
        causals = [False, True]
    
    results = run_full_benchmark(
        seqlens=args.seqlen,
        hdim=hdim,
        nheads=nheads,
        causal_modes=causals,
        stages=args.stage,
        warmup=args.warmup,
        repeat=args.rep
    )
    
    if args.csv:
        save_csv(results, args.csv)
    
    if args.plot:
        plot_roofline(results, args.plot_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    
    for stage in args.stage or range(5):
        stage_results = [r for r in results if r.stage == stage and not r.error]
        if stage_results:
            avg_tflops = sum(r.tflops for r in stage_results) / len(stage_results)
            avg_util = sum(r.tc_utilization_pct for r in stage_results) / len(stage_results)
            print(f"  Stage {stage}: {avg_tflops:.0f} TFLOPs/s avg ({avg_util:.0f}% TC util)")


if __name__ == "__main__":
    main()
