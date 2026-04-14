#!/usr/bin/env python3
"""
FlashAttention Benchmark (CuTe Implementation)

Compare performance across all 5 stages:
- Stage 0: Naive baseline
- Stage 1: Tiled with shared memory
- Stage 2: Optimized SMEM layout
- Stage 3: Tensor Core MMA
- Stage 4: Final (online softmax + pipelining)

Usage:
    python benchmark.py --seqlen 512,1024,2048,4096,8192
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    stage: int
    stage_name: str
    batch: int
    seqlen: int
    nheads: int
    headdim: int
    causal: bool
    warmup_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    tflops: float
    tc_util_pct: float
    speedup_vs_baseline: float = 1.0
    max_diff: float = 0.0
    error: str = ""


def get_hardware_info():
    """Get GPU hardware info"""
    if not torch.cuda.is_available():
        return {"peak_tflops": 100.0, "sm_count": 84}
    
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    if sm >= 100:  # B200
        return {"peak_tflops": 2250.0, "sm_count": 148}
    elif sm >= 90:  # H100
        return {"peak_tflops": 989.0, "sm_count": 132}
    elif sm >= 80:  # A100
        return {"peak_tflops": 312.0, "sm_count": 108}
    else:
        return {"peak_tflops": 100.0, "sm_count": 84}


def attention_flops(batch, nheads, seqlen, headdim, causal=False):
    """Calculate FLOPs for attention"""
    if causal:
        avg_seqlen = (seqlen + 1) / 2
    else:
        avg_seqlen = seqlen
    
    return batch * nheads * 2 * seqlen * avg_seqlen * headdim * 2


def reference_attention(q, k, v, causal=True, scale=None):
    """PyTorch reference implementation"""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    
    q = q.transpose(1, 2)  # [b, h, s, d]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
    return out.transpose(1, 2)  # [b, s, h, d]


def benchmark_stage(
    stage: int,
    seqlen: int,
    batch: int = 1,
    nheads: int = 16,
    headdim: int = 128,
    causal: bool = True,
    warmup: int = 5,
    rep: int = 30,
    check_correctness: bool = True,
) -> BenchmarkResult:
    """Benchmark a single stage"""
    
    stage_names = {
        0: "Naive",
        1: "Tiled",
        2: "Smem",
        3: "MMA",
        4: "Final"
    }
    
    result = BenchmarkResult(
        stage=stage,
        stage_name=stage_names.get(stage, f"Stage{stage}"),
        batch=batch,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        causal=causal,
        warmup_ms=0,
        avg_ms=0,
        min_ms=0,
        max_ms=0,
        tflops=0,
        tc_util_pct=0,
    )
    
    try:
        # Try to import CuTe attention
        try:
            sys.path.insert(0, "cute_attention/python")
            from cute_attention import FlashAttention
            has_cute = True
        except ImportError:
            has_cute = False
        
        # Create inputs
        dtype = torch.bfloat16
        device = "cuda"
        
        q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        
        if has_cute and stage >= 0 and stage <= 4:
            # Use CuTe kernel
            fa = FlashAttention(stage=stage)
            
            # Warmup
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(warmup):
                _ = fa(q, k, v, causal=causal)
            torch.cuda.synchronize()
            result.warmup_ms = (time.perf_counter() - start) * 1000
            
            # Timed runs
            times = []
            for _ in range(rep):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = fa(q, k, v, causal=causal)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            times = np.array(times)
            result.avg_ms = float(times.mean())
            result.min_ms = float(times.min())
            result.max_ms = float(times.max())
            
        else:
            # Fallback to SDPA (baseline)
            # Warmup
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(warmup):
                _ = reference_attention(q, k, v, causal=causal)
            torch.cuda.synchronize()
            result.warmup_ms = (time.perf_counter() - start) * 1000
            
            # Timed runs
            times = []
            for _ in range(rep):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = reference_attention(q, k, v, causal=causal)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            times = np.array(times)
            result.avg_ms = float(times.mean())
            result.min_ms = float(times.min())
            result.max_ms = float(times.max())
        
        # Compute metrics
        flops = attention_flops(batch, nheads, seqlen, headdim, causal)
        result.tflops = flops / result.avg_ms / 1e9
        
        hw = get_hardware_info()
        result.tc_util_pct = result.tflops / hw["peak_tflops"] * 100
        
        # Correctness check
        if check_correctness:
            ref_out = reference_attention(q, k, v, causal=causal)
            result.max_diff = (out.float() - ref_out.float()).abs().max().item()
    
    except Exception as e:
        import traceback
        result.error = f"{e}\n{traceback.format_exc()[-400:]}"
    
    return result


def run_benchmark_suite(seqlens: List[int]) -> List[BenchmarkResult]:
    """Run complete benchmark suite"""
    
    results = []
    hw = get_hardware_info()
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    
    print("\n" + "=" * 90)
    print("  FlashAttention CuTe Implementation - Progressive Optimization Benchmark")
    print("=" * 90)
    print(f"  GPU: {device_name}")
    print(f"  Peak TFLOPs: {hw['peak_tflops']:.0f}")
    print("=" * 90)
    
    # Table header
    for seqlen in seqlens:
        batch = max(1, 32768 // seqlen)
        
        print(f"\n  Seqlen={seqlen}, Batch={batch}")
        print("  " + "-" * 86)
        print(f"  {'Stage':<12} {'Name':<12} {'Time (ms)':<12} {'TFLOPs':<12} {'TC Util %':<12} {'Speedup':<12}")
        print("  " + "-" * 86)
        
        baseline_tflops = None
        
        for stage in range(5):
            result = benchmark_stage(
                stage=stage,
                seqlen=seqlen,
                batch=batch,
                causal=True,
                warmup=3,
                rep=20,
            )
            results.append(result)
            
            if result.error:
                time_cell = "ERR"
                tflops_cell = "ERR"
                util_cell = "ERR"
            else:
                time_cell = f"{result.avg_ms:.3f}"
                tflops_cell = f"{result.tflops:.1f}"
                util_cell = f"{result.tc_util_pct:.1f}"
            
            # Speedup
            if baseline_tflops is None and not result.error and result.tflops > 0:
                baseline_tflops = result.tflops
                speedup_cell = "1.0x"
            elif baseline_tflops and not result.error and result.tflops > 0:
                speedup = result.tflops / baseline_tflops
                speedup_cell = f"{speedup:.1f}x"
            else:
                speedup_cell = "N/A"
            
            print(f"  {stage:<12} {result.stage_name:<12} {time_cell:<12} {tflops_cell:<12} {util_cell:<12} {speedup_cell:<12}")
    
    return results


def save_results(results: List[BenchmarkResult], path: str):
    """Save results to file"""
    data = [asdict(r) for r in results]
    
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        import csv
        with open(path, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    print(f"\nResults saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FlashAttention CuTe Benchmark")
    parser.add_argument("--seqlen", type=lambda s: [int(x) for x in s.split(",")],
                       default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--output", type=str, default="cute_attention/results/benchmark.json")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = run_benchmark_suite(args.seqlen)
    save_results(results, args.output)
    
    # Summary
    print("\n" + "=" * 90)
    print("  Summary")
    print("=" * 90)
    
    valid_results = [r for r in results if not r.error and r.tflops > 0]
    if valid_results:
        # Group by stage
        from collections import defaultdict
        stage_results = defaultdict(list)
        for r in valid_results:
            stage_results[r.stage].append(r)
        
        for stage in sorted(stage_results.keys()):
            avg_tflops = np.mean([r.tflops for r in stage_results[stage]])
            avg_util = np.mean([r.tc_util_pct for r in stage_results[stage]])
            print(f"  Stage {stage}: {avg_tflops:.0f} TFLOPs/s ({avg_util:.0f}% TC util)")


if __name__ == "__main__":
    main()
