#!/usr/bin/env python3
"""
分析 Benchmark 结果

Usage:
    python analyze_results.py --latest
    python analyze_results.py --file results/benchmark_20260414T093000Z.json
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def load_results(results_dir: Path, filename: str = None) -> Dict[str, Any]:
    """加载 benchmark 结果"""
    
    if filename:
        filepath = results_dir / filename
    else:
        # 找最新的文件
        files = sorted(results_dir.glob("benchmark_*.json"))
        if not files:
            raise FileNotFoundError(f"No benchmark files in {results_dir}")
        filepath = files[-1]
    
    with open(filepath) as f:
        return json.load(f)


def analyze_results(data: Dict[str, Any]):
    """分析并展示结果"""
    
    print("\n" + "=" * 90)
    print("  FlashAttention CuTe DSL Benchmark Analysis")
    print("=" * 90)
    
    device = data.get("device", {})
    print(f"  GPU: {device.get('name', 'unknown')}")
    print(f"  Compute Capability: SM_{device.get('compute_capability', '?')}")
    print(f"  Peak TFLOPs: {device.get('peak_tflops', 0):.0f}")
    print("=" * 90)
    
    results = data.get("results", [])
    
    # 按 seqlen 分组
    from collections import defaultdict
    by_seqlen = defaultdict(list)
    for r in results:
        by_seqlen[r["seqlen"]].append(r)
    
    # 打印每个 seqlen 的性能
    for seqlen in sorted(by_seqlen.keys()):
        batch = by_seqlen[seqlen][0]["batch"]
        print(f"\n  Seqlen={seqlen}, Batch={batch}")
        print("  " + "-" * 86)
        print(f"  {'Stage':<8} {'Time(ms)':<12} {'TFLOPs':<12} {'TC Util%':<12} {'Speedup':<12}")
        print("  " + "-" * 86)
        
        baseline_tflops = None
        
        for r in sorted(by_seqlen[seqlen], key=lambda x: x["stage"]):
            if r.get("error"):
                time_cell = "ERR"
                tflops_cell = "ERR"
                util_cell = "ERR"
            else:
                time_cell = f"{r['avg_ms']:.3f}"
                tflops_cell = f"{r['tflops']:.1f}"
                util_cell = f"{r['tc_util_pct']:.2f}"
            
            if baseline_tflops is None and r.get("tflops", 0) > 0:
                baseline_tflops = r["tflops"]
                speedup_cell = "1.0x"
            elif baseline_tflops and r.get("tflops", 0) > 0:
                speedup = r["tflops"] / baseline_tflops
                speedup_cell = f"{speedup:.1f}x"
            else:
                speedup_cell = "N/A"
            
            print(f"  {r['stage']:<8} {time_cell:<12} {tflops_cell:<12} {util_cell:<12} {speedup_cell:<12}")
    
    # 总结
    print("\n" + "=" * 90)
    print("  Summary by Stage")
    print("=" * 90)
    
    stage_results = defaultdict(list)
    for r in results:
        if not r.get("error") and r.get("tflops", 0) > 0:
            stage_results[r["stage"]].append(r["tflops"])
    
    print(f"\n  {'Stage':<8} {'Avg TFLOPs':<15} {'Max TFLOPs':<15} {'Avg TC Util%':<15}")
    print("  " + "-" * 54)
    
    for stage in sorted(stage_results.keys()):
        tflops_list = stage_results[stage]
        avg_tflops = np.mean(tflops_list)
        max_tflops = np.max(tflops_list)
        avg_util = avg_tflops / device.get("peak_tflops", 100) * 100
        
        print(f"  {stage:<8} {avg_tflops:<15.1f} {max_tflops:<15.1f} {avg_util:<15.2f}")
    
    # 性能倍数
    if 0 in stage_results and 4 in stage_results:
        stage0_avg = np.mean(stage_results[0])
        stage4_avg = np.mean(stage_results[4])
        speedup = stage4_avg / stage0_avg
        print(f"\n  Overall Speedup (Stage 4 vs Stage 0): {speedup:.1f}x")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--latest", action="store_true", help="Analyze latest results")
    parser.add_argument("--file", type=str, help="Specific file to analyze")
    parser.add_argument("--results-dir", type=str, default="cute_attention/results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.file:
        data = load_results(results_dir, args.file)
    else:
        data = load_results(results_dir)
    
    analyze_results(data)


if __name__ == "__main__":
    main()
