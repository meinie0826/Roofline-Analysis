#!/usr/bin/env python3
"""
可视化 Benchmark 结果

Usage:
    python plot_results.py --file ../results/benchmark_20260414T102548Z.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath: str) -> dict:
    """加载 JSON 数据"""
    with open(filepath) as f:
        return json.load(f)


def plot_performance(data: dict, output_dir: Path):
    """绘制性能图表"""
    results = data["results"]
    
    # 按序列长度分组
    by_seqlen = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            by_seqlen[r["seqlen"]][r["stage"]].append(r["tflops"])
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：各序列长度的性能
    ax = axes[0]
    seqlens = sorted(by_seqlen.keys())
    stages = [0, 1, 2, 3, 4]
    
    x = np.arange(len(seqlens))
    width = 0.15
    
    for i, stage in enumerate(stages):
        tflops = [np.mean(by_seqlen[s][stage]) if stage in by_seqlen[s] else 0 
                  for s in seqlens]
        ax.bar(x + i * width, tflops, width, label=f"Stage {stage}")
    
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("TFLOPs/s")
    ax.set_title("FlashAttention Performance by Stage")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(seqlens)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 右图：加速比
    ax = axes[1]
    
    speedups = defaultdict(list)
    for seqlen in seqlens:
        baseline = np.mean(by_seqlen[seqlen][0]) if 0 in by_seqlen[seqlen] else 1
        for stage in stages:
            if stage in by_seqlen[seqlen]:
                speedup = np.mean(by_seqlen[seqlen][stage]) / baseline
                speedups[stage].append(speedup)
    
    for i, stage in enumerate(stages[1:], 1):
        ax.plot(seqlens, speedups[stage], marker="o", label=f"Stage {stage}")
    
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup vs Stage 0")
    ax.set_title("Optimization Speedup")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "performance_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    
    # 另一张图：TC Utilization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, stage in enumerate(stages):
        tc_util = [np.mean([r["tc_util_pct"] for r in results 
                           if r["stage"] == stage and r["seqlen"] == s])
                   if any(r["stage"] == stage and r["seqlen"] == s and not r.get("error") 
                          for r in results) else 0
                   for s in seqlens]
        ax.plot(seqlens, tc_util, marker="s", label=f"Stage {stage}", linewidth=2)
    
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tensor Core Utilization (%)")
    ax.set_title("Tensor Core Utilization by Stage")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 添加峰值线
    peak_tc_util = 50.0  # 理论峰值 TC 利用率
    ax.axhline(y=peak_tc_util, color="red", linestyle="--", alpha=0.5, label="Peak (50%)")
    
    plt.tight_layout()
    output_path = output_dir / "tc_utilization.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_roofline(data: dict, output_dir: Path):
    """绘制 Roofline 图"""
    device = data["device"]
    peak_tflops = device["peak_tflops"]
    peak_bw_gbps = 80 * 1024  # B200 HBM3 ~80TB/s
    
    results = data["results"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制 roofline
    # 计算密集区（水平线）
    x_compute = np.logspace(-1, 3, 100)
    y_compute = np.full_like(x_compute, peak_tflops)
    ax.plot(x_compute, y_compute, "r-", linewidth=2, label=f"Peak TFLOPs ({peak_tflops:.0f})")
    
    # 带宽密集区（斜线）
    x_bw = np.logspace(-2, 0, 100)
    y_bw = x_bw * peak_bw_gbps / 1000  # AI * BW = TFLOPs
    ax.plot(x_bw[:50], y_bw[:50], "b-", linewidth=2, label=f"Peak BW ({peak_bw_gbps/1024:.0f} TB/s)")
    
    # 绘制各 stage 的点
    colors = ["gray", "orange", "yellow", "lightgreen", "green"]
    markers = ["o", "s", "^", "D", "*"]
    
    for stage in range(5):
        stage_results = [r for r in results if r["stage"] == stage and not r.get("error")]
        if not stage_results:
            continue
        
        for r in stage_results:
            # 计算算术强度 (AI)
            # FLOPs/Byte = attention_flops / (input_bytes + output_bytes)
            batch, seqlen, headdim = r["batch"], r["seqlen"], r["headdim"]
            flops = batch * 16 * seqlen * seqlen * headdim * 2  # 简化
            bytes_io = batch * 16 * seqlen * headdim * 4 * 2  # Q,K,V,O (BF16)
            ai = flops / bytes_io
            
            ax.scatter(ai, r["tflops"], 
                      c=colors[stage], marker=markers[stage], s=100, 
                      alpha=0.7, edgecolor="black",
                      label=f"Stage {stage}" if r == stage_results[0] else "")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (TFLOPs/s)")
    ax.set_title(f"Roofline Model - {device['name']}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_path = output_dir / "roofline.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--file", type=str, required=True, help="JSON results file")
    parser.add_argument("--out-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    data = load_data(args.file)
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.file).parent
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_performance(data, out_dir)
    plot_roofline(data, out_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
