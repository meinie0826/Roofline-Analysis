#!/usr/bin/env python3
"""
FlashAttention B200 性能分析报告生成器
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def generate_report(data_file: str):
    """生成完整分析报告"""
    
    with open(data_file) as f:
        data = json.load(f)
    
    device = data["device"]
    install_status = data["install_status"]
    results = data["results"]
    
    # 分析数据
    print("\n" + "=" * 90)
    print("  FlashAttention B200 性能分析报告")
    print("=" * 90)
    print(f"  GPU: {device['name']}")
    print(f"  Compute: SM_{device['compute_capability']}")
    print(f"  Peak BF16: {device['peak_tflops']:.0f} TFLOPs")
    print(f"  Memory: {device['memory_gb']:.1f} GB")
    print("=" * 90)
    
    print("\n  Kernel 安装状态:")
    for name, installed in install_status.items():
        status = "✓" if installed else "✗"
        print(f"    {name}: {status}")
    
    # 按配置分组统计
    by_config = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            key = f"S={r['seqlen']} B={r['batch_size']} H={r['nheads']}/{r['nheads_kv']}"
            by_config[key][r["kernel"]].append(r["tflops"])
    
    print("\n" + "=" * 90)
    print("  性能对比（平均 TFLOPs）")
    print("=" * 90)
    
    kernels = ["SDPA", "FA2", "FA4", "Stage0", "Stage1", "Stage2", "Stage3", "Stage4"]
    
    print(f"\n  {'Config':<25}", end="")
    for k in kernels:
        print(f" {k:<10}", end="")
    print()
    print("  " + "-" * 100)
    
    kernel_avg = defaultdict(list)
    for config in sorted(by_config.keys()):
        print(f"  {config:<25}", end="")
        for kernel in kernels:
            if kernel in by_config[config] and by_config[config][kernel]:
                avg = np.mean(by_config[config][kernel])
                kernel_avg[kernel].append(avg)
                print(f" {avg:>8.1f}  ", end="")
            else:
                print(f" {'N/A':>8}  ", end="")
        print()
    
    # 总体统计
    print("\n" + "=" * 90)
    print("  总体统计")
    print("=" * 90)
    
    print(f"\n  {'Kernel':<10} {'Avg TFLOPs':<12} {'Peak TFLOPs':<12} {'TC Util%':<10} {'Efficiency':<10}")
    print("  " + "-" * 54)
    
    summary = {}
    for kernel in kernels:
        if kernel_avg[kernel]:
            avg = np.mean(kernel_avg[kernel])
            peak = np.max(kernel_avg[kernel])
            tc_util = avg / device['peak_tflops'] * 100
            efficiency = avg / device['peak_tflops']
            
            summary[kernel] = {
                "avg_tflops": avg,
                "peak_tflops": peak,
                "tc_util_pct": tc_util,
                "efficiency": efficiency
            }
            
            print(f"  {kernel:<10} {avg:<12.1f} {peak:<12.1f} {tc_util:<10.2f} {efficiency*100:<10.2f}%")
    
    # 对比分析
    print("\n" + "=" * 90)
    print("  对比分析")
    print("=" * 90)
    
    if "SDPA" in summary and "FA4" in summary:
        fa4_speedup = summary["FA4"]["avg_tflops"] / summary["SDPA"]["avg_tflops"]
        print(f"\n  FA4 vs SDPA: {fa4_speedup:.2f}x faster")
        print(f"  FA4 比 SDPA 快 {(fa4_speedup-1)*100:.1f}%")
    
    if "FA4" in summary:
        tc_gap = 100 - summary["FA4"]["tc_util_pct"]
        print(f"\n  FA4 距离峰值还有 {tc_gap:.1f}% 的 TC 利用率空间")
        print(f"  理论上还可以提升 {device['peak_tflops'] - summary['FA4']['avg_tflops']:.0f} TFLOPs")
    
    # 绘制图表
    plot_summary(summary, device, Path(data_file).parent)
    
    return summary


def plot_summary(summary: dict, device: dict, output_dir: Path):
    """绘制总结图表"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：性能对比
    ax = axes[0]
    kernels = list(summary.keys())
    avg_tflops = [summary[k]["avg_tflops"] for k in kernels]
    peak_tflops = [summary[k]["peak_tflops"] for k in kernels]
    
    x = np.arange(len(kernels))
    width = 0.35
    
    ax.bar(x - width/2, avg_tflops, width, label="Average", color="steelblue")
    ax.bar(x + width/2, peak_tflops, width, label="Peak", color="darkorange")
    
    ax.axhline(y=device['peak_tflops'], color='red', linestyle='--', 
               alpha=0.7, label=f"Peak ({device['peak_tflops']:.0f} TF)")
    
    ax.set_xlabel("Kernel")
    ax.set_ylabel("TFLOPs/s")
    ax.set_title(f"Performance Summary - {device['name']}")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 右图：TC 利用率
    ax = axes[1]
    tc_util = [summary[k]["tc_util_pct"] for k in kernels]
    
    colors = ['green' if t > 50 else 'orange' if t > 40 else 'red' for t in tc_util]
    bars = ax.bar(kernels, tc_util, color=colors, alpha=0.7)
    
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label="Peak (100%)")
    ax.axhline(y=50, color='green', linestyle=':', alpha=0.5, label="Target (50%)")
    
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Tensor Core Utilization (%)")
    ax.set_title("TC Utilization")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, tc_util):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "performance_summary.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n  保存图表: {output_path}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        data_file = "cute_attention/results/benchmark_comprehensive_20260414T113018Z.json"
    else:
        data_file = sys.argv[1]
    
    generate_report(data_file)


if __name__ == "__main__":
    main()
