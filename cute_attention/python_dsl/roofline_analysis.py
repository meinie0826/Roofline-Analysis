#!/usr/bin/env python3
"""
Roofline 分析工具

根据 benchmark 结果生成 Roofline 图，展示各优化方法在 roofline 上的位置
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================================
# Device Peak Performance
# ============================================================================

DEVICE_SPECS = {
    "NVIDIA B200": {
        "peak_tflops_bf16": 2250.0,  # Tensor Core BF16
        "peak_tflops_fp32": 1125.0,  # Tensor Core FP32
        "peak_bw_gbps": 80 * 1024,    # HBM3 ~80 TB/s
        "peak_tc_util": 0.50,         # 理论最大 TC 利用率
    },
    "NVIDIA H100": {
        "peak_tflops_bf16": 989.0,
        "peak_tflops_fp32": 495.0,
        "peak_bw_gbps": 80 * 1024,    # HBM3
        "peak_tc_util": 0.50,
    },
    "NVIDIA A100": {
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp32": 156.0,
        "peak_bw_gbps": 40 * 1024,    # HBM2e
        "peak_tc_util": 0.50,
    },
}


def calculate_roofline(device_info: dict, arithmetic_intensity: np.ndarray) -> np.ndarray:
    """计算 roofline 曲线"""
    device_name = device_info.get("name", "NVIDIA B200")
    specs = DEVICE_SPECS.get(device_name, DEVICE_SPECS["NVIDIA B200"])
    
    peak_tflops = specs["peak_tflops_bf16"]
    peak_bw = specs["peak_bw_gbps"]
    
    # 计算密集区 (AI > ridge point)
    ridge_point = peak_tflops * 1e12 / (peak_bw * 1e9)  # FLOPs/Byte
    y_compute = np.full_like(arithmetic_intensity, peak_tflops, dtype=float)
    
    # 带宽密集区 (AI < ridge point)
    y_memory = arithmetic_intensity * peak_bw / 1e6  # TFLOPs
    
    # Roofline: min of compute and memory
    roofline = np.minimum(y_compute, y_memory)
    
    return roofline, peak_tflops, ridge_point


def plot_roofline_with_results(data: dict, output_dir: Path):
    """绘制包含所有 kernel 结果的 Roofline 图"""
    
    device = data.get("device", {})
    device_name = device.get("name", "NVIDIA B200")
    specs = DEVICE_SPECS.get(device_name, DEVICE_SPECS["NVIDIA B200"])
    
    results = data.get("results", [])
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 左图：Roofline 模型 ==========
    ax = axes[0]
    
    # 绘制 roofline
    ai = np.logspace(-1, 4, 1000)
    roofline, peak_tflops, ridge_point = calculate_roofline(device, ai)
    
    ax.loglog(ai, roofline, "k-", linewidth=3, label=f"Peak ({peak_tflops:.0f} TFLOPs)")
    ax.axvline(x=ridge_point, color="gray", linestyle="--", alpha=0.5, 
               label=f"Ridge Point ({ridge_point:.1f} FLOPs/B)")
    
    # 标注带宽密集区和计算密集区
    ax.fill_between(ai[ai < ridge_point], roofline[ai < ridge_point], 
                    alpha=0.1, color="blue", label="Memory Bound")
    ax.fill_between(ai[ai >= ridge_point], roofline[ai >= ridge_point],
                    alpha=0.1, color="red", label="Compute Bound")
    
    stage_defs = data.get("stage_definitions", [])
    stage_name_map = {
        int(item["stage"]): item.get("name", f"Stage{item['stage']}")
        for item in stage_defs
        if "stage" in item
    }

    # Kernel 颜色和标记
    kernel_styles = {
        "SDPA": {"color": "blue", "marker": "o", "size": 100},
        "FA2": {"color": "cyan", "marker": "s", "size": 100},
        "FA3": {"color": "green", "marker": "^", "size": 100},
        "FA4": {"color": "lime", "marker": "D", "size": 120},
    }
    
    # 绘制各 kernel 的点
    for r in results:
        if r.get("error"):
            continue
        
        kernel = r.get("kernel", "Unknown")
        style = kernel_styles.get(kernel)
        if style is None and kernel.startswith("Stage"):
            try:
                stage_idx = int(kernel.replace("Stage", ""))
            except ValueError:
                continue
            color = plt.cm.plasma(stage_idx / max(1, len(stage_name_map) - 1 if stage_name_map else 7))
            markers = ["x", "v", "<", ">", "P", "*", "X", "d", "h"]
            style = {
                "color": color,
                "marker": markers[stage_idx % len(markers)],
                "size": 90 + 8 * stage_idx,
            }
        if style is None:
            continue

        ai_val = r.get("arithmetic_intensity", 10.0)
        tflops = r.get("tflops", 0.0)
        
        ax.scatter(ai_val, tflops, 
                   c=style["color"], marker=style["marker"], s=style["size"],
                   edgecolor="black", alpha=0.7,
                   label=f"{kernel}")
    
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPs/s)", fontsize=12)
    ax.set_title(f"Roofline Model - {device_name}", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1, peak_tflops * 1.2)
    
    # ========== 右图：性能对比柱状图 ==========
    ax = axes[1]
    
    # 按配置分组
    by_config = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("error"):
            key = f"S={r['seqlen']} H={r['nheads']}"
            by_config[key][r["kernel"]] = r["tflops"]
    
    configs = sorted(by_config.keys())[:5]  # 只显示前5个配置
    kernels = ["SDPA", "FA2", "FA3", "FA4"]
    stage_kernels = sorted(
        [k for k in {r.get("kernel", "") for r in results} if k.startswith("Stage")],
        key=lambda k: int(k.replace("Stage", ""))
    )
    kernels.extend(stage_kernels)
    
    x = np.arange(len(configs))
    width = 0.08
    
    for i, kernel in enumerate(kernels):
        tflops_list = []
        for config in configs:
            if kernel in by_config[config]:
                tflops_list.append(np.mean(by_config[config][kernel]))
            else:
                tflops_list.append(0)
        
        style = kernel_styles.get(kernel, {"color": "gray"})
        ax.bar(x + i * width, tflops_list, width, 
               label=kernel, color=style["color"], alpha=0.8)
    
    ax.axhline(y=peak_tflops, color="red", linestyle="--", alpha=0.5, label=f"Peak ({peak_tflops:.0f} TF)")
    
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("TFLOPs/s", fontsize=12)
    ax.set_title("Performance Comparison", fontsize=14)
    ax.set_xticks(x + width * 4)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "roofline_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    
    # ========== 消融分析图 ==========
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取 Stage 数据，计算每个优化的贡献
    stage_data = defaultdict(list)
    for r in results:
        if r["kernel"].startswith("Stage") and not r.get("error"):
            stage = int(r["kernel"].replace("Stage", ""))
            stage_data[stage].append(r["tflops"])
    
    if len(stage_data) > 0:
        stages = sorted(stage_data.keys())
        mean_tflops = [np.mean(stage_data[s]) for s in stages]
        
        colors = [plt.cm.plasma(i / max(1, len(stages) - 1)) for i in range(len(stages))]
        ax.bar(stages, mean_tflops, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
        
        # 标注加速比
        baseline = mean_tflops[0]
        for i, (s, tf) in enumerate(zip(stages, mean_tflops)):
            speedup = tf / baseline
            ax.annotate(f"{tf:.0f} TF\n({speedup:.1f}x)", 
                       (s, tf + 20), ha="center", fontsize=11, fontweight="bold")
        
        # 添加优化说明
        optimization_names = [
            stage_name_map.get(s, f"Stage{s}").replace(" + ", "\n+")
            for s in stages
        ]
        
        ax.set_xticks(stages)
        ax.set_xticklabels(optimization_names, fontsize=11)
        ax.set_xlabel("Optimization Stage", fontsize=12)
        ax.set_ylabel("TFLOPs/s", fontsize=12)
        ax.set_title(f"Ablation Study: Optimization Contributions\n{device_name}", fontsize=14)
        ax.axhline(y=peak_tflops, color="red", linestyle="--", alpha=0.7, 
                   label=f"Peak ({peak_tflops:.0f} TF)")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "ablation_analysis.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_path}")


def analyze_roofline_efficiency(data: dict) -> dict:
    """分析 roofline 效率"""
    
    device = data.get("device", {})
    device_name = device.get("name", "NVIDIA B200")
    specs = DEVICE_SPECS.get(device_name, DEVICE_SPECS["NVIDIA B200"])
    peak_tflops = specs["peak_tflops_bf16"]
    
    results = data.get("results", [])
    
    analysis = {}
    
    for r in results:
        if r.get("error"):
            continue
        
        kernel = r.get("kernel", "")
        tflops = r.get("tflops", 0)
        tc_util = r.get("tc_util_pct", 0)
        ai = r.get("arithmetic_intensity", 0)
        
        # 判断是带宽密集还是计算密集
        ridge_point = peak_tflops * 1e12 / (specs["peak_bw_gbps"] * 1e9)
        bound_type = "compute" if ai > ridge_point else "memory"
        
        # 计算效率
        efficiency = tflops / peak_tflops
        
        analysis[kernel] = {
            "tflops": tflops,
            "tc_util_pct": tc_util,
            "arithmetic_intensity": ai,
            "bound_type": bound_type,
            "efficiency": efficiency,
            "distance_to_peak": peak_tflops - tflops,
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Roofline Analysis")
    parser.add_argument("--file", type=str, required=True, help="Benchmark JSON file")
    parser.add_argument("--out-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    with open(args.file) as f:
        data = json.load(f)
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.file).parent
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析效率
    analysis = analyze_roofline_efficiency(data)
    
    # 保存分析结果
    analysis_path = out_dir / "efficiency_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: {analysis_path}")
    
    # 绘制图表
    plot_roofline_with_results(data, out_dir)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("  Efficiency Analysis Summary")
    print("=" * 80)
    print(f"  {'Kernel':<12} {'TFLOPs':<12} {'TC Util%':<12} {'Efficiency':<12} {'Bound Type':<12}")
    print("  " + "-" * 60)
    
    for kernel, info in sorted(analysis.items()):
        print(f"  {kernel:<12} {info['tflops']:<12.1f} {info['tc_util_pct']:<12.2f} "
              f"{info['efficiency']*100:<12.2f}% {info['bound_type']:<12}")
    
    print()


if __name__ == "__main__":
    main()
