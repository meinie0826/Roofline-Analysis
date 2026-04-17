"""
Roofline Analysis Script

Analyzes GEMM performance and generates roofline plots comparing
cuBLAS (BF16) and DeepGEMM (FP8) implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available")


@dataclass
class BenchmarkPoint:
    """Single benchmark data point."""
    M: int
    N: int
    K: int
    dtype: str
    backend: str
    time_ms: float
    gflops: float
    arithmetic_intensity: float
    bandwidth_gbps: float
    compute_efficiency: float


def load_results(filepath: str) -> List[BenchmarkPoint]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    points = []
    for r in data['results']:
        points.append(BenchmarkPoint(
            M=r['M'], N=r['N'], K=r['K'],
            dtype=r['dtype'],
            backend=r['backend'],
            time_ms=r['time_ms'],
            gflops=r['gflops'],
            arithmetic_intensity=r['arithmetic_intensity'],
            bandwidth_gbps=r['achieved_bandwidth_gbps'],
            compute_efficiency=r['compute_efficiency'],
        ))
    
    return points


def compute_roofline(ai: np.ndarray, 
                     peak_fp8_tflops: float = 5000,
                     peak_bf16_tflops: float = 1250,
                     peak_bandwidth_gbps: float = 8000) -> dict:
    """
    Compute roofline limits for different precisions.
    
    Returns dict with 'fp8' and 'bf16' roofline GFLOPS values.
    """
    ridge_fp8 = peak_fp8_tflops * 1000 / peak_bandwidth_gbps
    ridge_bf16 = peak_bf16_tflops * 1000 / peak_bandwidth_gbps
    
    gflops_fp8 = np.minimum(ai * peak_bandwidth_gbps, peak_fp8_tflops * 1000)
    gflops_bf16 = np.minimum(ai * peak_bandwidth_gbps, peak_bf16_tflops * 1000)
    
    return {
        'fp8': gflops_fp8,
        'bf16': gflops_bf16,
        'ridge_fp8': ridge_fp8,
        'ridge_bf16': ridge_bf16,
    }


def plot_roofline(points: List[BenchmarkPoint],
                   output_file: Optional[str] = None,
                   title: str = "GEMM Roofline Analysis"):
    """Generate roofline plot."""
    if not HAS_MPL:
        print("Error: matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # GPU specs for B300
    peak_fp8_tflops = 5000
    peak_bf16_tflops = 1250
    peak_bandwidth_gbps = 8000
    
    # Create roofline curves
    ai_values = np.logspace(-1, 4, 1000)
    roofline = compute_roofline(ai_values, peak_fp8_tflops, peak_bf16_tflops, peak_bandwidth_gbps)
    
    # Plot rooflines
    ax.loglog(ai_values, roofline['fp8'], 'b-', linewidth=2.5, label='FP8 Roofline (5000 TFLOPS)')
    ax.loglog(ai_values, roofline['bf16'], 'r-', linewidth=2.5, label='BF16 Roofline (1250 TFLOPS)')
    
    # Shade regions
    ridge_fp8 = roofline['ridge_fp8']
    ridge_bf16 = roofline['ridge_bf16']
    
    # Memory-bound region
    ai_mem = np.logspace(-1, max(np.log10(ridge_fp8), np.log10(ridge_bf16)), 100)
    mem_bound = ai_mem * peak_bandwidth_gbps
    ax.fill_between(ai_mem, mem_bound * 0.01, mem_bound, alpha=0.1, color='blue')
    
    # Compute-bound regions
    ai_comp_fp8 = np.logspace(np.log10(ridge_fp8), 4, 100)
    ax.fill_between(ai_comp_fp8, peak_fp8_tflops * 1000 * 0.01, 
                    peak_fp8_tflops * 1000, alpha=0.1, color='purple')
    
    ai_comp_bf16 = np.logspace(np.log10(ridge_bf16), 4, 100)
    ax.fill_between(ai_comp_bf16, peak_bf16_tflops * 1000 * 0.01,
                    peak_bf16_tflops * 1000, alpha=0.1, color='orange')
    
    # Separate by backend and dtype
    backends = {}
    for p in points:
        key = f"{p.backend} ({p.dtype})"
        if key not in backends:
            backends[key] = []
        backends[key].append(p)
    
    # Color scheme
    colors = {
        'cuBLAS (torch.matmul)': {'bf16': '#e74c3c', 'fp16': '#c0392b'},
        'DeepGEMM': {'fp8': '#3498db', 'bf16': '#2980b9'},
    }
    
    markers = {
        'cuBLAS (torch.matmul)': 'o',
        'DeepGEMM': 's',
    }
    
    # Plot data points
    for key, pts in backends.items():
        ai = np.array([p.arithmetic_intensity for p in pts])
        gflops = np.array([p.gflops for p in pts])
        
        # Extract backend and dtype from key
        parts = key.split(' (')
        backend = parts[0]
        dtype = parts[1].rstrip(')')
        
        color = colors.get(backend, {}).get(dtype, 'gray')
        marker = markers.get(backend, 'o')
        
        scatter = ax.scatter(ai, gflops, c=color, s=100, marker=marker,
                            alpha=0.8, edgecolors='black', linewidths=0.5,
                            label=key, zorder=10)
        
        # Annotate some points
        sorted_pts = sorted(pts, key=lambda x: x.arithmetic_intensity)
        n_annotate = min(8, len(sorted_pts))
        for i in range(0, len(sorted_pts), max(1, len(sorted_pts) // n_annotate)):
            p = sorted_pts[i]
            label = f"M={p.M}\nK={p.K}"
            ax.annotate(label, (p.arithmetic_intensity, p.gflops),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Ridge point markers
    ax.axvline(x=ridge_fp8, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=ridge_bf16, color='red', linestyle='--', alpha=0.5)
    
    ax.text(ridge_fp8 * 1.2, peak_fp8_tflops * 500, f'FP8 Ridge\nAI={ridge_fp8:.0f}',
           fontsize=9, color='blue')
    ax.text(ridge_bf16 * 1.2, peak_bf16_tflops * 500, f'BF16 Ridge\nAI={ridge_bf16:.0f}',
           fontsize=9, color='red')
    
    # Axis settings
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xlim(1, max(ai_values) * 2)
    ax.set_ylim(100, peak_fp8_tflops * 1500)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Add info box
    info_text = f'Peak FP8: {peak_fp8_tflops} TFLOPS\nPeak BF16: {peak_bf16_tflops} TFLOPS\nBW: {peak_bandwidth_gbps} GB/s'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return fig


def plot_comparison(points: List[BenchmarkPoint],
                    output_file: Optional[str] = None):
    """Plot performance comparison between backends."""
    if not HAS_MPL:
        return
    
    # Group by shape
    shapes = {}
    for p in points:
        key = (p.M, p.N, p.K)
        if key not in shapes:
            shapes[key] = []
        shapes[key].append(p)
    
    # Find shapes with multiple backends
    comparison_shapes = {k: v for k, v in shapes.items() if len(v) > 1}
    
    if not comparison_shapes:
        print("No shapes with multiple backends to compare")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x_labels = [f"{m}x{n}x{k}" for (m, n, k) in sorted(comparison_shapes.keys())]
    x = np.arange(len(x_labels))
    width = 0.35
    
    backends = set(p.backend for pts in comparison_shapes.values() for p in pts)
    colors = {'cuBLAS (torch.matmul)': '#e74c3c', 'DeepGEMM': '#3498db'}
    
    for i, backend in enumerate(sorted(backends)):
        values = []
        for key in sorted(comparison_shapes.keys()):
            pts = [p for p in comparison_shapes[key] if p.backend == backend]
            if pts:
                values.append(pts[0].gflops)
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=backend, color=colors.get(backend, 'gray'))
    
    ax.set_xlabel('Matrix Shape (MxNxK)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('GEMM Performance Comparison: cuBLAS vs DeepGEMM', fontsize=14)
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot roofline results")
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output plot file")
    parser.add_argument("--title", type=str, default="GEMM Roofline Analysis",
                       help="Plot title")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    points = load_results(args.results_file)
    print(f"Loaded {len(points)} benchmark results")
    
    # Summary stats
    print("\nPerformance Summary:")
    print("-" * 60)
    
    backends = {}
    for p in points:
        key = f"{p.backend} ({p.dtype})"
        if key not in backends:
            backends[key] = []
        backends[key].append(p)
    
    for key, pts in backends.items():
        gflops = [p.gflops for p in pts]
        print(f"\n{key}:")
        print(f"  Samples: {len(pts)}")
        print(f"  Max GFLOPS: {max(gflops):.1f}")
        print(f"  Min GFLOPS: {min(gflops):.1f}")
        print(f"  Avg GFLOPS: {np.mean(gflops):.1f}")
        print(f"  Avg compute efficiency: {np.mean([p.compute_efficiency for p in pts])*100:.1f}%")
    
    # Generate plots
    print("\nGenerating plots...")
    
    output_base = args.output or Path(args.results_file).stem
    plot_roofline(points, f"{output_base}_roofline.png", args.title)
    plot_comparison(points, f"{output_base}_comparison.png")


if __name__ == "__main__":
    main()
