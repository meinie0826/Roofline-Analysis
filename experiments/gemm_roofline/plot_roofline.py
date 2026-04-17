"""
Roofline Plotting Script

This script generates roofline plots from benchmark results to visualize
the performance characteristics of GEMM operations across different shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


@dataclass
class BenchmarkPoint:
    """A single benchmark data point."""
    M: int
    N: int
    K: int
    time_ms: float
    gflops: float
    arithmetic_intensity: float
    achieved_bandwidth_gbps: float
    compute_efficiency: float
    memory_efficiency: float


def load_results(filepath: str) -> List[BenchmarkPoint]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    points = []
    for r in data["results"]:
        points.append(BenchmarkPoint(
            M=r["M"], N=r["N"], K=r["K"],
            time_ms=r["time_ms"],
            gflops=r["gflops"],
            arithmetic_intensity=r["arithmetic_intensity"],
            achieved_bandwidth_gbps=r["achieved_bandwidth_gbps"],
            compute_efficiency=r["compute_efficiency"],
            memory_efficiency=r["memory_efficiency"],
        ))
    
    return points


def compute_roofline(ai: np.ndarray, peak_tflops: float, peak_bandwidth_gbps: float) -> np.ndarray:
    """
    Compute the roofline limit.
    
    The roofline is:
    - Memory-bound region: AI < Ridge Point, performance = AI * peak_bandwidth
    - Compute-bound region: AI >= Ridge Point, performance = peak_tflops
    
    Ridge Point = peak_tflops / peak_bandwidth
    """
    ridge_point = peak_tflops * 1000 / peak_bandwidth_gbps  # Convert TFLOPS to GFLOPS
    
    gflops = np.minimum(ai * peak_bandwidth_gbps, peak_tflops * 1000)
    return gflops


def plot_roofline(points: List[BenchmarkPoint],
                  peak_tflops: float = 2500,
                  peak_bandwidth_gbps: float = 8000,
                  output_file: Optional[str] = None,
                  title: str = "GEMM Roofline Analysis"):
    """
    Generate a roofline plot.
    
    Args:
        points: List of benchmark data points
        peak_tflops: Peak compute performance in TFLOPS
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s
        output_file: Output file path for the plot
        title: Plot title
    """
    if not HAS_MPL:
        print("Error: matplotlib is required for plotting")
        return
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Roofline parameters
    ridge_point = peak_tflops * 1000 / peak_bandwidth_gbps  # GFLOPS per Byte
    
    # Create roofline line
    ai_values = np.logspace(-1, 4, 1000)  # Arithmetic intensity from 0.1 to 10000
    roofline_gflops = compute_roofline(ai_values, peak_tflops, peak_bandwidth_gbps)
    
    # Plot roofline
    ax.loglog(ai_values, roofline_gflops, 'k-', linewidth=2.5, label='Roofline Limit')
    
    # Shade the regions
    ai_mem = np.logspace(-1, np.log10(ridge_point), 100)
    mem_bound = ai_mem * peak_bandwidth_gbps
    ax.fill_between(ai_mem, mem_bound * 0.01, mem_bound, alpha=0.15, color='blue', 
                   label='Memory-Bound Region')
    
    ai_comp = np.logspace(np.log10(ridge_point), 4, 100)
    comp_bound = np.ones_like(ai_comp) * peak_tflops * 1000
    ax.fill_between(ai_comp, comp_bound * 0.01, comp_bound, alpha=0.15, color='red',
                   label='Compute-Bound Region')
    
    # Extract data from points
    ai_data = np.array([p.arithmetic_intensity for p in points])
    gflops_data = np.array([p.gflops for p in points])
    
    # Color by K dimension (log scale) to show shape progression
    k_values = np.array([p.K for p in points])
    k_log = np.log2(k_values)
    k_norm = (k_log - k_log.min()) / (k_log.max() - k_log.min() + 1e-6)
    
    # Plot benchmark points
    scatter = ax.scatter(ai_data, gflops_data, c=k_values, cmap='viridis',
                         s=150, alpha=0.8, edgecolors='black', linewidths=1,
                         zorder=10, norm=plt.matplotlib.colors.LogNorm())
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('K Dimension', fontsize=12)
    
    # Annotate some key points
    # Find points at key transitions
    sorted_points = sorted(points, key=lambda p: p.arithmetic_intensity)
    n_annotate = min(10, len(sorted_points))
    for i in range(0, len(sorted_points), max(1, len(sorted_points) // n_annotate)):
        p = sorted_points[i]
        label = f"M={p.M}\nN={p.N}\nK={p.K}"
        ax.annotate(label, (p.arithmetic_intensity, p.gflops),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Mark the ridge point
    ax.axvline(x=ridge_point, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(ridge_point * 1.1, peak_tflops * 1000 * 0.5, 
           f'Ridge Point\nAI={ridge_point:.1f}',
           fontsize=10, color='purple', rotation=90, va='center')
    
    # Set axis labels and limits
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set reasonable axis limits
    ax.set_xlim(0.5, max(ai_data) * 2)
    ax.set_ylim(10, peak_tflops * 1500)  # Allow some headroom
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Add performance annotations
    ax.text(0.02, 0.98, f'Peak Compute: {peak_tflops} TFLOPS\nPeak Bandwidth: {peak_bandwidth_gbps} GB/s',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create legend
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return fig


def plot_performance_heatmap(points: List[BenchmarkPoint],
                             output_file: Optional[str] = None):
    """
    Generate a heatmap of performance for fixed K values.
    """
    if not HAS_MPL:
        print("Error: matplotlib is required for plotting")
        return
    
    # Group by K
    k_values = sorted(set(p.K for p in points))
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5), 
                            squeeze=False, sharey=True)
    
    for idx, k in enumerate(k_values):
        ax = axes[0, idx]
        k_points = [p for p in points if p.K == k]
        
        # Create M x N grid
        m_values = sorted(set(p.M for p in k_points))
        n_values = sorted(set(p.N for p in k_points))
        
        perf_grid = np.zeros((len(m_values), len(n_values)))
        for p in k_points:
            mi = m_values.index(p.M)
            ni = n_values.index(p.N)
            perf_grid[mi, ni] = p.gflops
        
        im = ax.imshow(perf_grid, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(n_values)))
        ax.set_yticks(range(len(m_values)))
        ax.set_xticklabels(n_values, rotation=45)
        ax.set_yticklabels(m_values)
        ax.set_xlabel('N')
        ax.set_ylabel('M')
        ax.set_title(f'K={k}')
        
        plt.colorbar(im, ax=ax, label='GFLOPS')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_file}")
    
    plt.show()
    return fig


def plot_transition_analysis(points: List[BenchmarkPoint],
                             peak_tflops: float = 2500,
                             peak_bandwidth_gbps: float = 8000,
                             output_file: Optional[str] = None):
    """
    Plot the transition from memory-bound to compute-bound.
    
    Shows efficiency vs arithmetic intensity.
    """
    if not HAS_MPL:
        print("Error: matplotlib is required for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by arithmetic intensity
    sorted_points = sorted(points, key=lambda p: p.arithmetic_intensity)
    ai_values = np.array([p.arithmetic_intensity for p in sorted_points])
    
    # Compute efficiency (how close to roofline)
    ridge_point = peak_tflops * 1000 / peak_bandwidth_gbps
    
    # Theoretical max performance for each AI
    theoretical_max = compute_roofline(ai_values, peak_tflops, peak_bandwidth_gbps)
    achieved = np.array([p.gflops for p in sorted_points])
    efficiency = achieved / theoretical_max
    
    # Plot 1: Performance vs AI with roofline
    ax1 = axes[0]
    ax1.loglog(ai_values, achieved, 'bo-', markersize=6, label='Achieved', alpha=0.7)
    ax1.loglog(ai_values, theoretical_max, 'r--', linewidth=2, label='Roofline')
    ax1.axvline(x=ridge_point, color='green', linestyle=':', linewidth=2, 
               label=f'Ridge Point (AI={ridge_point:.1f})')
    ax1.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax1.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax1.set_title('Performance vs Arithmetic Intensity', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency vs AI
    ax2 = axes[1]
    ax2.semilogx(ai_values, efficiency * 100, 'go-', markersize=6, alpha=0.7)
    ax2.axvline(x=ridge_point, color='red', linestyle=':', linewidth=2,
               label=f'Ridge Point (AI={ridge_point:.1f})')
    ax2.axhline(y=100, color='blue', linestyle='--', alpha=0.5, label='100% Efficiency')
    ax2.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax2.set_ylabel('Roofline Efficiency (%)', fontsize=12)
    ax2.set_title('Efficiency vs Arithmetic Intensity', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 120)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Transition analysis saved to {output_file}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate Roofline Plots")
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--peak-tflops", type=float, default=2500,
                       help="Peak TFLOPS for the GPU")
    parser.add_argument("--peak-bandwidth", type=float, default=8000,
                       help="Peak memory bandwidth in GB/s")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for the plot")
    parser.add_argument("--title", type=str, default="GEMM Roofline Analysis",
                       help="Plot title")
    
    args = parser.parse_args()
    
    # Load results
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    points = load_results(args.results_file)
    print(f"Loaded {len(points)} benchmark points")
    
    # Generate plots
    output_base = args.output or Path(args.results_file).stem
    
    # Main roofline plot
    plot_roofline(
        points,
        peak_tflops=args.peak_tflops,
        peak_bandwidth_gbps=args.peak_bandwidth,
        output_file=f"{output_base}_roofline.png",
        title=args.title
    )
    
    # Transition analysis
    plot_transition_analysis(
        points,
        peak_tflops=args.peak_tflops,
        peak_bandwidth_gbps=args.peak_bandwidth,
        output_file=f"{output_base}_transition.png"
    )


if __name__ == "__main__":
    main()
