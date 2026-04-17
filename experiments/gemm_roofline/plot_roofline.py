"""
Roofline Plot Generator

Generates roofline plots from benchmark results, showing the transition
from memory-bound to compute-bound regions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Error: matplotlib required")
    sys.exit(1)


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_roofline(ai: np.ndarray,
                     peak_tflops: float,
                     peak_bandwidth_gbps: float) -> dict:
    """
    Compute roofline curve.
    
    Args:
        ai: Arithmetic intensity array (FLOPs/Byte)
        peak_tflops: Peak compute throughput (TFLOPS)
        peak_bandwidth_gbps: Peak memory bandwidth (GB/s)
    
    Returns:
        dict with roofline GFLOPS values and ridge point
    """
    # Ridge point: where AI * BW = Peak TFLOPS
    ridge_point = peak_tflops * 1000 / peak_bandwidth_gbps
    
    # Roofline: min(AI * BW, Peak TFLOPS)
    gflops = np.minimum(ai * peak_bandwidth_gbps, peak_tflops * 1000)
    
    return {
        'gflops': gflops,
        'ridge_point': ridge_point,
    }


def plot_roofline(results_file: str,
                   output_file: Optional[str] = None,
                   title: str = "GEMM Roofline Analysis"):
    """Generate roofline plot from benchmark results."""
    
    data = load_results(results_file)
    results = data['results']
    gpu_specs = data.get('gpu_specs', {})
    
    # Get GPU specs
    peak_bf16_tflops = gpu_specs.get('peak_bf16_tflops', 1250)
    peak_bandwidth_gbps = gpu_specs.get('peak_bandwidth_gbps', 8000)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Generate roofline curve
    ai_range = np.logspace(-1, 4, 1000)
    roofline = compute_roofline(ai_range, peak_bf16_tflops, peak_bandwidth_gbps)
    
    # Plot roofline
    ax.loglog(ai_range, roofline['gflops'], 'k-', linewidth=3, 
              label=f'Peak BF16: {peak_bf16_tflops} TFLOPS')
    
    # Fill memory-bound region (light blue)
    ai_mem = np.logspace(-1, np.log10(roofline['ridge_point']), 200)
    ax.fill_between(ai_mem, 0, ai_mem * peak_bandwidth_gbps, 
                   alpha=0.15, color='blue')
    
    # Fill compute-bound region (light red)
    ai_comp = np.logspace(np.log10(roofline['ridge_point']), 4, 200)
    ax.fill_between(ai_comp, 0, peak_bf16_tflops * 1000, 
                   alpha=0.15, color='red')
    
    # Ridge point marker
    ax.axvline(x=roofline['ridge_point'], color='gray', linestyle='--', 
               alpha=0.7, linewidth=1.5)
    ax.text(roofline['ridge_point'] * 1.3, peak_bf16_tflops * 500,
           f'Ridge Point\nAI = {roofline["ridge_point"]:.0f}',
           fontsize=11, color='gray')
    
    # Separate points by backend
    backends = {}
    for r in results:
        key = r['backend']
        if key not in backends:
            backends[key] = {'ai': [], 'gflops': [], 'tflops': [], 'shape': []}
        backends[key]['ai'].append(r['arithmetic_intensity'])
        backends[key]['gflops'].append(r['gflops'])
        backends[key]['tflops'].append(r['tflops'])
        backends[key]['shape'].append(f"{r['M']}x{r['N']}x{r['K']}")
    
    # Color scheme for backends
    colors = {
        'cuBLAS': '#e74c3c',
        'DeepGEMM': '#3498db',
    }
    markers = {
        'cuBLAS': 'o',
        'DeepGEMM': 's',
    }
    
    # Plot benchmark points
    for backend, data_points in backends.items():
        ai = np.array(data_points['ai'])
        gflops = np.array(data_points['gflops'])
        tflops = np.array(data_points['tflops'])
        shapes = data_points['shape']
        
        color = colors.get(backend, 'gray')
        marker = markers.get(backend, 'o')
        
        scatter = ax.scatter(ai, gflops, c=color, s=150, marker=marker,
                            alpha=0.8, edgecolors='black', linewidths=1.5,
                            label=backend, zorder=10)
        
        # Annotate points
        # Sort by AI for better annotation placement
        sorted_indices = np.argsort(ai)
        n_annotate = min(10, len(sorted_indices))
        step = max(1, len(sorted_indices) // n_annotate)
        
        for i in range(0, len(sorted_indices), step):
            idx = sorted_indices[i]
            label = shapes[idx].replace('x', '\n')
            ax.annotate(label, (ai[idx], gflops[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', alpha=0.7))
    
    # Add horizontal line for peak TFLOPS
    ax.axhline(y=peak_bf16_tflops * 1000, color='red', linestyle=':', 
               alpha=0.5, linewidth=1)
    
    # Add diagonal line for memory bandwidth
    ai_bw = np.logspace(-1, np.log10(roofline['ridge_point']), 100)
    ax.plot(ai_bw, ai_bw * peak_bandwidth_gbps, 'b:', alpha=0.5, linewidth=1)
    
    # Set axis properties
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xlim(1, max(ai_range) * 1.5)
    ax.set_ylim(100, peak_bf16_tflops * 1500)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add info box
    info_text = (f'Peak BF16: {peak_bf16_tflops} TFLOPS\n'
                f'Peak BW: {peak_bandwidth_gbps} GB/s\n'
                f'Ridge: {roofline["ridge_point"]:.0f} FLOP/B')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add region labels
    ax.text(0.15, 0.85, 'Memory\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='blue', alpha=0.6)
    ax.text(0.75, 0.85, 'Compute\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='red', alpha=0.6)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return fig


def plot_performance_vs_shape(results_file: str,
                               output_file: Optional[str] = None):
    """Plot performance vs matrix size."""
    
    data = load_results(results_file)
    results = data['results']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Separate by backend
    backends = {}
    for r in results:
        key = r['backend']
        if key not in backends:
            backends[key] = {'sizes': [], 'tflops': []}
        
        # Calculate matrix size (total elements)
        size = r['M'] * r['N'] * r['K']
        backends[key]['sizes'].append(size)
        backends[key]['tflops'].append(r['tflops'])
    
    colors = {'cuBLAS': '#e74c3c', 'DeepGEMM': '#3498db'}
    
    for backend, data_points in backends.items():
        sizes = np.array(data_points['sizes'])
        tflops = np.array(data_points['tflops'])
        
        # Sort by size
        sorted_idx = np.argsort(sizes)
        sizes = sizes[sorted_idx]
        tflops = tflops[sorted_idx]
        
        ax.semilogx(sizes, tflops, 'o-', color=colors.get(backend, 'gray'),
                   markersize=10, linewidth=2, label=backend, alpha=0.8)
    
    ax.set_xlabel('Matrix Size (M×N×K)', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title('GEMM Performance vs Matrix Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate roofline plots")
    parser.add_argument("results_file", type=str, help="Path to results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--title", type=str, default="GEMM Roofline Analysis")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    print(f"Loading results from {args.results_file}")
    
    output_base = args.output or Path(args.results_file).stem
    
    print("\nGenerating roofline plot...")
    plot_roofline(args.results_file, f"{output_base}_roofline.png", args.title)
    
    print("\nGenerating performance comparison plot...")
    plot_performance_vs_shape(args.results_file, f"{output_base}_perf.png")


if __name__ == "__main__":
    main()
