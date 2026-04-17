"""
B300 Roofline Plot - Adjusted for Better Visibility

Since actual kernel efficiency is typically 0.3%-5%, we need to
adjust the plot scale to make data points visible.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Error: matplotlib required")
    sys.exit(1)


# B300 Correct Specifications
B300_SPECS = {
    "name": "NVIDIA B300",
    "peak_bf16_tflops": 4500,
    "peak_fp8_tflops": 9000,
    "peak_bandwidth_gbps": 7700,
    "memory_gb": 270,
}


def plot_llm_roofline(results_file: Optional[str] = None,
                      output_file: Optional[str] = None):
    """Generate roofline plot optimized for LLM GEMM performance."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # AI range for roofline
    ai_range = np.logspace(-1, 5, 1000)
    
    # Peak BW line (diagonal)
    peak_bw_gbps = B300_SPECS['peak_bandwidth_gbps']
    
    # Compute rooflines at different efficiencies
    efficiencies = [1.0, 0.1, 0.01, 0.005, 0.001]  # 100%, 10%, 1%, 0.5%, 0.1%
    
    colors = ['#e74c3c', '#e67e22', '#27ae60', '#3498db', '#9b59b6']
    
    for eff, color in zip(efficiencies, colors):
        # Effective peak TFLOPS
        effective_peak = B300_SPECS['peak_bf16_tflops'] * eff
        
        # Ridge point
        ridge = effective_peak * 1000 / peak_bw_gbps
        
        # Roofline GFLOPS
        gflops = np.minimum(ai_range * peak_bw_gbps, effective_peak * 1000)
        
        label = f"{eff*100:.1f}% ({effective_peak:.0f} TF)"
        ax.loglog(ai_range, gflops, '-', color=color, linewidth=2, 
                  alpha=0.7 if eff < 1.0 else 1.0,
                  label=label)
        
        # Mark ridge point
        if eff >= 0.01:
            ax.axvline(x=ridge, color=color, linestyle=':', alpha=0.5, linewidth=1)
    
    # Memory-bound region
    ai_mem = np.logspace(-1, np.log10(500), 200)
    ax.fill_between(ai_mem, 0, ai_mem * peak_bw_gbps,
                   alpha=0.08, color='blue', label='Memory-Bound')
    
    # Plot benchmark results
    if results_file and Path(results_file).exists():
        data = json.load(open(results_file))
        results = data.get('results', [])
        
        print(f"\nLoaded {len(results)} benchmark results")
        
        backends = {}
        for r in results:
            key = r['backend']
            if key not in backends:
                backends[key] = {'ai': [], 'gflops': [], 'tflops': [], 
                                'shape': [], 'eff': []}
            
            backends[key]['ai'].append(r['arithmetic_intensity'])
            backends[key]['gflops'].append(r['gflops'])
            backends[key]['tflops'].append(r['tflops'])
            backends[key]['eff'].append(r.get('compute_efficiency', r['tflops']/4500))
            
            # Shape label
            m, n, k = r['M'], r['N'], r['K']
            shape_label = f"{m}×{n}×{k}"
            backends[key]['shape'].append(shape_label)
        
        colors_scatter = {'cuBLAS': '#c0392b', 'DeepGEMM': '#2980b9'}
        markers = {'cuBLAS': 'o', 'DeepGEMM': 's'}
        
        for backend, data_points in backends.items():
            ai = np.array(data_points['ai'])
            gflops = np.array(data_points['gflops'])
            tflops = np.array(data_points['tflops'])
            shapes = data_points['shape']
            
            scatter = ax.scatter(ai, gflops, 
                                c=colors_scatter.get(backend, 'gray'),
                                s=200, marker=markers.get(backend, 'o'),
                                alpha=0.85, edgecolors='black', linewidths=2,
                                label=f"{backend} ({len(ai)} samples)", 
                                zorder=10)
            
            # Annotate outliers and key points
            sorted_idx = np.argsort(ai)
            
            # Annotate: min, max, median AI, max TFLOPS
            to_annotate = set()
            
            # Min and max AI
            to_annotate.add(np.argmin(ai))
            to_annotate.add(np.argmax(ai))
            
            # Max TFLOPS
            to_annotate.add(np.argmax(tflops))
            
            # Some evenly distributed points
            n_select = min(12, len(ai))
            step = max(1, len(sorted_idx) // n_select)
            for i in range(0, len(sorted_idx), step):
                to_annotate.add(sorted_idx[i])
            
            for idx in to_annotate:
                label = shapes[idx]
                ax.annotate(label, (ai[idx], gflops[idx]),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white', alpha=0.8))
        
        # Print statistics
        print("\n" + "=" * 70)
        print("Performance Analysis:")
        print("=" * 70)
        for backend, dp in backends.items():
            tflops_arr = np.array(dp['tflops'])
            eff_arr = np.array(dp['eff'])
            print(f"\n{backend}:")
            print(f"  Max TFLOPS:    {max(tflops_arr):>8.1f}  ({max(eff_arr)*100:>6.2f}%)")
            print(f"  Avg TFLOPS:    {np.mean(tflops_arr):>8.1f}  ({np.mean(eff_arr)*100:>6.2f}%)")
            print(f"  Median TFLOPS: {np.median(tflops_arr):>8.1f}")
            print(f"  Min TFLOPS:    {min(tflops_arr):>8.1f}  ({min(eff_arr)*100:>6.2f}%)")
        
        # Overall comparison
        if len(backends) >= 2:
            cublas_tflops = np.array(backends.get('cuBLAS', {}).get('tflops', [0]))
            deepgemm_tflops = np.array(backends.get('DeepGEMM', {}).get('tflops', [0]))
            
            if len(cublas_tflops) > 0 and len(deepgemm_tflops) > 0:
                speedup = np.mean(deepgemm_tflops) / np.mean(cublas_tflops)
                print(f"\nDeepGEMM vs cuBLAS: {speedup:.2f}x average speedup")
    
    # Axis settings
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    ax.set_title('NVIDIA B300 GEMM Roofline Analysis\nLLM Inference Shapes', 
                fontsize=18, fontweight='bold')
    
    # Set reasonable Y-axis range based on data
    ax.set_xlim(1, 1e5)
    
    # Get Y-axis range that makes sense for actual data
    # Typical: 1-100 GFLOPS for memory-bound, up to 60000 for compute-bound
    ax.set_ylim(10, 1e6)  # 10 GFLOPS to 1,000,000 GFLOPS (1000 TFLOPS)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    
    # Info box
    info_text = (f"NVIDIA B300 (Blackwell Ultra)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Peak BF16:    {B300_SPECS['peak_bf16_tflops']:,} TFLOPS\n"
                f"Peak FP8:     {B300_SPECS['peak_fp8_tflops']:,} TFLOPS\n"
                f"Memory BW:    {B300_SPECS['peak_bandwidth_gbps']:,} GB/s\n"
                f"Memory:       {B300_SPECS['memory_gb']} GB HBM3e\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Roofline curves show different\nefficiency levels from 0.1% to 100%")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Region labels
    ax.text(0.08, 0.70, 'Memory\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='blue', alpha=0.5, fontweight='bold')
    ax.text(0.82, 0.70, 'Compute\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='red', alpha=0.5, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="B300 LLM Roofline Plot")
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    output_file = args.output or "b300_llm_roofline_visible.png"
    plot_llm_roofline(args.results, output_file)


if __name__ == "__main__":
    main()
