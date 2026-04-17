"""
B300 Correct Roofline Plot Generator

B300 Specifications (Blackwell Ultra):
- BF16 Tensor Core: 4,500 TFLOPS = 4.5 PFLOPS
- FP8 Tensor Core: 9,000 TFLOPS = 9 PFLOPS  
- FP4 Tensor Core: 18,000 (sparse) / 14,000 (dense) PFLOPS
- Memory Bandwidth: 7,700 GB/s = 7.7 TB/s
- Memory: 270 GB HBM3e
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Error: matplotlib required")
    sys.exit(1)


B300_SPECS = {
    "name": "NVIDIA B300",
    "peak_bf16_tflops": 4500,
    "peak_fp8_tflops": 9000,
    "peak_fp4_sparse_tflops": 18000,
    "peak_fp4_dense_tflops": 14000,
    "peak_bandwidth_gbps": 7700,
    "memory_gb": 270,
}


def compute_roofline(ai: np.ndarray, peak_tflops: float, peak_bandwidth_gbps: float) -> dict:
    """Compute roofline curve."""
    ridge = peak_tflops * 1000 / peak_bandwidth_gbps
    gflops = np.minimum(ai * peak_bandwidth_gbps, peak_tflops * 1000)
    return {'gflops': gflops, 'ridge': ridge}


def plot_roofline_with_results(results_file: Optional[str] = None,
                                output_file: Optional[str] = None):
    """Generate B300 roofline plot with benchmark results."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    ai_range = np.logspace(-1, 5, 1000)
    
    # BF16 roofline
    roofline_bf16 = compute_roofline(ai_range, B300_SPECS['peak_bf16_tflops'],
                                     B300_SPECS['peak_bandwidth_gbps'])
    ax.loglog(ai_range, roofline_bf16['gflops'], 'b-', linewidth=3,
              label=f"BF16: {B300_SPECS['peak_bf16_tflops']:,} TFLOPS")
    
    # FP8 roofline
    roofline_fp8 = compute_roofline(ai_range, B300_SPECS['peak_fp8_tflops'],
                                    B300_SPECS['peak_bandwidth_gbps'])
    ax.loglog(ai_range, roofline_fp8['gflops'], 'g-', linewidth=3,
              label=f"FP8: {B300_SPECS['peak_fp8_tflops']:,} TFLOPS")
    
    # Memory-bound region
    ai_mem = np.logspace(-1, np.log10(max(roofline_bf16['ridge'], roofline_fp8['ridge'])), 200)
    ax.fill_between(ai_mem, 0, ai_mem * B300_SPECS['peak_bandwidth_gbps'],
                   alpha=0.12, color='blue', label='Memory-Bound')
    
    # Compute-bound regions
    ai_comp_bf16 = np.logspace(np.log10(roofline_bf16['ridge']), 5, 200)
    ax.fill_between(ai_comp_bf16, 0, B300_SPECS['peak_bf16_tflops'] * 1000,
                   alpha=0.12, color='red')
    
    ai_comp_fp8 = np.logspace(np.log10(roofline_fp8['ridge']), 5, 200)
    ax.fill_between(ai_comp_fp8, 0, B300_SPECS['peak_fp8_tflops'] * 1000,
                   alpha=0.12, color='green', label='Compute-Bound')
    
    # Ridge lines
    ax.axvline(x=roofline_bf16['ridge'], color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=roofline_fp8['ridge'], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Ridge annotations
    ax.text(roofline_bf16['ridge'] * 1.15, B300_SPECS['peak_bf16_tflops'] * 500,
           f"BF16 Ridge\nAI = {roofline_bf16['ridge']:.0f}",
           fontsize=11, color='blue', fontweight='bold')
    
    ax.text(roofline_fp8['ridge'] * 1.15, B300_SPECS['peak_fp8_tflops'] * 500,
           f"FP8 Ridge\nAI = {roofline_fp8['ridge']:.0f}",
           fontsize=11, color='green', fontweight='bold')
    
    # Plot benchmark results
    if results_file and Path(results_file).exists():
        data = json.load(open(results_file))
        results = data.get('results', [])
        
        print(f"\nLoaded {len(results)} benchmark results")
        
        # Separate by backend
        backends = {}
        for r in results:
            key = r['backend']
            if key not in backends:
                backends[key] = {'ai': [], 'gflops': [], 'tflops': [], 'label': []}
            backends[key]['ai'].append(r['arithmetic_intensity'])
            backends[key]['gflops'].append(r['gflops'])
            backends[key]['tflops'].append(r['tflops'])
            
            # Create label with shape info
            shape_type = ""
            if r['M'] == r['N'] == r['K']:
                shape_type = "■"
            elif r['N'] == 4 * r['K']:  # FFN up
                shape_type = "↑"
            elif r['K'] == 4 * r['N']:  # FFN down
                shape_type = "↓"
            elif r['N'] != r['M']:  # Attention-like
                shape_type = "∘"
            
            label = f"{shape_type} {r['M']}x{r['N']}x{r['K']}"
            backends[key]['label'].append(label)
        
        colors = {'cuBLAS': '#e74c3c', 'DeepGEMM': '#3498db'}
        markers = {'cuBLAS': 'o', 'DeepGEMM': 's'}
        
        for backend, data_points in backends.items():
            ai = np.array(data_points['ai'])
            gflops = np.array(data_points['gflops'])
            labels = data_points['label']
            
            scatter = ax.scatter(ai, gflops, c=colors.get(backend, 'gray'),
                                s=120, marker=markers.get(backend, 'o'),
                                alpha=0.75, edgecolors='black', linewidths=1.5,
                                label=backend, zorder=10)
            
            # Smart annotation: annotate points at different AI regions
            sorted_idx = np.argsort(ai)
            n_annotate = min(15, len(sorted_idx))
            
            # Select points spread across AI range
            selected = []
            ai_bins = np.percentile(ai[sorted_idx], np.linspace(0, 100, n_annotate + 2)[1:-1])
            
            for target_ai in ai_bins:
                # Find closest point to target AI
                idx = np.argmin(np.abs(ai - target_ai))
                if idx not in selected:
                    selected.append(idx)
            
            for idx in selected:
                label = labels[idx].strip()
                ax.annotate(label, (ai[idx], gflops[idx]),
                           xytext=(6, 6), textcoords='offset points',
                           fontsize=7, alpha=0.65,
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', alpha=0.75))
        
        # Print summary statistics
        print("\nPerformance Summary:")
        print("-" * 70)
        for backend, data_points in backends.items():
            tflops = np.array(data_points['tflops'])
            print(f"\n{backend}:")
            print(f"  Samples: {len(tflops)}")
            print(f"  Max TFLOPS: {max(tflops):.1f}")
            print(f"  Avg TFLOPS: {np.mean(tflops):.1f}")
            print(f"  Avg efficiency: {np.mean(tflops) / B300_SPECS['peak_bf16_tflops'] * 100:.2f}%")
    
    # Axis settings
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title('NVIDIA B300 GPU Roofline Model\nLLM GEMM Performance Analysis', 
                fontsize=18, fontweight='bold')
    
    ax.set_xlim(1, 1e5)
    ax.set_ylim(100, B300_SPECS['peak_fp8_tflops'] * 1500)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    # Info box
    info_text = (f"NVIDIA B300 (Blackwell Ultra)\n"
                f"{'─' * 28}\n"
                f"Peak BF16: {B300_SPECS['peak_bf16_tflops']:,} TFLOPS\n"
                f"Peak FP8:  {B300_SPECS['peak_fp8_tflops']:,} TFLOPS\n"
                f"Peak FP4:  {B300_SPECS['peak_fp4_dense_tflops']:,} TFLOPS\n"
                f"Memory BW: {B300_SPECS['peak_bandwidth_gbps']:,} GB/s\n"
                f"Memory:    {B300_SPECS['memory_gb']} GB HBM3e\n"
                f"{'─' * 28}\n"
                f"BF16 Ridge: {roofline_bf16['ridge']:.0f} FLOP/B\n"
                f"FP8 Ridge:  {roofline_fp8['ridge']:.0f} FLOP/B")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Region labels
    ax.text(0.10, 0.78, 'Memory\nBound', transform=ax.transAxes,
           fontsize=15, ha='center', color='blue', alpha=0.6, fontweight='bold')
    ax.text(0.85, 0.78, 'Compute\nBound', transform=ax.transAxes,
           fontsize=15, ha='center', color='red', alpha=0.6, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    
    plt.show()
    return fig


def print_specs():
    """Print B300 specifications."""
    ridge_bf16 = B300_SPECS['peak_bf16_tflops'] * 1000 / B300_SPECS['peak_bandwidth_gbps']
    ridge_fp8 = B300_SPECS['peak_fp8_tflops'] * 1000 / B300_SPECS['peak_bandwidth_gbps']
    
    print("\n" + "=" * 60)
    print("NVIDIA B300 (Blackwell Ultra) Specifications")
    print("=" * 60)
    for key, value in B300_SPECS.items():
        if isinstance(value, int) and value > 1000:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    print("\nRidge Points:")
    print(f"  BF16: {ridge_bf16:.0f} FLOP/Byte")
    print(f"  FP8:  {ridge_fp8:.0f} FLOP/Byte")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="B300 Roofline Plot")
    parser.add_argument("--results", type=str, default=None, help="Benchmark results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output plot file")
    parser.add_argument("--specs", action="store_true", help="Print B300 specs")
    
    args = parser.parse_args()
    
    if args.specs:
        print_specs()
        return
    
    print_specs()
    
    output_file = args.output or "b300_llm_roofline.png"
    plot_roofline_with_results(args.results, output_file)


if __name__ == "__main__":
    main()
