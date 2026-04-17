"""
B300 Correct Roofline Plot Generator

B300 Specifications (from NVIDIA/Exxact):
- BF16 Tensor Core: 4.5 PFLOPS = 4500 TFLOPS
- FP8 Tensor Core: 9 PFLOPS = 9000 TFLOPS  
- FP4 Tensor Core: 18 (sparse) / 14 (dense) PFLOPS
- Memory Bandwidth: 7.7 TB/s = 7700 GB/s
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


# B300 Correct Specifications
B300_SPECS = {
    "name": "NVIDIA B300",
    "peak_bf16_tflops": 4500,      # 4.5 PFLOPS
    "peak_fp8_tflops": 9000,      # 9 PFLOPS
    "peak_fp4_sparse_tflops": 18000,  # 18 PFLOPS (sparse)
    "peak_fp4_dense_tflops": 14000,   # 14 PFLOPS (dense)
    "peak_bandwidth_gbps": 7700,  # 7.7 TB/s
    "memory_gb": 270,
}


def compute_roofline(ai: np.ndarray, peak_tflops: float, peak_bandwidth_gbps: float) -> dict:
    """Compute roofline curve."""
    # Ridge point
    ridge = peak_tflops * 1000 / peak_bandwidth_gbps
    
    # Roofline GFLOPS
    gflops = np.minimum(ai * peak_bandwidth_gbps, peak_tflops * 1000)
    
    return {
        'gflops': gflops,
        'ridge': ridge,
    }


def plot_roofline(results_file: Optional[str] = None,
                   output_file: Optional[str] = None,
                   show_theoretical: bool = True):
    """Generate B300 roofline plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    
    # AI range
    ai_range = np.logspace(-1, 5, 1000)
    
    # Plot BF16 roofline
    roofline_bf16 = compute_roofline(ai_range, B300_SPECS['peak_bf16_tflops'], 
                                     B300_SPECS['peak_bandwidth_gbps'])
    ax.loglog(ai_range, roofline_bf16['gflops'], 'b-', linewidth=3,
              label=f"BF16 Peak: {B300_SPECS['peak_bf16_tflops']} TFLOPS")
    
    # Plot FP8 roofline
    roofline_fp8 = compute_roofline(ai_range, B300_SPECS['peak_fp8_tflops'],
                                    B300_SPECS['peak_bandwidth_gbps'])
    ax.loglog(ai_range, roofline_fp8['gflops'], 'g-', linewidth=3,
              label=f"FP8 Peak: {B300_SPECS['peak_fp8_tflops']} TFLOPS")
    
    # Fill memory-bound region
    ai_mem = np.logspace(-1, np.log10(max(roofline_bf16['ridge'], roofline_fp8['ridge'])), 200)
    ax.fill_between(ai_mem, 0, ai_mem * B300_SPECS['peak_bandwidth_gbps'],
                   alpha=0.15, color='blue', label='Memory-Bound Region')
    
    # Fill compute-bound regions
    ai_comp_bf16 = np.logspace(np.log10(roofline_bf16['ridge']), 5, 200)
    ax.fill_between(ai_comp_bf16, 0, B300_SPECS['peak_bf16_tflops'] * 1000,
                   alpha=0.15, color='red')
    
    ai_comp_fp8 = np.logspace(np.log10(roofline_fp8['ridge']), 5, 200)
    ax.fill_between(ai_comp_fp8, 0, B300_SPECS['peak_fp8_tflops'] * 1000,
                   alpha=0.15, color='green', label='Compute-Bound Region')
    
    # Ridge point markers
    ax.axvline(x=roofline_bf16['ridge'], color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=roofline_fp8['ridge'], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Annotate ridge points
    ax.text(roofline_bf16['ridge'] * 1.2, B300_SPECS['peak_bf16_tflops'] * 600,
           f"BF16 Ridge\nAI = {roofline_bf16['ridge']:.0f}",
           fontsize=11, color='blue', fontweight='bold')
    
    ax.text(roofline_fp8['ridge'] * 1.2, B300_SPECS['peak_fp8_tflops'] * 600,
           f"FP8 Ridge\nAI = {roofline_fp8['ridge']:.0f}",
           fontsize=11, color='green', fontweight='bold')
    
    # Load and plot benchmark results if provided
    if results_file and Path(results_file).exists():
        data = json.load(open(results_file))
        results = data.get('results', [])
        
        backends = {}
        for r in results:
            key = r['backend']
            if key not in backends:
                backends[key] = {'ai': [], 'gflops': [], 'shape': []}
            backends[key]['ai'].append(r['arithmetic_intensity'])
            backends[key]['gflops'].append(r['gflops'])
            backends[key]['shape'].append(f"{r['M']}x{r['N']}x{r['K']}")
        
        colors = {'cuBLAS': '#e74c3c', 'DeepGEMM': '#3498db'}
        markers = {'cuBLAS': 'o', 'DeepGEMM': 's'}
        
        for backend, data_points in backends.items():
            ai = np.array(data_points['ai'])
            gflops = np.array(data_points['gflops'])
            
            scatter = ax.scatter(ai, gflops, c=colors.get(backend, 'gray'),
                                s=180, marker=markers.get(backend, 'o'),
                                alpha=0.85, edgecolors='black', linewidths=2,
                                label=backend, zorder=10)
            
            # Annotate selected points
            sorted_idx = np.argsort(ai)
            n_annotate = min(8, len(sorted_idx))
            step = max(1, len(sorted_idx) // n_annotate)
            
            for i in range(0, len(sorted_idx), step):
                idx = sorted_idx[i]
                label = data_points['shape'][idx].replace('x', '\n')
                ax.annotate(label, (ai[idx], gflops[idx]),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=9, alpha=0.75,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.8))
    
    # Axis settings
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax.set_title('NVIDIA B300 GPU Roofline Model\n(Blackwell Ultra Architecture)', 
                fontsize=18, fontweight='bold')
    
    ax.set_xlim(1, 1e5)
    ax.set_ylim(100, B300_SPECS['peak_fp8_tflops'] * 1500)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    # Info box with correct specs
    info_text = (f"NVIDIA B300 (Blackwell Ultra)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Peak BF16: {B300_SPECS['peak_bf16_tflops']:,} TFLOPS\n"
                f"Peak FP8: {B300_SPECS['peak_fp8_tflops']:,} TFLOPS\n"
                f"Peak FP4: {B300_SPECS['peak_fp4_dense_tflops']:,} TFLOPS\n"
                f"Memory BW: {B300_SPECS['peak_bandwidth_gbps']:,} GB/s\n"
                f"Memory: {B300_SPECS['memory_gb']} GB HBM3e\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"BF16 Ridge: {roofline_bf16['ridge']:.0f} FLOP/B\n"
                f"FP8 Ridge: {roofline_fp8['ridge']:.0f} FLOP/B")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Region labels
    ax.text(0.12, 0.82, 'Memory\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='blue', alpha=0.7, fontweight='bold')
    ax.text(0.82, 0.82, 'Compute\nBound', transform=ax.transAxes,
           fontsize=14, ha='center', color='red', alpha=0.7, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()
    return fig


def print_specs():
    """Print B300 specifications."""
    print("\n" + "=" * 60)
    print("NVIDIA B300 (Blackwell Ultra) Specifications")
    print("=" * 60)
    
    for key, value in B300_SPECS.items():
        if 'tflops' in key or 'gbps' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    ridge_bf16 = B300_SPECS['peak_bf16_tflops'] * 1000 / B300_SPECS['peak_bandwidth_gbps']
    ridge_fp8 = B300_SPECS['peak_fp8_tflops'] * 1000 / B300_SPECS['peak_bandwidth_gbps']
    
    print("\nRidge Points:")
    print(f"  BF16 Ridge: {ridge_bf16:.0f} FLOP/Byte")
    print(f"  FP8 Ridge: {ridge_fp8:.0f} FLOP/Byte")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="B300 Roofline Plot")
    parser.add_argument("--results", type=str, default=None, help="Benchmark results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output plot file")
    parser.add_argument("--specs", action="store_true", help="Print B300 specifications")
    
    args = parser.parse_args()
    
    if args.specs:
        print_specs()
        return
    
    print_specs()
    
    output_file = args.output or "b300_roofline.png"
    plot_roofline(args.results, output_file)


if __name__ == "__main__":
    main()
