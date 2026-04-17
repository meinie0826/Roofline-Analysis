"""
Classic Roofline Plot - Williams et al. Style

Clean, professional visualization
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib required")
    sys.exit(1)


PEAK_BF16_TFLOPS = 4500
PEAK_BW_GBPS = 7700


def plot_roofline(results_file: str, output_file: str):
    """Generate clean roofline plot."""
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#e0e0e0', alpha=0.8)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.2, color='#f0f0f0', alpha=0.6)
    ax.set_axisbelow(True)
    
    # AI range
    ai = np.logspace(0.5, 4, 500)
    ridge = PEAK_BF16_TFLOPS * 1000 / PEAK_BW_GBPS
    gflops_peak = np.minimum(ai * PEAK_BW_GBPS, PEAK_BF16_TFLOPS * 1000)
    
    # Roofline curve
    ax.loglog(ai, gflops_peak, 'k-', linewidth=2.5, zorder=3)
    
    # Reference lines
    ax.loglog(ai[ai < ridge * 1.05], ai[ai < ridge * 1.05] * PEAK_BW_GBPS, 
              'b--', linewidth=1, alpha=0.4, zorder=2)
    ax.loglog(ai, np.full_like(ai, PEAK_BF16_TFLOPS * 1000), 
              'r:', linewidth=1, alpha=0.4, zorder=2)
    
    # Ridge point
    ax.scatter([ridge], [PEAK_BF16_TFLOPS * 1000], c='black', s=80, 
               marker='D', zorder=5, edgecolors='white', linewidths=1.5)
    
    # Load and plot data
    max_tflops = 0
    if results_file and Path(results_file).exists():
        with open(results_file) as f:
            data = json.load(f).get('results', [])
        
        max_tflops = max(r['tflops'] for r in data) if data else 0
        
        cublas = [(r['arithmetic_intensity'], r['gflops'], r['M'], r['N']) 
                  for r in data if r['backend'] == 'cuBLAS']
        dg = [(r['arithmetic_intensity'], r['gflops'], r['M'], r['N']) 
              for r in data if r['backend'] == 'DeepGEMM']
        
        # cuBLAS - empty circles
        if cublas:
            ax.scatter([x[0] for x in cublas], [x[1] for x in cublas],
                      facecolors='none', edgecolors='#c0392b',
                      s=50, marker='o', linewidths=1.5, alpha=0.7,
                      label='cuBLAS', zorder=10)
        
        # DeepGEMM - filled triangles
        if dg:
            ax.scatter([x[0] for x in dg], [x[1] for x in dg],
                      c='#2980b9', s=50, marker='^', alpha=0.7,
                      edgecolors='white', linewidths=0.5,
                      label='DeepGEMM', zorder=10)
        
        # Shape labels below X-axis (inside plot area)
        all_pts = [(r['arithmetic_intensity'], r['M'], r['N'], r['tflops']) for r in data]
        if all_pts:
            all_pts.sort(key=lambda x: x[0])
            n = len(all_pts)
            
            # Pick 7 well-distributed points
            indices = [0, n//5, n//3, n//2, 2*n//3, 4*n//5, n-1]
            indices = sorted(set(indices))[:7]
            
            for idx in indices:
                ai_v, m, n_v, _ = all_pts[idx]
                # Vertical dotted line
                ax.axvline(x=ai_v, color='#999999', linestyle=':', 
                          alpha=0.4, linewidth=0.7, zorder=1)
                # Label at bottom (inside plot)
                ax.text(ai_v, 1.2e3, f'{m}×{n_v}', 
                       ha='center', va='bottom', fontsize=7, alpha=0.55,
                       rotation=30)
    
    # Region fills
    ai_mem = np.logspace(0.5, np.log10(ridge), 100)
    ax.fill_between(ai_mem, 0, ai_mem * PEAK_BW_GBPS, color='#3498db', alpha=0.03)
    ai_comp = np.logspace(np.log10(ridge), 4, 100)
    ax.fill_between(ai_comp, 0, PEAK_BF16_TFLOPS * 1000, color='#e74c3c', alpha=0.03)
    
    # Region labels - horizontal (not rotated)
    ax.text(15, 1e5, 'Memory\nBound', fontsize=11, color='#2980b9', 
            fontweight='bold', ha='center', va='center')
    ax.text(3000, 2.5e6, 'Compute\nBound', fontsize=11, color='#c0392b', 
            fontweight='bold', ha='center', va='center')
    
    # Ridge annotation
    ax.annotate(f'Ridge\nAI={int(ridge)}', 
                xy=(ridge, PEAK_BF16_TFLOPS * 1000),
                xytext=(ridge * 2.5, PEAK_BF16_TFLOPS * 700),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.9))
    
    # Axis labels
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=11, fontweight='bold')
    ax.set_title('NVIDIA B300 GEMM Roofline Model', fontsize=13, fontweight='bold', pad=8)
    
    # Limits with headroom
    y_max = max(PEAK_BF16_TFLOPS * 1000 * 1.4, (max_tflops * 1000) * 1.6)
    ax.set_xlim(3, 1e4)
    ax.set_ylim(1e3, y_max)
    
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # Info box
    ax.text(0.02, 0.97, 
            f"B300 (Blackwell Ultra)\n"
            f"Peak: {PEAK_BF16_TFLOPS:,} TF\n"
            f"BW:   {PEAK_BW_GBPS:,} GB/s\n"
            f"Ridge: {int(ridge)} FLOP/B",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                     edgecolor='#aaa', alpha=0.95, linewidth=1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="roofline.png")
    args = parser.parse_args()
    plot_roofline(args.results, args.output)


if __name__ == "__main__":
    main()
