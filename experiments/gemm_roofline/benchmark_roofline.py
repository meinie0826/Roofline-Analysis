"""
Roofline Benchmark Script for GEMM

This script benchmarks GEMM performance across different shapes and calculates
arithmetic intensity to observe the transition from memory-bound to compute-bound.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Check for CUDA
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch is required for this benchmark")
    sys.exit(1)


@dataclass
class GEMMResult:
    """Result from a single GEMM benchmark."""
    M: int
    N: int
    K: int
    time_ms: float
    gflops: float
    bytes_accessed: int
    arithmetic_intensity: float  # FLOPs/Byte
    achieved_bandwidth_gbps: float
    theoretical_peak_tflops: float
    peak_bandwidth_gbps: float
    compute_efficiency: float  # Achieved / Theoretical peak
    memory_efficiency: float  # Achieved / Peak bandwidth


def calculate_arithmetic_intensity(M: int, N: int, K: int, 
                                   element_size: int = 2,  # FP16 = 2 bytes
                                   c_reads: bool = False) -> Tuple[int, float]:
    """
    Calculate arithmetic intensity for GEMM: C = A @ B
    
    Arithmetic Intensity = FLOPs / Bytes Accessed
    
    FLOPs = 2 * M * N * K (multiply-add for each output element)
    Bytes = 
        - A reads: M * K * element_size
        - B reads: K * N * element_size  
        - C writes: M * N * element_size
        - Optional C reads: M * N * element_size (for beta != 0)
    
    Returns:
        (total_bytes, arithmetic_intensity)
    """
    flops = 2 * M * N * K
    
    # Data accessed from global memory
    bytes_a = M * K * element_size
    bytes_b = K * N * element_size
    bytes_c_write = M * N * element_size
    bytes_c_read = M * N * element_size if c_reads else 0
    
    total_bytes = bytes_a + bytes_b + bytes_c_write + bytes_c_read
    arithmetic_intensity = flops / total_bytes
    
    return total_bytes, arithmetic_intensity


def get_gpu_specs() -> dict:
    """Get GPU specifications for theoretical peak calculations."""
    # Default specs - can be overridden via command line
    # B300 (Blackwell) specs
    return {
        "name": "B300",
        "peak_fp16_tflops": 2500,  # ~2.5 PFLOPS for FP16 Tensor Core
        "peak_fp32_tflops": 125,    # ~125 TFLOPS for FP32
        "peak_bandwidth_gbps": 8000,  # HBM3e ~8 TB/s
        "sm_count": 114,
        "max_smem_per_sm": 227 * 1024,  # 227 KB per SM
    }


def benchmark_shape(M: int, N: int, K: int,
                    warmup: int = 5,
                    iterations: int = 20,
                    device: str = "cuda",
                    dtype: torch.dtype = torch.float16) -> GEMMResult:
    """
    Benchmark a single GEMM shape using PyTorch matmul.
    
    Args:
        M, N, K: Matrix dimensions
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        device: Device to run on
        dtype: Data type (float16 for Tensor Cores)
    
    Returns:
        GEMMResult with performance metrics
    """
    torch.manual_seed(42)
    
    # Create tensors
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = torch.empty(M, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
    
    # Timed runs
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start.record()
        torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # Calculate metrics
    avg_time_ms = np.median(times)
    std_time_ms = np.std(times)
    
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    bytes_accessed, arithmetic_intensity = calculate_arithmetic_intensity(M, N, K)
    achieved_bandwidth_gbps = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    # GPU specs
    specs = get_gpu_specs()
    
    # Efficiency calculations
    # Use FP16 peak for FP16 inputs (assuming Tensor Core usage)
    peak_tflops = specs["peak_fp16_tflops"]
    peak_bandwidth = specs["peak_bandwidth_gbps"]
    
    compute_efficiency = gflops / 1000 / peak_tflops  # Convert to TFLOPS
    memory_efficiency = achieved_bandwidth_gbps / peak_bandwidth
    
    return GEMMResult(
        M=M, N=N, K=K,
        time_ms=avg_time_ms,
        gflops=gflops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=arithmetic_intensity,
        achieved_bandwidth_gbps=achieved_bandwidth_gbps,
        theoretical_peak_tflops=peak_tflops,
        peak_bandwidth_gbps=peak_bandwidth,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
    )


def generate_shapes(shape_type: str = "balanced") -> List[Tuple[int, int, int]]:
    """
    Generate a list of GEMM shapes for benchmarking.
    
    Args:
        shape_type: Type of shape progression
            - "balanced": M=N=K, doubling from 64 to 8192
            - "memory_heavy": Small K, varying M*N
            - "compute_heavy": Large K, small M*N
            - "layer_like": Shapes typical in transformer layers
    
    Returns:
        List of (M, N, K) tuples
    """
    shapes = []
    
    if shape_type == "balanced":
        # Square matrices, doubling sizes
        sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        for size in sizes:
            shapes.append((size, size, size))
    
    elif shape_type == "memory_heavy":
        # Small K -> low arithmetic intensity
        base_sizes = [128, 256, 512, 1024, 2048, 4096]
        k_values = [32, 64, 128]
        for size in base_sizes:
            for k in k_values:
                shapes.append((size, size, k))
    
    elif shape_type == "compute_heavy":
        # Large K -> high arithmetic intensity  
        mn_sizes = [64, 128, 256, 512, 1024]
        k_values = [256, 512, 1024, 2048, 4096, 8192]
        for mn in mn_sizes:
            for k in k_values:
                shapes.append((mn, mn, k))
    
    elif shape_type == "layer_like":
        # Transformer-like shapes (batch * seq_len, hidden_dim)
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_lens = [128, 256, 512, 1024, 2048]
        hidden_dim = 4096
        intermediate_dim = 14336  # FFN intermediate dimension
        
        for bs in batch_sizes:
            for seq in seq_lens:
                m = bs * seq
                # QKV projection
                shapes.append((m, hidden_dim * 3, hidden_dim))
                # Attention output
                shapes.append((m, hidden_dim, hidden_dim))
                # FFN up
                shapes.append((m, intermediate_dim, hidden_dim))
                # FFN down
                shapes.append((m, hidden_dim, intermediate_dim))
    
    elif shape_type == "comprehensive":
        # Comprehensive sweep across all dimensions
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        for m in sizes:
            for n in sizes:
                for k in [64, 256, 1024, 4096]:
                    shapes.append((m, n, k))
    
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")
    
    return shapes


def run_benchmark(shapes: List[Tuple[int, int, int]],
                 warmup: int = 5,
                 iterations: int = 20,
                 output_dir: Optional[str] = None) -> List[GEMMResult]:
    """
    Run benchmark across all shapes.
    
    Args:
        shapes: List of (M, N, K) tuples
        warmup: Warmup iterations
        iterations: Timed iterations
        output_dir: Directory to save results
    
    Returns:
        List of GEMMResults
    """
    results = []
    
    print(f"Benchmarking {len(shapes)} shapes...")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} | {'GFLOPS':>10} | {'AI(F/B)':>10} | {'BW(GB/s)':>10}")
    print("-" * 70)
    
    for i, (M, N, K) in enumerate(shapes):
        try:
            result = benchmark_shape(M, N, K, warmup, iterations)
            results.append(result)
            
            print(f"{M:>6} {N:>6} {K:>6} | {result.time_ms:>10.3f} | {result.gflops:>10.1f} | "
                  f"{result.arithmetic_intensity:>10.1f} | {result.achieved_bandwidth_gbps:>10.1f}")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} | Error: {e}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"roofline_results_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "gpu_specs": get_gpu_specs(),
            "results": [
                {
                    "M": r.M, "N": r.N, "K": r.K,
                    "time_ms": r.time_ms,
                    "gflops": r.gflops,
                    "arithmetic_intensity": r.arithmetic_intensity,
                    "achieved_bandwidth_gbps": r.achieved_bandwidth_gbps,
                    "compute_efficiency": r.compute_efficiency,
                    "memory_efficiency": r.memory_efficiency,
                }
                for r in results
            ]
        }
        
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GEMM Roofline Benchmark")
    parser.add_argument("--shape-type", type=str, default="balanced",
                       choices=["balanced", "memory_heavy", "compute_heavy", 
                               "layer_like", "comprehensive"],
                       help="Type of shape progression")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--peak-tflops", type=float, default=2500,
                       help="Peak TFLOPS for the GPU")
    parser.add_argument("--peak-bandwidth", type=float, default=8000,
                       help="Peak memory bandwidth in GB/s")
    
    args = parser.parse_args()
    
    # Update GPU specs from args
    specs = get_gpu_specs()
    specs["peak_fp16_tflops"] = args.peak_tflops
    specs["peak_bandwidth_gbps"] = args.peak_bandwidth
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"Peak TFLOPS: {args.peak_tflops}")
    print(f"Peak Bandwidth: {args.peak_bandwidth} GB/s")
    print()
    
    # Generate shapes
    shapes = generate_shapes(args.shape_type)
    
    # Run benchmark
    results = run_benchmark(
        shapes,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total shapes benchmarked: {len(results)}")
    
    if results:
        best_perf = max(results, key=lambda r: r.gflops)
        worst_perf = min(results, key=lambda r: r.gflops)
        
        print(f"  Best performance: {best_perf.gflops:.1f} GFLOPS (M={best_perf.M}, N={best_perf.N}, K={best_perf.K})")
        print(f"  Worst performance: {worst_perf.gflops:.1f} GFLOPS (M={worst_perf.M}, N={worst_perf.N}, K={worst_perf.K})")
        
        # Find crossover point (where arithmetic intensity exceeds ~100)
        crossover = [r for r in results if r.arithmetic_intensity > 100]
        if crossover:
            print(f"  Shapes crossing into compute-bound region (AI > 100):")
            for r in crossover[:5]:
                print(f"    M={r.M}, N={r.N}, K={r.K}: AI={r.arithmetic_intensity:.1f}")


if __name__ == "__main__":
    main()
