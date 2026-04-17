"""
DeepGEMM Roofline Benchmark

This script benchmarks GEMM performance using DeepGEMM for FP8 and cuBLAS for BF16,
analyzing performance characteristics across different matrix shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch is required")
    sys.exit(1)

# DeepGEMM availability check
HAS_DEEPGEMM = False
try:
    import deep_gemm
    HAS_DEEPGEMM = True
except ImportError:
    print("Warning: DeepGEMM not installed. FP8 benchmarks will be skipped.")
    print("Install with: pip install deep-gemm")


@dataclass
class GEMMResult:
    """Result from a single GEMM benchmark."""
    M: int
    N: int
    K: int
    dtype: str
    backend: str
    time_ms: float
    gflops: float
    bytes_accessed: int
    arithmetic_intensity: float
    achieved_bandwidth_gbps: float
    theoretical_peak_tflops: float
    peak_bandwidth_gbps: float
    compute_efficiency: float
    memory_efficiency: float


def calculate_arithmetic_intensity(M: int, N: int, K: int, 
                                   element_size: int = 2,  # FP16/BF16 = 2, FP8 = 1
                                   c_reads: bool = False) -> Tuple[int, float]:
    """
    Calculate arithmetic intensity for GEMM: C = A @ B
    
    Returns:
        (total_bytes, arithmetic_intensity)
    """
    flops = 2 * M * N * K
    
    bytes_a = M * K * element_size
    bytes_b = K * N * element_size
    bytes_c_write = M * N * element_size
    bytes_c_read = M * N * element_size if c_reads else 0
    
    total_bytes = bytes_a + bytes_b + bytes_c_write + bytes_c_read
    arithmetic_intensity = flops / total_bytes
    
    return total_bytes, arithmetic_intensity


def get_gpu_specs() -> dict:
    """Get GPU specifications for theoretical peak calculations."""
    return {
        "name": "B300",
        "peak_fp16_tflops": 2500,
        "peak_fp8_tflops": 5000,
        "peak_bf16_tflops": 1250,
        "peak_bandwidth_gbps": 8000,
        "sm_count": 114,
        "max_smem_per_sm": 227 * 1024,
    }


def benchmark_torch_matmul(M: int, N: int, K: int,
                           dtype: torch.dtype,
                           warmup: int = 5,
                           iterations: int = 20) -> GEMMResult:
    """
    Benchmark PyTorch matmul (cuBLAS backend).
    
    Args:
        M, N, K: Matrix dimensions
        dtype: torch.float16 or torch.bfloat16
        warmup: Warmup iterations
        iterations: Timed iterations
    
    Returns:
        GEMMResult with performance metrics
    """
    torch.manual_seed(42)
    device = "cuda"
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = torch.empty(M, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
    
    # Timed runs using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start_event.record()
        torch.matmul(A, B, out=C)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    avg_time_ms = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    element_size = 2 if dtype in [torch.float16, torch.bfloat16] else 1
    bytes_accessed, ai = calculate_arithmetic_intensity(M, N, K, element_size)
    bandwidth = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    specs = get_gpu_specs()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    peak_tflops = specs[f"peak_{dtype_str}_tflops"]
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype=dtype_str,
        backend="cuBLAS (torch.matmul)",
        time_ms=avg_time_ms,
        gflops=gflops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=ai,
        achieved_bandwidth_gbps=bandwidth,
        theoretical_peak_tflops=peak_tflops,
        peak_bandwidth_gbps=specs["peak_bandwidth_gbps"],
        compute_efficiency=gflops / 1000 / peak_tflops,
        memory_efficiency=bandwidth / specs["peak_bandwidth_gbps"],
    )


def benchmark_deepgemm_fp8(M: int, N: int, K: int,
                          warmup: int = 5,
                          iterations: int = 20) -> Optional[GEMMResult]:
    """
    Benchmark DeepGEMM FP8 GEMM.
    
    DeepGEMM expects:
    - A: (M, K) in E4M3 format
    - B: (K, N) in E4M3 format (transposed in memory)
    - Scaling factors for A and B in FP32
    
    Args:
        M, N, K: Matrix dimensions
        warmup: Warmup iterations
        iterations: Timed iterations
    
    Returns:
        GEMMResult with performance metrics, or None if DeepGEMM not available
    """
    if not HAS_DEEPGEMM:
        return None
    
    torch.manual_seed(42)
    device = "cuda"
    
    # Create FP8 tensors (E4M3)
    A_fp8 = torch.randn(M, K, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    B_fp8 = torch.randn(K, N, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    
    # Create scaling factors (FP32, required by DeepGEMM)
    # For simplicity, use scale of 1.0
    # Note: DeepGEMM requires transposed scaling factor layout for LHS
    scale_a = torch.ones((M,), dtype=torch.float32, device=device)
    scale_b = torch.ones((N,), dtype=torch.float32, device=device)
    
    # Output tensor
    C = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(warmup):
        try:
            deep_gemm.fp8_gemm_nt(C, A_fp8, scale_a, B_fp8, scale_b)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"DeepGEMM warmup error: {e}")
            return None
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            deep_gemm.fp8_gemm_nt(C, A_fp8, scale_a, B_fp8, scale_b)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        except Exception as e:
            print(f"DeepGEMM timing error: {e}")
            return None
    
    avg_time_ms = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    # FP8 = 1 byte
    bytes_accessed, ai = calculate_arithmetic_intensity(M, N, K, element_size=1)
    bandwidth = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    specs = get_gpu_specs()
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="fp8",
        backend="DeepGEMM",
        time_ms=avg_time_ms,
        gflops=gflops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=ai,
        achieved_bandwidth_gbps=bandwidth,
        theoretical_peak_tflops=specs["peak_fp8_tflops"],
        peak_bandwidth_gbps=specs["peak_bandwidth_gbps"],
        compute_efficiency=gflops / 1000 / specs["peak_fp8_tflops"],
        memory_efficiency=bandwidth / specs["peak_bandwidth_gbps"],
    )


def benchmark_deepgemm_bf16(M: int, N: int, K: int,
                           warmup: int = 5,
                           iterations: int = 20) -> Optional[GEMMResult]:
    """
    Benchmark DeepGEMM BF16 GEMM.
    
    Note: DeepGEMM's primary focus is FP8, but it may have BF16 kernels.
    If not available, fall back to cuBLAS.
    
    Args:
        M, N, K: Matrix dimensions
        warmup: Warmup iterations
        iterations: Timed iterations
    
    Returns:
        GEMMResult with performance metrics, or None if not available
    """
    if not HAS_DEEPGEMM:
        return None
    
    # Check if DeepGEMM has BF16 kernels
    if not hasattr(deep_gemm, 'bf16_gemm_nt'):
        # Fall back to cuBLAS
        return benchmark_torch_matmul(M, N, K, torch.bfloat16, warmup, iterations)
    
    torch.manual_seed(42)
    device = "cuda"
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(warmup):
        deep_gemm.bf16_gemm_nt(C, A, B)
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        deep_gemm.bf16_gemm_nt(C, A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time_ms = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    bytes_accessed, ai = calculate_arithmetic_intensity(M, N, K, element_size=2)
    bandwidth = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    specs = get_gpu_specs()
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="bf16",
        backend="DeepGEMM BF16",
        time_ms=avg_time_ms,
        gflops=gflops,
        bytes_accessed=bytes_accessed,
        arithmetic_intensity=ai,
        achieved_bandwidth_gbps=bandwidth,
        theoretical_peak_tflops=specs["peak_bf16_tflops"],
        peak_bandwidth_gbps=specs["peak_bandwidth_gbps"],
        compute_efficiency=gflops / 1000 / specs["peak_bf16_tflops"],
        memory_efficiency=bandwidth / specs["peak_bandwidth_gbps"],
    )


def generate_shapes(shape_type: str = "balanced") -> List[Tuple[int, int, int]]:
    """Generate test shapes."""
    shapes = []
    
    if shape_type == "balanced":
        sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        for size in sizes:
            shapes.append((size, size, size))
    elif shape_type == "memory_heavy":
        base_sizes = [128, 256, 512, 1024, 2048, 4096]
        k_values = [32, 64, 128]
        for size in base_sizes:
            for k in k_values:
                shapes.append((size, size, k))
    elif shape_type == "compute_heavy":
        mn_sizes = [64, 128, 256, 512, 1024]
        k_values = [256, 512, 1024, 2048, 4096]
        for mn in mn_sizes:
            for k in k_values:
                shapes.append((mn, mn, k))
    elif shape_type == "inference_like":
        # Common inference shapes: batch_size * seq_len patterns
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_lens = [128, 256, 512, 1024, 2048]
        hidden_dim = 4096
        
        for bs in batch_sizes:
            for seq in seq_lens:
                m = bs * seq
                shapes.append((m, hidden_dim, hidden_dim))  # QKV projection
                shapes.append((m, hidden_dim * 4, hidden_dim))  # FFN
    
    return shapes


def run_benchmark(shapes: List[Tuple[int, int, int]],
                 warmup: int = 5,
                 iterations: int = 20,
                 output_dir: Optional[str] = None) -> List[GEMMResult]:
    """
    Run benchmark across all shapes and backends.
    
    Args:
        shapes: List of (M, N, K) tuples
        warmup: Warmup iterations
        iterations: Timed iterations
        output_dir: Output directory for results
    
    Returns:
        List of GEMMResult objects
    """
    results = []
    
    print(f"Benchmarking {len(shapes)} shapes...")
    print("=" * 100)
    print(f"{'Shape':<18} | {'Backend':<20} | {'Time(ms)':<10} | {'GFLOPS':<12} | {'AI':<10} | {'BW(GB/s)':<12}")
    print("=" * 100)
    
    for i, (M, N, K) in enumerate(shapes):
        # BF16 - cuBLAS
        try:
            result = benchmark_torch_matmul(M, N, K, torch.bfloat16, warmup, iterations)
            results.append(result)
            print(f"{M:>6}x{N:>6}x{K:>6} | {result.backend:<20} | {result.time_ms:>10.3f} | "
                  f"{result.gflops:>12.1f} | {result.arithmetic_intensity:>10.1f} | "
                  f"{result.achieved_bandwidth_gbps:>12.1f}")
        except Exception as e:
            print(f"{M:>6}x{N:>6}x{K:>6} | cuBLAS BF16 Error: {e}")
        
        # FP8 - DeepGEMM
        if HAS_DEEPGEMM:
            try:
                result = benchmark_deepgemm_fp8(M, N, K, warmup, iterations)
                if result:
                    results.append(result)
                    print(f"{M:>6}x{N:>6}x{K:>6} | {result.backend:<20} | {result.time_ms:>10.3f} | "
                          f"{result.gflops:>12.1f} | {result.arithmetic_intensity:>10.1f} | "
                          f"{result.achieved_bandwidth_gbps:>12.1f}")
            except Exception as e:
                print(f"{M:>6}x{N:>6}x{K:>6} | DeepGEMM FP8 Error: {e}")
        
        # BF16 - DeepGEMM (if available)
        if HAS_DEEPGEMM:
            try:
                result = benchmark_deepgemm_bf16(M, N, K, warmup, iterations)
                if result and result.backend == "DeepGEMM BF16":
                    results.append(result)
                    print(f"{M:>6}x{N:>6}x{K:>6} | {result.backend:<20} | {result.time_ms:>10.3f} | "
                          f"{result.gflops:>12.1f} | {result.arithmetic_intensity:>10.1f} | "
                          f"{result.achieved_bandwidth_gbps:>12.1f}")
            except Exception as e:
                pass  # DeepGEMM BF16 may not be available
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"deepgemm_roofline_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "gpu_specs": get_gpu_specs(),
            "has_deepgemm": HAS_DEEPGEMM,
            "results": [
                {
                    "M": r.M, "N": r.N, "K": r.K,
                    "dtype": r.dtype,
                    "backend": r.backend,
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
    parser = argparse.ArgumentParser(description="DeepGEMM Roofline Benchmark")
    parser.add_argument("--shape-type", type=str, default="balanced",
                       choices=["balanced", "memory_heavy", "compute_heavy", "inference_like"],
                       help="Type of shape progression")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"DeepGEMM available: {HAS_DEEPGEMM}")
    specs = get_gpu_specs()
    print(f"GPU specs: {specs['name']}, Peak FP8: {specs['peak_fp8_tflops']} TFLOPS, "
          f"Peak BF16: {specs['peak_bf16_tflops']} TFLOPS, BW: {specs['peak_bandwidth_gbps']} GB/s")
    print()
    
    shapes = generate_shapes(args.shape_type)
    results = run_benchmark(
        shapes,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n" + "=" * 100)
    print("Summary:")
    print(f"  Total benchmarks: {len(results)}")
    
    if results:
        by_backend = {}
        for r in results:
            key = f"{r.backend} ({r.dtype})"
            if key not in by_backend:
                by_backend[key] = []
            by_backend[key].append(r)
        
        for backend, backend_results in by_backend.items():
            best = max(backend_results, key=lambda x: x.gflops)
            worst = min(backend_results, key=lambda x: x.gflops)
            avg_eff = np.mean([r.compute_efficiency for r in backend_results])
            
            print(f"\n  {backend}:")
            print(f"    Best: {best.gflops:.1f} GFLOPS (M={best.M}, N={best.N}, K={best.K})")
            print(f"    Worst: {worst.gflops:.1f} GFLOPS (M={worst.M}, N={worst.N}, K={worst.K})")
            print(f"    Avg compute efficiency: {avg_eff*100:.1f}%")


if __name__ == "__main__":
    main()
