"""
DeepGEMM Roofline Benchmark (BF16 focus)

This script benchmarks GEMM performance using DeepGEMM and cuBLAS for BF16,
analyzing performance characteristics across different matrix shapes.

Note: FP8 in DeepGEMM requires specific block-wise scaling format.
For simplicity, we focus on BF16 which works out-of-the-box.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch is required")
    sys.exit(1)

# DeepGEMM check
HAS_DEEPGEMM = False
try:
    import deep_gemm
    HAS_DEEPGEMM = True
    # Check for BF16 API
    HAS_DEEPGEMM_BF16 = hasattr(deep_gemm, 'bf16_gemm_nt')
except ImportError:
    pass


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
    arithmetic_intensity: float
    bandwidth_gbps: float
    compute_efficiency: float
    memory_efficiency: float


def get_gpu_specs() -> dict:
    """Get GPU specifications."""
    return {
        "name": "B300",
        "peak_bf16_tflops": 1250,
        "peak_bandwidth_gbps": 8000,
    }


def calculate_arithmetic_intensity(M: int, N: int, K: int, element_size: int = 2) -> float:
    """Calculate arithmetic intensity for GEMM."""
    flops = 2 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * element_size
    return flops / bytes_accessed


def benchmark_cublas_bf16(M: int, N: int, K: int, warmup: int = 5, iterations: int = 20) -> GEMMResult:
    """Benchmark PyTorch matmul (cuBLAS backend) for BF16."""
    torch.manual_seed(42)
    device = "cuda"
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(warmup):
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for _ in range(iterations):
        start.record()
        torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time_ms = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    ai = calculate_arithmetic_intensity(M, N, K, element_size=2)
    bytes_accessed = (M * K + K * N + M * N) * 2
    bandwidth = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    specs = get_gpu_specs()
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="bf16",
        backend="cuBLAS",
        time_ms=avg_time_ms,
        gflops=gflops,
        arithmetic_intensity=ai,
        bandwidth_gbps=bandwidth,
        compute_efficiency=gflops / 1000 / specs["peak_bf16_tflops"],
        memory_efficiency=bandwidth / specs["peak_bandwidth_gbps"],
    )


def benchmark_deepgemm_bf16(M: int, N: int, K: int, warmup: int = 5, iterations: int = 20) -> Optional[GEMMResult]:
    """Benchmark DeepGEMM BF16 if available."""
    if not HAS_DEEPGEMM or not HAS_DEEPGEMM_BF16:
        return None
    
    torch.manual_seed(42)
    device = "cuda"
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(warmup):
        deep_gemm.bf16_gemm_nt(A, B, D)
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for _ in range(iterations):
        start.record()
        deep_gemm.bf16_gemm_nt(A, B, D)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time_ms = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time_ms * 1e-3) / 1e9
    
    ai = calculate_arithmetic_intensity(M, N, K, element_size=2)
    bytes_accessed = (M * K + K * N + M * N) * 2
    bandwidth = bytes_accessed / (avg_time_ms * 1e-3) / 1e9
    
    specs = get_gpu_specs()
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="bf16",
        backend="DeepGEMM",
        time_ms=avg_time_ms,
        gflops=gflops,
        arithmetic_intensity=ai,
        bandwidth_gbps=bandwidth,
        compute_efficiency=gflops / 1000 / specs["peak_bf16_tflops"],
        memory_efficiency=bandwidth / specs["peak_bandwidth_gbps"],
    )


def generate_shapes(shape_type: str) -> List[Tuple[int, int, int]]:
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
    
    return shapes


def run_benchmark(shapes: List[Tuple[int, int, int]], 
                  warmup: int = 5, 
                  iterations: int = 20,
                  output_dir: Optional[str] = None) -> List[GEMMResult]:
    """Run benchmark across all shapes."""
    results = []
    
    print(f"Benchmarking {len(shapes)} shapes...")
    print("=" * 100)
    print(f"{'Shape':<18} | {'Backend':<12} | {'Time(ms)':<10} | {'TFLOPS':<10} | {'AI':<10} | {'BW(GB/s)':<10} | {'Eff':<8}")
    print("=" * 100)
    
    for M, N, K in shapes:
        # cuBLAS
        try:
            r = benchmark_cublas_bf16(M, N, K, warmup, iterations)
            results.append(r)
            tflops = r.gflops / 1000
            print(f"{M:>6}x{N:>6}x{K:>6} | {r.backend:<12} | {r.time_ms:>10.3f} | {tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.bandwidth_gbps:>10.1f} | {r.compute_efficiency*100:>6.1f}%")
        except Exception as e:
            print(f"{M:>6}x{N:>6}x{K:>6} | cuBLAS error: {e}")
        
        # DeepGEMM
        if HAS_DEEPGEMM_BF16:
            try:
                r = benchmark_deepgemm_bf16(M, N, K, warmup, iterations)
                if r:
                    results.append(r)
                    tflops = r.gflops / 1000
                    print(f"{M:>6}x{N:>6}x{K:>6} | {r.backend:<12} | {r.time_ms:>10.3f} | {tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.bandwidth_gbps:>10.1f} | {r.compute_efficiency*100:>6.1f}%")
            except Exception as e:
                print(f"{M:>6}x{N:>6}x{K:>6} | DeepGEMM error: {e}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"gemm_roofline_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "gpu_specs": get_gpu_specs(),
            "has_deepgemm": HAS_DEEPGEMM,
            "has_deepgemm_bf16": HAS_DEEPGEMM_BF16,
            "results": [
                {
                    "M": r.M, "N": r.N, "K": r.K,
                    "dtype": r.dtype,
                    "backend": r.backend,
                    "time_ms": r.time_ms,
                    "gflops": r.gflops,
                    "tflops": r.gflops / 1000,
                    "arithmetic_intensity": r.arithmetic_intensity,
                    "bandwidth_gbps": r.bandwidth_gbps,
                    "compute_efficiency": r.compute_efficiency,
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
                       choices=["balanced", "memory_heavy", "compute_heavy"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DeepGEMM available: {HAS_DEEPGEMM}")
    print(f"DeepGEMM BF16 available: {HAS_DEEPGEMM_BF16}")
    specs = get_gpu_specs()
    print(f"Peak BF16: {specs['peak_bf16_tflops']} TFLOPS, BW: {specs['peak_bandwidth_gbps']} GB/s")
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
            if r.backend not in by_backend:
                by_backend[r.backend] = []
            by_backend[r.backend].append(r)
        
        for backend, br in by_backend.items():
            best = max(br, key=lambda x: x.gflops)
            avg_eff = np.mean([x.compute_efficiency for x in br])
            print(f"\n  {backend}:")
            print(f"    Best: {best.gflops/1000:.2f} TFLOPS (M={best.M}, N={best.N}, K={best.K})")
            print(f"    Avg compute efficiency: {avg_eff*100:.1f}%")


if __name__ == "__main__":
    main()
