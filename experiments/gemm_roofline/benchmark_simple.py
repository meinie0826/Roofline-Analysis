"""
GEMM Roofline Benchmark - Compatible with DeepGEMM BF16 API

DeepGEMM bf16_gemm_nt requires N == K (hidden dimension matrices).
This script generates shapes that satisfy this constraint while still
providing good coverage of memory-bound to compute-bound regions.

Shapes tested:
- Attention QKV projections: M = batch*seq, N = K = hidden
- Various batch sizes and sequence lengths
- Various hidden dimensions (LLaMA, Qwen, etc.)
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
    print("Error: PyTorch required")
    sys.exit(1)

# Check DeepGEMM
HAS_DEEPGEMM = False
HAS_DEEPGEMM_BF16 = False
try:
    import deep_gemm
    HAS_DEEPGEMM = True
    HAS_DEEPGEMM_BF16 = hasattr(deep_gemm, 'bf16_gemm_nt')
    if HAS_DEEPGEMM_BF16:
        print("✓ DeepGEMM BF16 API available")
except ImportError:
    print("✗ DeepGEMM not installed (only cuBLAS will be tested)")


@dataclass
class GEMMResult:
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


# B300 Specifications
GPU_SPECS = {
    "name": "NVIDIA B300",
    "peak_bf16_tflops": 4500,
    "peak_bandwidth_gbps": 7700,
}


def calculate_ai(M: int, N: int, K: int, element_size: int = 2) -> float:
    """Calculate arithmetic intensity."""
    flops = 2 * M * N * K
    # Read A (M×K), B (K×N), write C (M×N)
    bytes_accessed = (M * K + K * N + M * N) * element_size
    return flops / bytes_accessed


def benchmark_cublas(M: int, N: int, K: int, warmup: int = 5, iters: int = 20) -> GEMMResult:
    """Benchmark cuBLAS via torch.matmul."""
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
    
    for _ in range(iters):
        start.record()
        torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time * 1e-3) / 1e9
    
    ai = calculate_ai(M, N, K, element_size=2)
    bytes_accessed = (M * K + K * N + M * N) * 2
    bandwidth = bytes_accessed / (avg_time * 1e-3) / 1e9
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="bf16",
        backend="cuBLAS",
        time_ms=avg_time,
        gflops=gflops,
        arithmetic_intensity=ai,
        bandwidth_gbps=bandwidth,
        compute_efficiency=gflops / 1000 / GPU_SPECS["peak_bf16_tflops"],
    )


def benchmark_deepgemm(M: int, N: int, K: int, warmup: int = 5, iters: int = 20) -> Optional[GEMMResult]:
    """Benchmark DeepGEMM BF16."""
    if not HAS_DEEPGEMM_BF16:
        return None
    
    # DeepGEMM requires N == K
    if N != K:
        print(f"  Skipping DeepGEMM: N({N}) != K({K})")
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
    
    for _ in range(iters):
        start.record()
        deep_gemm.bf16_gemm_nt(A, B, D)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = np.median(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time * 1e-3) / 1e9
    
    ai = calculate_ai(M, N, K, element_size=2)
    bytes_accessed = (M * K + K * N + M * N) * 2
    bandwidth = bytes_accessed / (avg_time * 1e-3) / 1e9
    
    return GEMMResult(
        M=M, N=N, K=K,
        dtype="bf16",
        backend="DeepGEMM",
        time_ms=avg_time,
        gflops=gflops,
        arithmetic_intensity=ai,
        bandwidth_gbps=bandwidth,
        compute_efficiency=gflops / 1000 / GPU_SPECS["peak_bf16_tflops"],
    )


def generate_shapes() -> List[Tuple[int, int, int]]:
    """
    Generate shapes compatible with DeepGEMM BF16 API.
    
    Constraint: N == K (hidden dimension)
    Vary: M (batch × seq_len), hidden dimension
    """
    shapes = []
    
    # Popular model hidden dimensions
    hidden_dims = [
        4096,   # Llama 7B
        5120,   # Qwen 14B
        6144,   # Qwen 32B
        8192,   # Llama 70B
    ]
    
    # Batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Sequence lengths
    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # ===========================================
    # 1. Attention projections: M = batch*seq, N = K = hidden
    # ===========================================
    for hidden in hidden_dims:
        for batch in batch_sizes:
            for seq in seq_lens:
                M = batch * seq
                N = hidden
                K = hidden  # N == K for DeepGEMM
                shapes.append((M, N, K))
    
    # ===========================================
    # 2. Small shapes (memory-bound region detail)
    # ===========================================
    for size in [32, 64, 128, 256]:
        shapes.append((size, size, size))
    
    # ===========================================
    # 3. Medium shapes (near ridge point)
    # ===========================================
    for hidden in [2048, 4096, 8192]:
        for M in [512, 1024, 2048]:
            shapes.append((M, hidden, hidden))
    
    # ===========================================
    # 4. Large shapes (compute-bound region)
    # ===========================================
    for hidden in [4096, 8192, 16384]:
        for M in [4096, 8192, 16384]:
            shapes.append((M, hidden, hidden))
    
    # ===========================================
    # 5. Very large M (long sequences)
    # ===========================================
    for hidden in [4096, 8192]:
        for M in [32768, 65536, 131072]:
            shapes.append((M, hidden, hidden))
    
    # Deduplicate and sort
    unique_shapes = sorted(set(shapes))
    
    return unique_shapes


def run_benchmark(shapes: List[Tuple[int, int, int]],
                  warmup: int = 5,
                  iters: int = 20,
                  output_dir: Optional[str] = None) -> List[GEMMResult]:
    """Run benchmark on all shapes."""
    results = []
    
    print(f"\nBenchmarking {len(shapes)} shapes (N == K for DeepGEMM compatibility)")
    print("=" * 100)
    print(f"{'Shape':<20} | {'Backend':<12} | {'Time(ms)':<10} | {'TFLOPS':<10} | {'AI':<10} | {'Eff':<8}")
    print("=" * 100)
    
    for i, (M, N, K) in enumerate(shapes):
        # Skip very large shapes that might OOM
        est_mem = (M * K + K * N + M * N) * 2  # BF16
        if est_mem > 500 * 1024**2:  # > 500MB per matrix set
            # Try anyway but handle OOM
            pass
        
        # cuBLAS
        try:
            r = benchmark_cublas(M, N, K, warmup, iters)
            results.append(r)
            tflops = r.gflops / 1000
            print(f"{M:>7}×{N:>7}×{K:>7} | {r.backend:<12} | {r.time_ms:>10.3f} | "
                  f"{tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.compute_efficiency*100:>6.2f}%")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{M:>7}×{N:>7}×{K:>7} | cuBLAS: OOM")
                continue
            else:
                print(f"{M:>7}×{N:>7}×{K:>7} | cuBLAS error: {e}")
        
        # DeepGEMM
        if HAS_DEEPGEMM_BF16:
            try:
                r = benchmark_deepgemm(M, N, K, warmup, iters)
                if r:
                    results.append(r)
                    tflops = r.gflops / 1000
                    print(f"{M:>7}×{N:>7}×{K:>7} | {r.backend:<12} | {r.time_ms:>10.3f} | "
                          f"{tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.compute_efficiency*100:>6.2f}%")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pass
                else:
                    print(f"{M:>7}×{N:>7}×{K:>7} | DeepGEMM error: {e}")
    
    # Save results
    if output_dir and results:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"gemm_roofline_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "gpu_specs": GPU_SPECS,
            "has_deepgemm": HAS_DEEPGEMM,
            "has_deepgemm_bf16": HAS_DEEPGEMM_BF16,
            "constraint": "N == K for DeepGEMM BF16 compatibility",
            "num_shapes": len(shapes),
            "num_results": len(results),
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
    parser = argparse.ArgumentParser(description="GEMM Roofline Benchmark (DeepGEMM compatible)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DeepGEMM BF16: {HAS_DEEPGEMM_BF16}")
    
    shapes = generate_shapes()
    print(f"\nGenerated {len(shapes)} shapes")
    
    results = run_benchmark(
        shapes,
        warmup=args.warmup,
        iters=args.iterations,
        output_dir=args.output_dir
    )
    
    # Summary
    if results:
        print("\n" + "=" * 100)
        print("Summary:")
        
        by_backend = {}
        for r in results:
            if r.backend not in by_backend:
                by_backend[r.backend] = []
            by_backend[r.backend].append(r)
        
        for backend, br in by_backend.items():
            tflops_arr = np.array([x.gflops / 1000 for x in br])
            print(f"\n{backend}:")
            print(f"  Samples: {len(br)}")
            print(f"  Max TFLOPS:    {max(tflops_arr):>8.1f}")
            print(f"  Avg TFLOPS:    {np.mean(tflops_arr):>8.1f}")
            print(f"  Median TFLOPS: {np.median(tflops_arr):>8.1f}")
            
            best = max(br, key=lambda x: x.gflops)
            print(f"  Best shape: {best.M}×{best.N}×{best.K}")


if __name__ == "__main__":
    main()
