"""
Extended GEMM Roofline Benchmark with LLM-Realistic Shapes

Tests various matrix shapes common in LLM inference and training:
- Attention projections (QKV)
- MLP/FFN layers (up/down/gate projections)
- Different batch sizes and sequence lengths
- Non-square matrices (M ≠ N ≠ K)
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

HAS_DEEPGEMM = False
HAS_DEEPGEMM_BF16 = False
try:
    import deep_gemm
    HAS_DEEPGEMM = True
    HAS_DEEPGEMM_BF16 = hasattr(deep_gemm, 'bf16_gemm_nt')
except ImportError:
    pass


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


# B300 Correct Specifications
GPU_SPECS = {
    "peak_bf16_tflops": 4500,
    "peak_bandwidth_gbps": 7700,
}


def get_ai(M: int, N: int, K: int, element_size: int = 2) -> float:
    """Calculate arithmetic intensity."""
    flops = 2 * M * N * K
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
    
    ai = get_ai(M, N, K, element_size=2)
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
    """Benchmark DeepGEMM BF16 if available."""
    if not HAS_DEEPGEMM_BF16:
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
    
    ai = get_ai(M, N, K, element_size=2)
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


def generate_llm_shapes() -> List[Tuple[int, int, int]]:
    """
    Generate LLM-realistic matrix shapes.
    
    Common patterns:
    - Attention: (batch*seq, hidden) @ (hidden, hidden)
    - QKV projection: M = batch*seq, N = hidden, K = hidden (same)
    - Attention output: M = batch*seq, N = hidden, K = num_heads*head_dim
    - MLP up: (batch*seq, hidden) @ (hidden, 4*hidden)
    - MLP down: (batch*seq, 4*hidden) @ (4*hidden, hidden)
    """
    shapes = []
    
    # Common hidden dimensions
    hidden_dims = [4096, 5120, 6144, 8192]  # Llama 7B, 13B, Qwen 14B, Llama 70B
    
    # Common batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    # Common sequence lengths
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    
    # =========================================
    # 1. Attention Projections (QKV)
    # M = batch * seq_len, N = hidden, K = hidden
    # =========================================
    for hidden in hidden_dims:
        for batch in batch_sizes:
            for seq in seq_lens:
                M = batch * seq
                N = hidden
                K = hidden
                shapes.append((M, N, K))
    
    # =========================================
    # 2. MLP/FFN Layers
    # =========================================
    for hidden in [4096, 8192]:  # Most common
        ffn_hidden = hidden * 4  # Standard FFN expansion
        
        for batch in [1, 4, 16, 64]:
            for seq in [128, 512, 2048]:
                M = batch * seq
                
                # Up projection: hidden -> 4*hidden
                shapes.append((M, ffn_hidden, hidden))
                
                # Down projection: 4*hidden -> hidden
                shapes.append((M, hidden, ffn_hidden))
                
                # Gate projection (SwiGLU): same as up
                shapes.append((M, ffn_hidden, hidden))
    
    # =========================================
    # 3. Attention Score Computation
    # Q @ K^T: (batch*seq, hidden) @ (hidden, seq)
    # =========================================
    for hidden in [4096, 8192]:
        for batch in [1, 8, 32]:
            for seq in [256, 1024, 4096]:
                M = batch * seq
                N = seq  # K^T has seq_len columns
                K = hidden
                shapes.append((M, N, K))
    
    # =========================================
    # 4. Attention Output
    # (batch*seq, seq) @ (seq, hidden)
    # =========================================
    for hidden in [4096, 8192]:
        for batch in [1, 8, 32]:
            for seq in [256, 1024, 4096]:
                M = batch * seq
                N = hidden
                K = seq
                shapes.append((M, N, K))
    
    # =========================================
    # 5. Special shapes for edge cases
    # =========================================
    
    # Very small (extreme memory-bound)
    for size in [32, 64, 128]:
        shapes.append((size, size, size))
    
    # Tall matrices (large M, small N)
    shapes.extend([
        (8192, 128, 4096),
        (16384, 256, 4096),
        (32768, 512, 4096),
    ])
    
    # Wide matrices (small M, large N)
    shapes.extend([
        (128, 8192, 4096),
        (256, 16384, 4096),
        (512, 32768, 4096),
    ])
    
    # Slender K
    shapes.extend([
        (1024, 4096, 64),
        (4096, 4096, 128),
        (8192, 4096, 256),
    ])
    
    return shapes


def generate_balanced_shapes() -> List[Tuple[int, int, int]]:
    """Generate balanced square-ish shapes for smooth roofline curve."""
    shapes = []
    
    # Powers of 2
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for size in sizes:
        shapes.append((size, size, size))
    
    # Near-square variations
    for base in [1024, 2048, 4096]:
        shapes.append((base, base * 2, base))
        shapes.append((base, base, base * 2))
        shapes.append((base * 2, base, base))
    
    return shapes


def generate_memory_bound_shapes() -> List[Tuple[int, int, int]]:
    """Generate shapes that are likely memory-bound (low AI)."""
    shapes = []
    
    for M in [128, 256, 512, 1024, 2048]:
        for N in [4096, 8192]:
            for K in [32, 64, 128]:
                shapes.append((M, N, K))
    
    return shapes


def generate_compute_bound_shapes() -> List[Tuple[int, int, int]]:
    """Generate shapes that are likely compute-bound (high AI)."""
    shapes = []
    
    for M in [1024, 2048, 4096]:
        for N in [4096, 8192]:
            for K in [4096, 8192, 16384]:
                shapes.append((M, N, K))
    
    return shapes


def run_benchmark(shapes: List[Tuple[int, int, int]],
                  warmup: int = 5,
                  iters: int = 20,
                  output_dir: Optional[str] = None) -> List[GEMMResult]:
    """Run benchmark across all shapes."""
    results = []
    
    # Deduplicate shapes
    unique_shapes = sorted(set(shapes))
    
    print(f"Benchmarking {len(unique_shapes)} unique shapes...")
    print("=" * 110)
    print(f"{'Shape':<20} | {'Backend':<12} | {'Time(ms)':<10} | {'TFLOPS':<10} | {'AI':<10} | {'BW(GB/s)':<10} | {'Eff':<8}")
    print("=" * 110)
    
    for i, (M, N, K) in enumerate(unique_shapes):
        # Skip if too large for memory
        est_mem = (M * K + K * N + M * N) * 2  # BF16
        if est_mem > 200 * 1024**2:  # Skip if > 200MB per matrix
            # Try anyway but catch OOM
            pass
        
        # cuBLAS
        try:
            r = benchmark_cublas(M, N, K, warmup, iters)
            results.append(r)
            tflops = r.gflops / 1000
            print(f"{M:>7}x{N:>7}x{K:>7} | {r.backend:<12} | {r.time_ms:>10.3f} | {tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.bandwidth_gbps:>10.1f} | {r.compute_efficiency*100:>6.1f}%")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{M:>7}x{N:>7}x{K:>7} | cuBLAS: OOM, skipping")
                continue
            else:
                print(f"{M:>7}x{N:>7}x{K:>7} | cuBLAS error: {e}")
        
        # DeepGEMM
        if HAS_DEEPGEMM_BF16:
            try:
                r = benchmark_deepgemm(M, N, K, warmup, iters)
                if r:
                    results.append(r)
                    tflops = r.gflops / 1000
                    print(f"{M:>7}x{N:>7}x{K:>7} | {r.backend:<12} | {r.time_ms:>10.3f} | {tflops:>10.2f} | {r.arithmetic_intensity:>10.1f} | {r.bandwidth_gbps:>10.1f} | {r.compute_efficiency*100:>6.1f}%")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pass
                else:
                    print(f"{M:>7}x{N:>7}x{K:>7} | DeepGEMM error: {e}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"llm_shapes_benchmark_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "gpu_specs": GPU_SPECS,
            "num_shapes": len(unique_shapes),
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
    parser = argparse.ArgumentParser(description="Extended GEMM Benchmark with LLM Shapes")
    parser.add_argument("--mode", type=str, default="llm",
                       choices=["llm", "balanced", "memory", "compute", "all"],
                       help="Shape generation mode")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DeepGEMM BF16: {HAS_DEEPGEMM_BF16}")
    print()
    
    # Generate shapes
    shapes = []
    if args.mode == "llm":
        shapes = generate_llm_shapes()
    elif args.mode == "balanced":
        shapes = generate_balanced_shapes()
    elif args.mode == "memory":
        shapes = generate_memory_bound_shapes()
    elif args.mode == "compute":
        shapes = generate_compute_bound_shapes()
    elif args.mode == "all":
        shapes = (generate_llm_shapes() + generate_balanced_shapes() + 
                  generate_memory_bound_shapes() + generate_compute_bound_shapes())
    
    print(f"Generated {len(shapes)} shapes in '{args.mode}' mode")
    print()
    
    results = run_benchmark(
        shapes,
        warmup=args.warmup,
        iters=args.iterations,
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n" + "=" * 110)
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
            print(f"    Samples: {len(br)}")
            print(f"    Best: {best.gflops/1000:.2f} TFLOPS (M={best.M}, N={best.N}, K={best.K})")
            print(f"    Avg compute efficiency: {avg_eff*100:.2f}%")


if __name__ == "__main__":
    main()
