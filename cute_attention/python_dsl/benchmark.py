#!/usr/bin/env python3
"""
FlashAttention Benchmark with SDPA Comparison
"""

import torch
import torch.nn.functional as F
import time
import argparse
import math

try:
    from kernels.stage0_attention import attention_forward, compute_tflops, compute_tc_utilization, HEAD_DIM
    HAS_KERNEL = True
except ImportError as e:
    HAS_KERNEL = False
    KERNEL_ERROR = str(e)


def sdpa_attention(Q, K, V, scale=None):
    """PyTorch SDPA baseline"""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    
    # Use scaled_dot_product_attention
    output = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    return output


def benchmark(func, Q, K, V, warmup=10, repeat=100, verbose=True):
    """Benchmark an attention function"""
    # Warmup
    for _ in range(warmup):
        _ = func(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = func(Q, K, V)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    
    tflops = compute_tflops(Q, mean_time)
    tc_util = compute_tc_utilization(tflops)
    
    metrics = {
        'time_ms': mean_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'tflops': tflops,
        'tc_util': tc_util,
    }
    
    if verbose:
        print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms")
        print(f"  TFLOPs: {tflops:.1f}")
        print(f"  TC Util: {tc_util:.2f}%")
    
    return metrics


def run_benchmark(seqlens=[1024, 2048, 4096, 8192, 16384, 32768], 
                  batch_size=1, 
                  nheads=32, 
                  headdim=128):
    """Run benchmark comparing Stage 0 and SDPA"""
    
    print("="*90)
    print(" FlashAttention Benchmark - Stage 0 vs SDPA")
    print("="*90)
    
    if not HAS_KERNEL:
        print(f"\nWARNING: Custom kernel not available ({KERNEL_ERROR})")
        print("Showing SDPA baseline only\n")
    
    print(f"\nConfig: B={batch_size}, H={nheads}, d={headdim}")
    print()
    
    # Header
    print(f"{'N':<8} {'SDPA (ms)':<12} {'Stage0 (ms)':<12} {'Speedup':<10} {'SDPA TFLOPs':<12} {'Stage0 TFLOPs':<12}")
    print("-"*90)
    
    results = []
    
    for N in seqlens:
        torch.manual_seed(42)
        
        Q = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        K = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        V = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        
        # Benchmark SDPA
        sdpa_metrics = benchmark(sdpa_attention, Q, K, V, verbose=False)
        
        # Benchmark our kernel (if available)
        if HAS_KERNEL:
            try:
                kernel_metrics = benchmark(attention_forward, Q, K, V, verbose=False)
                speedup = sdpa_metrics['time_ms'] / kernel_metrics['time_ms']
                kernel_time = f"{kernel_metrics['time_ms']:.3f}"
                kernel_tflops = f"{kernel_metrics['tflops']:.1f}"
            except Exception as e:
                kernel_time = "ERROR"
                kernel_tflops = "N/A"
                speedup = 0.0
        else:
            kernel_time = "N/A"
            kernel_tflops = "N/A"
            speedup = 0.0
        
        print(f"{N:<8} {sdpa_metrics['time_ms']:<12.3f} {kernel_time:<12} {speedup:<10.2f} {sdpa_metrics['tflops']:<12.1f} {kernel_tflops:<12}")
        
        results.append({
            'seqlen': N,
            'sdpa': sdpa_metrics,
            'kernel': kernel_metrics if HAS_KERNEL else None,
            'speedup': speedup
        })
    
    print("="*90)
    
    # Save results
    import json
    import os
    from datetime import datetime
    
    os.makedirs("cute_attention/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    result_file = f"cute_attention/results/benchmark_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'stage': 0,
            'config': {
                'batch_size': batch_size,
                'nheads': nheads,
                'headdim': headdim,
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    return results


def run_correctness_test():
    """Run correctness test"""
    from tests.test_correctness import run_all_tests
    
    print("="*60)
    print(" Running Correctness Tests")
    print("="*60)
    
    if not HAS_KERNEL:
        print(f"\nERROR: Kernel not available ({KERNEL_ERROR})")
        return False
    
    run_all_tests(attention_forward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run correctness tests')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    parser.add_argument('--seqlen', type=int, nargs='+', default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=32)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
    
    if args.test:
        run_correctness_test()
        print()
    
    if args.bench or not args.test:
        run_benchmark(
            seqlens=args.seqlen,
            batch_size=args.batch,
            nheads=args.heads
        )
