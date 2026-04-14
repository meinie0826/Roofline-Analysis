#!/usr/bin/env python3
"""
FlashAttention Benchmark
"""

import torch
import time
import argparse
from kernels.stage0_attention import attention_forward, compute_tflops, compute_tc_utilization, HAS_CUTE


def benchmark(func, Q, K, V, warmup=10, repeat=100, verbose=True):
    """
    Benchmark an attention function
    
    Args:
        func: attention function
        Q, K, V: input tensors
        warmup: warmup iterations
        repeat: benchmark iterations
        verbose: print results
    
    Returns:
        metrics: dict with time_ms, tflops, tc_util
    """
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
        print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms (min: {min_time:.3f})")
        print(f"  TFLOPs: {tflops:.1f}")
        print(f"  TC Util: {tc_util:.2f}%")
    
    return metrics


def run_benchmark(seqlens=[128, 256, 512, 1024, 2048], 
                  batch_size=1, 
                  nheads=16, 
                  headdim=64):
    """Run benchmark for different sequence lengths"""
    print("="*70)
    print(" FlashAttention Benchmark - Stage 0 (Naive)")
    print("="*70)
    
    if not HAS_CUTE:
        raise RuntimeError(f"CuTe DSL not available: {CUTE_ERROR}")
    
    print("\nCuTe DSL: ✓ Available")
    print(f"\nConfig: B={batch_size}, H={nheads}, d={headdim}")
    print()
    
    print(f"{'N':<8} {'Time (ms)':<15} {'TFLOPs':<12} {'TC Util (%)':<12}")
    print("-"*70)
    
    results = []
    
    for N in seqlens:
        torch.manual_seed(42)
        
        Q = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        K = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        V = torch.randn(batch_size, nheads, N, headdim, 
                       device='cuda', dtype=torch.float32)
        
        metrics = benchmark(attention_forward, Q, K, V, verbose=False)
        
        print(f"{N:<8} {metrics['time_ms']:<15.3f} {metrics['tflops']:<12.1f} {metrics['tc_util']:<12.2f}")
        
        results.append({
            'seqlen': N,
            **metrics
        })
    
    print("="*70)
    
    return results


def run_correctness_test():
    """Run correctness test"""
    from tests.test_correctness import run_all_tests
    run_all_tests(attention_forward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run correctness tests')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    parser.add_argument('--seqlen', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=16)
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
