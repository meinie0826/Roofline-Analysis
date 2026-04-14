#!/usr/bin/env python3
"""
FlashAttention-4 Ablation Benchmark (基于真实 FA4 接口)

由于 FA4 内部 ablation 接口不公开，我们用不同参数来模拟：
- 对比不同 seqlen 的性能
- 对比 causal vs non-causal
- 测量实际 TFLOPs/s 和 TC 利用率

使用方法:
    python hopper/benchmark_ablation.py --seqlen 1024,2048,4096,8192
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    """单次 benchmark 结果"""
    batch: int
    seqlen: int
    nheads: int
    headdim: int
    causal: bool
    warmup_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    tflops: float
    tc_util_pct: float
    max_diff: float = 0.0
    error: str = ""


def get_peak_tflops():
    """获取当前 GPU 的峰值 TFLOPs"""
    if not torch.cuda.is_available():
        return 100.0
    
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    # B200/Blackwell
    if sm >= 100:
        return 2250.0
    # H100/Hopper
    elif sm >= 90:
        return 989.0
    # A100
    elif sm >= 80:
        return 312.0
    else:
        return 100.0


def flash_attn_flops(batch, nheads, seqlen, headdim, causal=False):
    """计算 attention FLOPs"""
    if causal:
        avg_seqlen = (seqlen + 1) / 2
    else:
        avg_seqlen = seqlen
    
    # QK^T + PV
    return batch * nheads * 2 * seqlen * avg_seqlen * headdim * 2


def benchmark_fa4(
    seqlen: int,
    batch: int = 1,
    nheads: int = 16,
    headdim: int = 128,
    causal: bool = True,
    warmup: int = 10,
    rep: int = 50,
    check_correctness: bool = True,
) -> BenchmarkResult:
    """Benchmark FlashAttention-4"""
    
    result = BenchmarkResult(
        batch=batch,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        causal=causal,
        warmup_ms=0,
        avg_ms=0,
        min_ms=0,
        max_ms=0,
        tflops=0,
        tc_util_pct=0,
    )
    
    try:
        # 导入 FA4
        from flash_attn_interface import flash_attn_func
        has_fa4 = True
    except ImportError:
        try:
            # 尝试从本地 hopper 目录导入
            import sys
            sys.path.insert(0, 'flash-attention/hopper')
            from flash_attn_interface import flash_attn_func
            has_fa4 = True
        except ImportError:
            has_fa4 = False
            result.error = "FA4 not available, using PyTorch SDPA"
    
    try:
        # 创建输入
        dtype = torch.bfloat16
        device = "cuda"
        
        q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        
        softmax_scale = 1.0 / math.sqrt(headdim)
        
        # Warmup
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if has_fa4:
            for _ in range(warmup):
                _ = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
        else:
            # Fallback to SDPA
            q_sdpa = q.transpose(1, 2)  # [b, h, s, d]
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            for _ in range(warmup):
                _ = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, 
                    scale=softmax_scale, 
                    is_causal=causal
                )
        
        torch.cuda.synchronize()
        result.warmup_ms = (time.perf_counter() - start) * 1000
        
        # Timed runs
        times = []
        for _ in range(rep):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            if has_fa4:
                out = flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
            else:
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    scale=softmax_scale,
                    is_causal=causal
                )
                out = out.transpose(1, 2)  # 转回 [b, s, h, d]
            
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        import numpy as np
        times = np.array(times)
        
        result.avg_ms = float(times.mean())
        result.min_ms = float(times.min())
        result.max_ms = float(times.max())
        
        # Compute metrics
        flops = flash_attn_flops(batch, nheads, seqlen, headdim, causal)
        result.tflops = flops / result.avg_ms / 1e9
        
        peak = get_peak_tflops()
        result.tc_util_pct = result.tflops / peak * 100
        
        # Correctness check
        if check_correctness and has_fa4:
            q_ref = q.transpose(1, 2).float()
            k_ref = k.transpose(1, 2).float()
            v_ref = v.transpose(1, 2).float()
            
            ref_out = F.scaled_dot_product_attention(
                q_ref, k_ref, v_ref,
                scale=softmax_scale,
                is_causal=causal
            ).transpose(1, 2).to(dtype)
            
            result.max_diff = (out.float() - ref_out.float()).abs().max().item()
        
    except Exception as e:
        import traceback
        result.error = str(e) + "\n" + traceback.format_exc()[-300:]
    
    return result


def run_benchmark_suite(seqlens: List[int], causal_only: bool = True) -> List[BenchmarkResult]:
    """运行完整 benchmark 套件"""
    
    results = []
    peak_tflops = get_peak_tflops()
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    
    print("\n" + "=" * 80)
    print("  FlashAttention-4 Performance Benchmark")
    print("=" * 80)
    print(f"  GPU: {device_name}")
    print(f"  Peak BF16: {peak_tflops:.0f} TFLOPs/s")
    print("=" * 80)
    
    causals = [True] if causal_only else [False, True]
    
    for causal in causals:
        print(f"\n  Causal={causal}")
        print("  " + "-" * 76)
        
        header = f"  {'Seqlen':<10}" + "".join(f" {'s='+str(s):>12}" for s in seqlens)
        print(header)
        
        for seqlen in seqlens:
            batch = max(1, 32768 // seqlen)  # 保持总 tokens 大致一致
            
            result = benchmark_fa4(
                seqlen=seqlen,
                batch=batch,
                causal=causal,
                warmup=5,
                rep=30,
            )
            results.append(result)
            
            if result.error:
                cell = "ERR"
            elif result.tflops > 0:
                cell = f"{result.tflops:.0f}T"
            else:
                cell = "FAIL"
            
            print(f"  {f's={seqlen}':<10}" + f" {cell:>12}")
    
    # Summary
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    
    valid_results = [r for r in results if not r.error and r.tflops > 0]
    if valid_results:
        avg_tflops = sum(r.tflops for r in valid_results) / len(valid_results)
        avg_util = sum(r.tc_util_pct for r in valid_results) / len(valid_results)
        print(f"  Average: {avg_tflops:.0f} TFLOPs/s ({avg_util:.0f}% TC util)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="FA4 Benchmark")
    parser.add_argument("--seqlen", type=lambda s: [int(x) for x in s.split(",")],
                       default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--causal-only", action="store_true", default=True)
    parser.add_argument("--csv", type=str, default=None)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    
    results = run_benchmark_suite(args.seqlen, args.causal_only)
    
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "batch", "seqlen", "nheads", "headdim", "causal",
                "avg_ms", "tflops", "tc_util_pct", "max_diff", "error"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "batch": r.batch,
                    "seqlen": r.seqlen,
                    "nheads": r.nheads,
                    "headdim": r.headdim,
                    "causal": r.causal,
                    "avg_ms": f"{r.avg_ms:.4f}" if r.avg_ms else "",
                    "tflops": f"{r.tflops:.1f}" if r.tflops else "",
                    "tc_util_pct": f"{r.tc_util_pct:.1f}" if r.tc_util_pct else "",
                    "max_diff": f"{r.max_diff:.6f}" if r.max_diff else "",
                    "error": r.error[:100] if r.error else "",
                })
        print(f"\nCSV saved: {args.csv}")


if __name__ == "__main__":
    main()
