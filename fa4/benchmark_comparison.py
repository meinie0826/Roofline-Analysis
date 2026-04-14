#!/usr/bin/env python3
"""
FlashAttention-4 性能分析与对比

测试内容：
1. FA4 在不同 seqlen 下的性能
2. 与 PyTorch SDPA 对比
3. Roofline 分析

注意：由于 FA4 的 ablation 接口不公开，我们无法做逐优化的 ablation。
      但可以展示 FA4 相比 naive 实现的性能提升。
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    method: str
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


def get_hardware_info():
    """获取硬件信息"""
    if not torch.cuda.is_available():
        return {"peak_tflops": 100.0, "peak_bw_gbps": 1000.0, "sm_count": 84}
    
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    
    if sm >= 100:  # B200
        return {"peak_tflops": 2250.0, "peak_bw_gbps": 8000.0, "sm_count": 148}
    elif sm >= 90:  # H100
        return {"peak_tflops": 989.0, "peak_bw_gbps": 3352.0, "sm_count": 132}
    elif sm >= 80:  # A100
        return {"peak_tflops": 312.0, "peak_bw_gbps": 2039.0, "sm_count": 108}
    else:
        return {"peak_tflops": 100.0, "peak_bw_gbps": 1000.0, "sm_count": 84}


def attention_flops(batch, nheads, seqlen, headdim, causal=False):
    """计算 attention FLOPs"""
    if causal:
        avg_seqlen = (seqlen + 1) / 2
    else:
        avg_seqlen = seqlen
    
    # QK^T: 2 * seqlen * avg_seqlen * headdim
    # PV:   2 * seqlen * avg_seqlen * headdim
    return batch * nheads * 2 * seqlen * avg_seqlen * headdim * 2


def benchmark_sdpa(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    causal: bool,
    warmup: int = 5,
    rep: int = 30,
    dtype: torch.dtype = torch.bfloat16,
) -> BenchmarkResult:
    """Benchmark PyTorch SDPA"""
    
    result = BenchmarkResult(
        method="SDPA",
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
        device = "cuda"
        q = torch.randn(batch, nheads, seqlen, headdim, dtype=dtype, device=device)
        k = torch.randn(batch, nheads, seqlen, headdim, dtype=dtype, device=device)
        v = torch.randn(batch, nheads, seqlen, headdim, dtype=dtype, device=device)
        
        scale = 1.0 / math.sqrt(headdim)
        
        # Warmup
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
        torch.cuda.synchronize()
        result.warmup_ms = (time.perf_counter() - start) * 1000
        
        # Timed runs
        times = []
        for _ in range(rep):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        import numpy as np
        times = np.array(times)
        
        result.avg_ms = float(times.mean())
        result.min_ms = float(times.min())
        result.max_ms = float(times.max())
        
        flops = attention_flops(batch, nheads, seqlen, headdim, causal)
        result.tflops = flops / result.avg_ms / 1e9
        
        hw = get_hardware_info()
        result.tc_util_pct = result.tflops / hw["peak_tflops"] * 100
        
    except Exception as e:
        import traceback
        result.error = f"{e}\n{traceback.format_exc()[-300:]}"
    
    return result


def benchmark_fa4(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    causal: bool,
    warmup: int = 5,
    rep: int = 30,
    dtype: torch.dtype = torch.bfloat16,
) -> BenchmarkResult:
    """Benchmark FlashAttention-4"""
    
    result = BenchmarkResult(
        method="FA4",
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
        # 尝试导入 FA4
        try:
            from flash_attn_interface import flash_attn_func
        except ImportError:
            # 尝试从本地路径导入
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flash-attention', 'hopper'))
            from flash_attn_interface import flash_attn_func
        
        device = "cuda"
        # FA4 使用 [batch, seqlen, nheads, headdim] 布局
        q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        
        scale = 1.0 / math.sqrt(headdim)
        
        # Warmup
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(warmup):
            _ = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        torch.cuda.synchronize()
        result.warmup_ms = (time.perf_counter() - start) * 1000
        
        # Timed runs
        times = []
        for _ in range(rep):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        import numpy as np
        times = np.array(times)
        
        result.avg_ms = float(times.mean())
        result.min_ms = float(times.min())
        result.max_ms = float(times.max())
        
        flops = attention_flops(batch, nheads, seqlen, headdim, causal)
        result.tflops = flops / result.avg_ms / 1e9
        
        hw = get_hardware_info()
        result.tc_util_pct = result.tflops / hw["peak_tflops"] * 100
        
        # Correctness check
        q_ref = q.transpose(1, 2)  # [b, h, s, d]
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        ref_out = F.scaled_dot_product_attention(
            q_ref.float(), k_ref.float(), v_ref.float(),
            scale=scale, is_causal=causal
        ).transpose(1, 2).to(dtype)
        
        result.max_diff = (out.float() - ref_out.float()).abs().max().item()
        
    except Exception as e:
        import traceback
        result.error = f"{e}\n{traceback.format_exc()[-300:]}"
    
    return result


def run_comparison(seqlens: List[int]) -> List[BenchmarkResult]:
    """运行对比测试"""
    
    results = []
    hw = get_hardware_info()
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    
    print("\n" + "=" * 80)
    print("  FlashAttention-4 vs PyTorch SDPA Performance")
    print("=" * 80)
    print(f"  GPU: {device_name}")
    print(f"  Peak TFLOPs: {hw['peak_tflops']:.0f}")
    print("=" * 80)
    
    print(f"\n  {'Seqlen':<10} {'FA4 (TF)':<12} {'SDPA (TF)':<12} {'Speedup':<10} {'FA4 util%':<10}")
    print("  " + "-" * 70)
    
    for seqlen in seqlens:
        batch = max(1, 32768 // seqlen)
        nheads = 16
        headdim = 128
        
        # FA4
        fa4_result = benchmark_fa4(batch, seqlen, nheads, headdim, causal=True)
        results.append(fa4_result)
        
        # SDPA
        sdpa_result = benchmark_sdpa(batch, seqlen, nheads, headdim, causal=True)
        results.append(sdpa_result)
        
        if fa4_result.error:
            fa4_cell = "ERR"
        else:
            fa4_cell = f"{fa4_result.tflops:.0f}"
        
        if sdpa_result.error:
            sdpa_cell = "ERR"
        else:
            sdpa_cell = f"{sdpa_result.tflops:.0f}"
        
        if fa4_result.tflops and sdpa_result.tflops and not fa4_result.error and not sdpa_result.error:
            speedup = fa4_result.tflops / sdpa_result.tflops
            speedup_cell = f"{speedup:.1f}x"
        else:
            speedup_cell = "N/A"
        
        util_cell = f"{fa4_result.tc_util_pct:.0f}" if fa4_result.tc_util_pct else "N/A"
        
        print(f"  {seqlen:<10} {fa4_cell:<12} {sdpa_cell:<12} {speedup_cell:<10} {util_cell:<10}")
    
    return results


def save_results(results: List[BenchmarkResult], path: str):
    """保存结果"""
    data = [asdict(r) for r in results]
    
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        import csv
        with open(path, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    print(f"\nResults saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FA4 Performance Comparison")
    parser.add_argument("--seqlen", type=lambda s: [int(x) for x in s.split(",")],
                       default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    
    results = run_comparison(args.seqlen)
    
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
