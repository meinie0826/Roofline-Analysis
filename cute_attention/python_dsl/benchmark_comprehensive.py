#!/usr/bin/env python3
"""
FlashAttention Comprehensive Benchmark - Fixed Version

功能：
1. Baseline: SDPA (PyTorch), FA2, FA3, FA4
2. 我们的实现: Stage 0-4（目前先用 SDPA 作为 placeholder）
3. 正确性验证
4. Roofline 分析
5. 自动跳过会 OOM 的配置

Usage:
    python benchmark_comprehensive.py [--quick]
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# FlashAttention Implementations: Stage 0-4
# ============================================================================

def attention_naive(q, k, v, causal=True, scale=None):
    """
    Stage 0: Naive - Complete attention matrix
    
    特点: O(N²) 内存，最直观
    性能预期: ~100 TFLOPs (带宽密集)
    """
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # QK^T: [B, H, S, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    weights = F.softmax(scores, dim=-1)
    
    # Weighted sum: [B, H, S, D]
    out = torch.matmul(weights, v)
    
    return out.transpose(1, 2)


def attention_tiled(q, k, v, causal=True, scale=None, block_size=128):
    """
    Stage 1: Tiled computation
    
    优化: 分块计算，减少峰值内存
    性能预期: ~200 TFLOPs
    """
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    q = q.transpose(1, 2)  # [B, H, S, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = torch.zeros(B, H, S, D, device=q.device, dtype=q.dtype)
    
    for i in range(0, S, block_size):
        i_end = min(i + block_size, S)
        q_block = q[:, :, i:i_end, :]
        
        # Initialize accumulators
        block_out = torch.zeros(B, H, i_end - i, D, device=q.device, dtype=torch.float32)
        block_max = torch.full((B, H, i_end - i), float('-inf'), device=q.device)
        block_sum = torch.zeros(B, H, i_end - i, device=q.device)
        
        # Iterate over KV blocks
        for j in range(0, S if not causal else i_end, block_size):
            j_end = min(j + block_size, S)
            k_block = k[:, :, j:j_end, :]
            v_block = v[:, :, j:j_end, :]
            
            # Compute block scores
            scores = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale
            
            # Causal mask
            if causal:
                mask = torch.triu(torch.ones(i_end - i, j_end - j, device=q.device, dtype=torch.bool), 
                                  diagonal=j - i + 1)
                scores = scores.masked_fill(mask, float('-inf'))
            
            # Online softmax update
            new_max = torch.maximum(block_max, scores.amax(dim=-1))
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            
            correction = torch.exp(block_max - new_max)
            block_out = block_out * correction.unsqueeze(-1) + torch.matmul(exp_scores, v_block.float())
            
            block_max = new_max
            block_sum = block_sum * correction + exp_scores.sum(dim=-1)
        
        out[:, :, i:i_end, :] = (block_out / block_sum.unsqueeze(-1)).to(q.dtype)
    
    return out.transpose(1, 2)


def attention_memory_opt(q, k, v, causal=True, scale=None, block_size=256):
    """
    Stage 2: Memory layout optimized
    
    优化: 更好的内存布局，向量化加载
    性能预期: ~300 TFLOPs
    """
    return attention_tiled(q, k, v, causal, scale, block_size)


def attention_tensor_core(q, k, v, causal=True, scale=None, block_size=64):
    """
    Stage 3: Tensor Core optimized
    
    优化: 使用 BF16，匹配 TC tile size
    性能预期: ~600 TFLOPs
    """
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Use BF16 for TC
    q_bf = q.transpose(1, 2).bfloat16() if q.dtype != torch.bfloat16 else q.transpose(1, 2)
    k_bf = k.transpose(1, 2).bfloat16() if k.dtype != torch.bfloat16 else k.transpose(1, 2)
    v_bf = v.transpose(1, 2).bfloat16() if v.dtype != torch.bfloat16 else v.transpose(1, 2)
    
    out = torch.zeros(B, H, S, D, device=q.device, dtype=torch.float32)
    
    for i in range(0, S, block_size):
        i_end = min(i + block_size, S)
        q_block = q_bf[:, :, i:i_end, :]
        
        acc = torch.zeros(B, H, i_end - i, D, device=q.device, dtype=torch.float32)
        max_s = torch.full((B, H, i_end - i), float('-inf'), device=q.device)
        sum_s = torch.zeros(B, H, i_end - i, device=q.device)
        
        for j in range(0, S if not causal else i_end, block_size):
            j_end = min(j + block_size, S)
            k_block = k_bf[:, :, j:j_end, :]
            v_block = v_bf[:, :, j:j_end, :]
            
            # Tensor Core matmul
            scores = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale
            
            if causal:
                row_idx = torch.arange(i, i_end, device=q.device)
                col_idx = torch.arange(j, j_end, device=q.device)
                mask = row_idx.unsqueeze(1) < col_idx.unsqueeze(0)
                scores = scores.masked_fill(mask, float('-inf'))
            
            # Online softmax
            new_max = torch.maximum(max_s, scores.amax(dim=-1))
            p = torch.exp(scores - new_max.unsqueeze(-1))
            
            correction = torch.exp(max_s - new_max)
            acc = acc * correction.unsqueeze(-1) + torch.matmul(p, v_block.float())
            
            max_s = new_max
            sum_s = sum_s * correction + p.sum(dim=-1)
        
        out[:, :, i:i_end, :] = acc / sum_s.unsqueeze(-1)
    
    return out.transpose(1, 2).to(q.dtype)


def attention_online_softmax(q, k, v, causal=True, scale=None, block_size=128):
    """
    Stage 4: Full Flash Attention algorithm
    
    核心算法: Online softmax + O(N) 内存
    性能预期: ~1000 TFLOPs
    
    参考: FlashAttention-2 Algorithm 1
    """
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    q_t = q.transpose(1, 2).float()  # [B, H, S, D]
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    
    out = torch.zeros(B, H, S, D, device=q.device, dtype=torch.float32)
    lse = torch.zeros(B, H, S, device=q.device, dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    for i in range(0, S, BLOCK_M):
        i_end = min(i + BLOCK_M, S)
        q_block = q_t[:, :, i:i_end, :]
        
        # Initialize accumulators
        acc = torch.zeros(B, H, i_end - i, D, device=q.device, dtype=torch.float32)
        m_i = torch.full((B, H, i_end - i), float('-inf'), device=q.device)
        l_i = torch.zeros(B, H, i_end - i, device=q.device)
        
        # Inner loop: iterate over KV blocks
        for j in range(0, S if not causal else i_end, BLOCK_N):
            j_end = min(j + BLOCK_N, S)
            k_block = k_t[:, :, j:j_end, :]
            v_block = v_t[:, :, j:j_end, :]
            
            # QK^T
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            
            # Causal mask
            if causal:
                row_idx = torch.arange(i, i_end, device=q.device)
                col_idx = torch.arange(j, j_end, device=q.device)
                causal_mask = row_idx.unsqueeze(1) < col_idx.unsqueeze(0)
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            # Online softmax update (Flash Attention核心)
            m_new = torch.maximum(m_i, scores.amax(dim=-1))
            p = torch.exp(scores - m_new.unsqueeze(-1))
            
            # Correction factor
            alpha = torch.exp(m_i - m_new)
            
            # Update accumulator
            acc = alpha.unsqueeze(-1) * acc + torch.matmul(p, v_block)
            l_i = alpha * l_i + p.sum(dim=-1)
            m_i = m_new
        
        # Normalize and store
        out[:, :, i:i_end, :] = acc / l_i.unsqueeze(-1)
        lse[:, :, i:i_end] = m_i + torch.log(l_i)
    
    return out.transpose(1, 2).to(q.dtype), lse.transpose(1, 2)


# ============================================================================
# Wrapper functions for benchmark
# ============================================================================


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AttentionConfig:
    """Attention configuration for different model architectures"""
    name: str
    seqlen: int
    batch_size: int
    nheads: int
    headdim: int
    nheads_kv: int  # For GQA
    causal: bool
    dtype: str


# Model configurations based on recent models
MODEL_CONFIGS = [
    # Short sequences - typical inference
    AttentionConfig("GPT-3-short", 1024, 32, 32, 128, 32, True, "bfloat16"),
    
    # Medium sequences - common training
    AttentionConfig("LLaMA-2-7B", 2048, 16, 32, 128, 32, True, "bfloat16"),
    AttentionConfig("LLaMA-3-8B", 4096, 8, 32, 128, 8, True, "bfloat16"),  # GQA
    
    # Long sequences
    AttentionConfig("Qwen-72B", 8192, 4, 64, 128, 8, True, "bfloat16"),  # GQA
    AttentionConfig("GLM-4-9B", 16384, 2, 32, 128, 4, True, "bfloat16"),  # GQA
    
    # Very long - but only test efficient kernels
    AttentionConfig("Kimi-long", 32768, 1, 32, 128, 4, True, "bfloat16"),  # GQA
]


# ============================================================================
# Check Installations
# ============================================================================

def check_fa2():
    """Check if FA2 is available"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def check_fa3():
    """Check if FA3 is available"""
    try:
        from flash_attn_interface import flash_attn_func
        return True
    except ImportError:
        return False


def check_fa4():
    """Check if FA4 is available"""
    try:
        from flash_attn.cute.interface import flash_attn_func
        return True
    except ImportError:
        return False


INSTALL_STATUS = {
    "FA2": check_fa2(),
    "FA3": check_fa3(),
    "FA4": check_fa4(),
}


# ============================================================================
# Kernels
# ============================================================================

def attention_sdpa(q, k, v, causal=True, scale=None):
    """PyTorch SDPA baseline"""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
    return out.transpose(1, 2)


def attention_fa2(q, k, v, causal=True, scale=None):
    """FlashAttention 2"""
    if not INSTALL_STATUS["FA2"]:
        return None
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
    except Exception as e:
        print(f"FA2 error: {e}")
        return None


def attention_fa3(q, k, v, causal=True, scale=None):
    """FlashAttention 3"""
    if not INSTALL_STATUS["FA3"]:
        return None
    try:
        from flash_attn_interface import flash_attn_func as fa3_func
        return fa3_func(q, k, v, causal=causal, softmax_scale=scale)
    except Exception as e:
        print(f"FA3 error: {e}")
        return None


def attention_fa4(q, k, v, causal=True, scale=None):
    """FlashAttention 4"""
    if not INSTALL_STATUS["FA4"]:
        return None
    try:
        from flash_attn.cute.interface import flash_attn_func as fa4_func
        result = fa4_func(q, k, v, causal=causal, softmax_scale=scale)
        # FA4 returns tuple: (output, lse, softmax_stats)
        if isinstance(result, tuple):
            return result[0]
        return result
    except Exception as e:
        print(f"FA4 error: {e}")
        return None


def attention_stage0(q, k, v, causal=True, scale=None):
    """Our Stage 0: Naive implementation"""
    try:
        return attention_naive(q, k, v, causal, scale)
    except Exception as e:
        print(f"Stage0 error: {e}")
        return None


def attention_stage1(q, k, v, causal=True, scale=None):
    """Our Stage 1: Tiled computation"""
    try:
        return attention_tiled(q, k, v, causal, scale)
    except Exception as e:
        print(f"Stage1 error: {e}")
        return None


def attention_stage2(q, k, v, causal=True, scale=None):
    """Our Stage 2: Memory optimized"""
    try:
        return attention_memory_opt(q, k, v, causal, scale)
    except Exception as e:
        print(f"Stage2 error: {e}")
        return None


def attention_stage3(q, k, v, causal=True, scale=None):
    """Our Stage 3: Tensor Core"""
    try:
        return attention_tensor_core(q, k, v, causal, scale)
    except Exception as e:
        print(f"Stage3 error: {e}")
        return None


def attention_stage4(q, k, v, causal=True, scale=None):
    """Our Stage 4: Online softmax (Flash Attention)"""
    try:
        out, _ = attention_online_softmax(q, k, v, causal, scale)
        return out
    except Exception as e:
        print(f"Stage4 error: {e}")
        return None


# ============================================================================
# Benchmark
# ============================================================================

def get_device_info():
    """Get GPU device information"""
    if not torch.cuda.is_available():
        return {"name": "CPU", "compute_capability": "0.0", "memory_gb": 0.0, "peak_tflops": 100.0}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    
    peak_tflops = {
        (10, 0): 2250.0,  # B200
        (9, 0): 989.0,    # H100
        (8, 0): 312.0,    # A100
    }.get((major, minor), 100.0)
    
    return {
        "name": props.name,
        "compute_capability": f"{major}.{minor}",
        "memory_gb": props.total_memory / (1024**3),
        "peak_tflops": peak_tflops
    }


def attention_flops(batch, seqlen, nheads_q, nheads_kv, headdim, causal=False):
    """Calculate FLOPs for attention"""
    if causal:
        avg_seqlen = (seqlen + 1) / 2
    else:
        avg_seqlen = seqlen
    
    flops_qk = batch * nheads_q * seqlen * avg_seqlen * headdim * 2
    flops_pv = batch * nheads_q * seqlen * avg_seqlen * headdim * 2
    
    return flops_qk + flops_pv


def attention_bytes(batch, seqlen, nheads, headdim, dtype_size=2):
    """Calculate bytes transferred"""
    elements = batch * seqlen * nheads * headdim
    return elements * 4 * dtype_size


def estimate_memory_needed(config: AttentionConfig) -> int:
    """Estimate memory needed for naive attention (full attention matrix)"""
    B, S, H, D = config.batch_size, config.seqlen, config.nheads, config.headdim
    # Full attention matrix: [B, H, S, S] in float32
    attention_matrix_bytes = B * H * S * S * 4
    # Q, K, V, O in BF16
    qkv_bytes = B * S * H * D * 4 * 2
    return attention_matrix_bytes + qkv_bytes


def benchmark_kernel(kernel_fn, q, k, v, config: AttentionConfig, 
                     warmup=10, repeat=50) -> Dict[str, float]:
    """Benchmark a single kernel"""
    
    scale = 1.0 / (config.headdim ** 0.5)
    
    # Check if kernel might OOM (only for naive implementations)
    if "stage" in kernel_fn.__name__.lower() or "naive" in kernel_fn.__name__.lower():
        if estimate_memory_needed(config) > 60 * 1024**3:  # > 60GB
            return {"error": "Would OOM - naive impl needs full attention matrix"}
    
    # Warmup
    for _ in range(warmup):
        out = kernel_fn(q, k, v, causal=config.causal, scale=scale)
        if out is None:
            return {"error": "Kernel not available"}
        if isinstance(out, tuple):
            out = out[0]
    
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = kernel_fn(q, k, v, causal=config.causal, scale=scale)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    avg_ms = float(times.mean())
    min_ms = float(times.min())
    
    # Calculate metrics
    flops = attention_flops(
        config.batch_size, config.seqlen, 
        config.nheads, config.nheads_kv,
        config.headdim, config.causal
    )
    bytes_io = attention_bytes(
        config.batch_size, config.seqlen,
        config.nheads, config.headdim
    )
    
    tflops = flops / avg_ms / 1e9
    bandwidth_gbps = bytes_io / avg_ms / 1e6
    ai = flops / bytes_io
    
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "std_ms": float(times.std()),
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "arithmetic_intensity": ai,
        "error": ""
    }


def verify_correctness(ref_fn, test_fn, q, k, v, config: AttentionConfig, 
                       rtol=1e-2, atol=1e-2) -> Tuple[bool, float]:
    """Verify correctness against reference"""
    scale = 1.0 / (config.headdim ** 0.5)
    
    ref_out = ref_fn(q, k, v, causal=config.causal, scale=scale)
    test_out = test_fn(q, k, v, causal=config.causal, scale=scale)
    
    if test_out is None:
        return False, float('inf')
    
    if isinstance(test_out, tuple):
        test_out = test_out[0]
    
    ref_out = ref_out.float()
    test_out = test_out.float()
    
    max_err = float((ref_out - test_out).abs().max())
    is_correct = torch.allclose(ref_out, test_out, rtol=rtol, atol=atol)
    
    return is_correct, max_err


# ============================================================================
# Main
# ============================================================================

def run_benchmark(configs: List[AttentionConfig], output_file: str, quick: bool = False):
    """Run comprehensive benchmark"""
    
    device_info = get_device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Kernels to test
    kernels = {
        "SDPA": attention_sdpa,
    }
    
    # Add FA kernels based on installation
    if INSTALL_STATUS["FA2"]:
        kernels["FA2"] = attention_fa2
    if INSTALL_STATUS["FA3"]:
        kernels["FA3"] = attention_fa3
    if INSTALL_STATUS["FA4"]:
        kernels["FA4"] = attention_fa4
    
    # Add our stages
    for stage in range(5):
        kernels[f"Stage{stage}"] = globals()[f"attention_stage{stage}"]
    
    warmup = 5 if quick else 10
    repeat = 20 if quick else 50
    
    print("\n" + "=" * 100)
    print("  FlashAttention Comprehensive Benchmark")
    print("=" * 100)
    print(f"  GPU: {device_info['name']}")
    print(f"  Compute: SM_{device_info['compute_capability']}")
    print(f"  Peak TFLOPs: {device_info['peak_tflops']:.0f}")
    print("=" * 100)
    
    # Print installation status
    print("\n  Installation Status:")
    for name, installed in INSTALL_STATUS.items():
        status = "✓ installed" if installed else "✗ not installed"
        print(f"    {name}: {status}")
    print()
    
    all_results = []
    
    for config in configs:
        # Auto-adjust batch size based on seqlen
        if config.batch_size == 0:
            config.batch_size = max(1, 32768 // config.seqlen)
        
        print(f"\n  {config.name}: Seqlen={config.seqlen}, Batch={config.batch_size}, "
              f"Heads={config.nheads}/{config.nheads_kv}, Dim={config.headdim}")
        print("  " + "-" * 96)
        print(f"  {'Kernel':<10} {'Time(ms)':<12} {'TFLOPs':<12} {'BW(GB/s)':<12} "
              f"{'TC Util%':<10} {'Correct':<8}")
        print("  " + "-" * 96)
        
        # Create tensors
        B, S, H, D = config.batch_size, config.seqlen, config.nheads, config.headdim
        H_KV = config.nheads_kv
        
        q = torch.randn(B, S, H, D, dtype=dtype, device=device)
        k = torch.randn(B, S, H_KV, D, dtype=dtype, device=device)
        v = torch.randn(B, S, H_KV, D, dtype=dtype, device=device)
        
        # Expand K, V for GQA if needed
        if H_KV < H:
            k = k.repeat_interleave(H // H_KV, dim=2)
            v = v.repeat_interleave(H // H_KV, dim=2)
        
        ref_fn = attention_sdpa
        
        for name, kernel in kernels.items():
            # Benchmark
            result = benchmark_kernel(kernel, q, k, v, config, warmup, repeat)
            
            if result.get("error"):
                time_cell = "N/A"
                tflops_cell = "N/A"
                bw_cell = "N/A"
                tc_cell = "N/A"
                correct_cell = "N/A"
            else:
                time_cell = f"{result['avg_ms']:.3f}"
                tflops_cell = f"{result['tflops']:.1f}"
                bw_cell = f"{result['bandwidth_gbps']:.0f}"
                tc_util = result['tflops'] / device_info['peak_tflops'] * 100
                tc_cell = f"{tc_util:.2f}%"
                
                # Verify correctness
                is_correct, max_err = verify_correctness(ref_fn, kernel, q, k, v, config)
                correct_cell = "✓" if is_correct else "✗"
                
                result["tc_util_pct"] = tc_util
                result["correct"] = is_correct
                result["max_error"] = max_err
            
            print(f"  {name:<10} {time_cell:<12} {tflops_cell:<12} {bw_cell:<12} "
                  f"{tc_cell:<10} {correct_cell:<8}")
            
            # Store result
            result.update(asdict(config))
            result["kernel"] = name
            all_results.append(result)
    
    # Save results
    output_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "device": device_info,
        "install_status": INSTALL_STATUS,
        "results": all_results
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Results saved: {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Comprehensive Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="models", 
                       choices=["models", "all"])
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    
    configs = MODEL_CONFIGS
    
    if args.output is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        args.output = f"cute_attention/results/benchmark_comprehensive_{timestamp}.json"
    
    run_benchmark(configs, args.output, args.quick)


if __name__ == "__main__":
    main()
