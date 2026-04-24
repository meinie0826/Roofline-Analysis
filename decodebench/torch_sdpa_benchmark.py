#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"torch SDPA reference supports bf16/fp16, got: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch scaled_dot_product_attention reference decode.")
    parser.add_argument("--backend", choices=["auto", "cudnn", "flash", "math"], default="auto")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--kv-dtype", required=True, choices=["bf16", "fp16"])
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def attention_name(num_q_heads: int, num_kv_heads: int) -> str:
    if num_q_heads == num_kv_heads:
        return "MHA"
    if num_kv_heads == 1:
        return "MQA"
    return "GQA"


def backend_context(torch, backend: str):
    if backend == "auto":
        return nullcontext()
    sdpa_kernel = getattr(torch.nn.attention, "sdpa_kernel", None)
    sdpa_backend = getattr(torch.nn.attention, "SDPBackend", None)
    if sdpa_kernel is None or sdpa_backend is None:
        raise RuntimeError("torch.nn.attention.sdpa_kernel is unavailable")
    selected = {
        "cudnn": sdpa_backend.CUDNN_ATTENTION,
        "flash": sdpa_backend.FLASH_ATTENTION,
        "math": sdpa_backend.MATH,
    }[backend]
    return sdpa_kernel(selected)


def time_kernel(torch, fn, warmup_steps: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup_steps):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    samples_us.sort()
    avg_us = sum(samples_us) / len(samples_us)
    p50_us = samples_us[len(samples_us) // 2]
    p95_us = samples_us[min(len(samples_us) - 1, int(len(samples_us) * 0.95))]
    return avg_us, p50_us, p95_us


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    device = torch.device("cuda")
    dtype = dtype_from_name(torch, args.kv_dtype)
    q = torch.randn(args.batch_size, args.num_q_heads, 1, args.head_dim, dtype=dtype, device=device)
    k = torch.randn(args.batch_size, args.num_kv_heads, args.context_len, args.head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    enable_gqa = args.num_q_heads != args.num_kv_heads

    def run():
        with backend_context(torch, args.backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    avg_us, p50_us, p95_us = time_kernel(torch, run, args.warmup_steps, args.repeat)
    dtype_bytes = 2
    kv_bytes = args.batch_size * args.context_len * args.num_kv_heads * args.head_dim * 2 * dtype_bytes
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "backend": f"torch_sdpa_{args.backend}",
        "kernel_path": f"torch.nn.functional.scaled_dot_product_attention backend={args.backend}",
        "layer": "framework_reference",
        "workload_id": os.environ.get("DECODEBENCH_WORKLOAD_ID"),
        "attention": attention_name(args.num_q_heads, args.num_kv_heads),
        "kv_dtype": args.kv_dtype,
        "page_size": args.page_size,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "decode_steps": 1,
        "compare_latency_us": p50_us,
        "kernel_latency_avg_us": avg_us,
        "kernel_latency_p50_us": p50_us,
        "kernel_latency_p95_us": p95_us,
        "approx_kv_bytes_read": kv_bytes,
        "approx_effective_kv_bandwidth_gb_s": kv_bytes / p50_us / 1e3 if p50_us > 0 else None,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated()) / 1e9,
        "selected_backend": f"torch_sdpa_{args.backend}",
        "fallback": False,
        "fallback_reason": None,
        "notes": "Framework/reference SDPA decode timing. Uses dense KV tensors, not paged KV cache.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
