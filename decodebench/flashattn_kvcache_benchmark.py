#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flashattn_kvcache_kernel import DecodeShape, FlashAttnKVCacheKernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark flash-attention flash_attn_with_kvcache decode kernel.")
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
    shape = DecodeShape(
        batch_size=args.batch_size,
        context_len=args.context_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        kv_dtype=args.kv_dtype,
        page_size=args.page_size,
    )
    kernel = FlashAttnKVCacheKernel(shape)
    avg_us, p50_us, p95_us = time_kernel(kernel.torch, kernel.run, args.warmup_steps, args.repeat)
    peak_allocated_gb = float(kernel.torch.cuda.max_memory_allocated()) / 1e9
    bandwidth = shape.kv_bytes / p50_us / 1e3 if p50_us > 0 else None

    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": kernel.torch.cuda.get_device_name(0),
        "backend": "flashattn_kvcache",
        "kernel_path": "flash_attn.flash_attn_with_kvcache",
        "layer": "kernel",
        "workload_id": os.environ.get("DECODEBENCH_WORKLOAD_ID"),
        "attention": shape.attention,
        "kv_dtype": shape.kv_dtype,
        "page_size": shape.page_size,
        "batch_size": shape.batch_size,
        "context_len": shape.context_len,
        "decode_steps": 1,
        "kernel_latency_avg_us": avg_us,
        "kernel_latency_p50_us": p50_us,
        "kernel_latency_p95_us": p95_us,
        "approx_kv_bytes_read": shape.kv_bytes,
        "approx_effective_kv_bandwidth_gb_s": bandwidth,
        "peak_allocated_gb": peak_allocated_gb,
        "selected_backend": "flash_attn_with_kvcache",
        "fallback": False,
        "fallback_reason": None,
        "notes": "Dense KV-cache decode benchmark via flash_attn_with_kvcache. page_size is ignored for execution and kept only for matrix compatibility.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
