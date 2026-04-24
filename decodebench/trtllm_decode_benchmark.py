#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trtllm_decode_kernel import DecodeShape, FlashInferTRTLLMDecodeKernel


def cuda_time_ms(torch, fn, repeat: int) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, math.ceil(pct / 100 * len(values)) - 1))
    return values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TRTLLM decode path exposed by FlashInfer.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--kv-dtype", required=True, choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--page-size", type=int, required=True, choices=[16, 32, 64, 128])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    kernel = FlashInferTRTLLMDecodeKernel(shape)
    torch = kernel.torch

    for _ in range(args.warmup_steps):
        kernel.run()
    torch.cuda.synchronize()

    times_ms = cuda_time_ms(torch, kernel.run, args.repeat)
    avg_ms = sum(times_ms) / len(times_ms)
    p50_ms = percentile(times_ms, 50)
    p95_ms = percentile(times_ms, 95)

    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "backend": "flashinfer_trtllm_decode",
        "kernel_path": "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
        "layer": "kernel",
        "workload_id": os.environ.get("DECODEBENCH_WORKLOAD_ID"),
        "attention": shape.attention,
        "kv_dtype": shape.kv_dtype,
        "page_size": shape.page_size,
        "batch_size": shape.batch_size,
        "context_len": shape.context_len,
        "decode_steps": 1,
        "compare_latency_us": p50_ms * 1000.0,
        "kernel_latency_avg_us": avg_ms * 1000.0,
        "kernel_latency_p50_us": p50_ms * 1000.0,
        "kernel_latency_p95_us": p95_ms * 1000.0,
        "approx_kv_bytes_read": shape.kv_bytes,
        "approx_effective_kv_bandwidth_gb_s": shape.kv_bytes / (p50_ms * 1e6),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "selected_backend": "flashinfer_trtllm_decode",
        "fallback": False,
        "fallback_reason": None,
        "notes": "TRTLLM decode path benchmarked through FlashInfer's trtllm_batch_decode_with_kv_cache.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
