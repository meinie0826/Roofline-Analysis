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

from third_party_paths import find_vllm_benchmark_dir, find_vllm_python
from vllm_attention_kernel import DecodeShape, VLLMAttentionBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM attention backend using vLLM's attention benchmark suite.")
    parser.add_argument("--backend", required=True, choices=["flash", "flashinfer"])
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
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--python-bin")
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
    bench = VLLMAttentionBenchmark(
        backend=args.backend,
        shape=shape,
        benchmark_dir=find_vllm_benchmark_dir(
            str(args.benchmark_dir) if args.benchmark_dir else None
        ),
        python_bin=find_vllm_python(args.python_bin),
    )
    row = bench.run(repeats=args.repeat, warmup_steps=args.warmup_steps)

    mean_time_s = row["mean_time"]
    if row.get("error") or not isinstance(mean_time_s, (int, float)) or not math.isfinite(mean_time_s):
        raise RuntimeError(str(row.get("error") or f"invalid vLLM benchmark mean_time: {mean_time_s}"))
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": row.get("config", {}).get("device", "cuda"),
        "backend": f"vllm_{args.backend}",
        "kernel_path": f"vLLM attention_benchmarks backend={args.backend}",
        "layer": "kernel_or_framework",
        "workload_id": os.environ.get("DECODEBENCH_WORKLOAD_ID"),
        "attention": shape.attention,
        "kv_dtype": shape.kv_dtype,
        "page_size": shape.page_size,
        "batch_size": shape.batch_size,
        "context_len": shape.context_len,
        "decode_steps": 1,
        "compare_latency_us": mean_time_s * 1e6,
        "kernel_latency_avg_us": mean_time_s * 1e6,
        "kernel_latency_p50_us": mean_time_s * 1e6,
        "kernel_latency_p95_us": row["max_time"] * 1e6,
        "approx_kv_bytes_read": None,
        "approx_effective_kv_bandwidth_gb_s": None,
        "peak_allocated_gb": (row.get("memory_allocated_mb") or 0) / 1024,
        "selected_backend": args.backend,
        "fallback": False,
        "fallback_reason": None,
        "notes": "Metric is vLLM attention benchmark mean_time, not CUDA-event p50. Useful for cross-backend comparison within vLLM benchmark suite.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
