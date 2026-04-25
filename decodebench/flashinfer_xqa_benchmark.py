#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer direct XQA paged decode kernel.")
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


def dtype_from_name(torch, name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported kv dtype: {name}")


def attention_name(num_q_heads: int, num_kv_heads: int) -> str:
    if num_q_heads == num_kv_heads:
        return "MHA"
    if num_kv_heads == 1:
        return "MQA"
    return "GQA"


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


class FlashInferXQADecodeKernel:
    def __init__(self, args: argparse.Namespace):
        import torch

        try:
            import flashinfer
        except ImportError as error:
            raise ImportError(
                "FlashInfer is not installed. Install it with: "
                "python3 -m pip install flashinfer-python flashinfer-cubin"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if args.num_q_heads % args.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.torch = torch
        self.flashinfer = flashinfer
        self.args = args
        self.device = torch.device("cuda")
        self.q_dtype = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float16
        self.kv_dtype = dtype_from_name(torch, args.kv_dtype)
        self.pages_per_seq = math.ceil(args.context_len / args.page_size)
        self.num_pages = args.batch_size * self.pages_per_seq
        self.sm_scale = float(1.0 / (args.head_dim**0.5))

        self.query = torch.randn(
            args.batch_size,
            args.num_q_heads,
            args.head_dim,
            dtype=self.q_dtype,
            device=self.device,
        )
        self.k_cache, self.v_cache = self._make_kv_cache()
        self.block_tables = torch.arange(
            self.num_pages,
            dtype=torch.int32,
            device=self.device,
        ).reshape(args.batch_size, self.pages_per_seq)
        self.seq_lens = torch.full((args.batch_size,), args.context_len, dtype=torch.uint32, device=self.device)
        self.workspace_buffer = torch.zeros(1024 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.output = torch.empty_like(self.query)

    def _make_kv_cache(self):
        torch = self.torch
        args = self.args
        cache_shape = (self.num_pages, args.page_size, args.num_kv_heads, args.head_dim)
        if self.kv_dtype == torch.float8_e4m3fn:
            k_cache = torch.randn(*cache_shape, dtype=torch.float16, device=self.device).to(self.kv_dtype)
            v_cache = torch.randn(*cache_shape, dtype=torch.float16, device=self.device).to(self.kv_dtype)
            return k_cache, v_cache
        return (
            torch.randn(*cache_shape, dtype=self.kv_dtype, device=self.device),
            torch.randn(*cache_shape, dtype=self.kv_dtype, device=self.device),
        )

    def run(self):
        return self.flashinfer.decode.xqa_batch_decode_with_kv_cache(
            query=self.query,
            kv_cache=(self.k_cache, self.v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            max_seq_len=self.args.context_len,
            bmm1_scale=self.sm_scale,
            bmm2_scale=1.0,
            out=self.output,
            kv_layout="NHD",
        )


def main() -> int:
    args = parse_args()
    kernel = FlashInferXQADecodeKernel(args)
    torch = kernel.torch
    avg_us, p50_us, p95_us = time_kernel(torch, kernel.run, args.warmup_steps, args.repeat)
    dtype_bytes = 1 if args.kv_dtype == "fp8" else 2
    kv_bytes = args.batch_size * args.context_len * args.num_kv_heads * args.head_dim * 2 * dtype_bytes
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "backend": "flashinfer_xqa_decode",
        "kernel_path": "flashinfer.decode.xqa_batch_decode_with_kv_cache",
        "layer": "kernel",
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
        "selected_backend": "flashinfer_xqa_decode",
        "fallback": False,
        "fallback_reason": None,
        "notes": "FlashInfer direct XQA paged decode benchmark. No model TPS/TPOT.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
