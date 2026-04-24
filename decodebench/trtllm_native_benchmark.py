#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT-LLM native paged decode attention.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--kv-dtype", required=True, choices=["bf16", "fp16"])
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
    raise ValueError(f"TensorRT-LLM native benchmark supports bf16/fp16, got: {name}")


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


class TRTLLMNativeDecodeKernel:
    def __init__(self, args: argparse.Namespace):
        os.environ.setdefault("TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION", "0")

        import torch

        try:
            import tensorrt_llm  # noqa: F401
            from tensorrt_llm._torch.attention_backend.interface import (
                AttentionInputType,
                PredefinedAttentionMask,
            )
            from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionWrapper
        except ImportError as error:
            raise ImportError("TensorRT-LLM is not installed or its PyTorch attention backend is unavailable") from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if args.num_q_heads % args.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.torch = torch
        self.AttentionInputType = AttentionInputType
        self.PredefinedAttentionMask = PredefinedAttentionMask
        self.args = args
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(torch, args.kv_dtype)
        self.pages_per_seq = math.ceil(args.context_len / args.page_size)
        self.num_pages = args.batch_size * self.pages_per_seq
        self.max_blocks_per_seq = self.pages_per_seq

        self.kv_cache = torch.randn(
            self.num_pages,
            2,
            args.page_size,
            args.num_kv_heads,
            args.head_dim,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()
        block_ids = torch.arange(self.num_pages, dtype=torch.int32, device=self.device).reshape(
            args.batch_size, self.pages_per_seq
        )
        self.kv_cache_block_offsets = torch.empty(
            1,
            args.batch_size,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_cache_block_offsets[0, :, 0, :] = block_ids
        self.kv_cache_block_offsets[0, :, 1, :] = block_ids

        self.host_kv_cache_pool_pointers = torch.tensor(
            [[self.kv_cache.data_ptr(), 0]], dtype=torch.int64, device="cpu"
        )
        self.host_kv_cache_pool_mapping = torch.zeros(1, 1, dtype=torch.int32, device="cpu")

        self.sequence_length = torch.full(
            (args.batch_size,), args.context_len, dtype=torch.int32, device=self.device
        )
        self.host_past_key_value_lengths = torch.full(
            (args.batch_size,), args.context_len, dtype=torch.int32, device="cpu"
        )
        self.host_total_kv_lens = torch.tensor([0, args.batch_size * args.context_len], dtype=torch.int32, device="cpu")
        self.context_lengths = torch.ones(args.batch_size, dtype=torch.int32, device=self.device)
        self.host_context_lengths = torch.ones(args.batch_size, dtype=torch.int32, device="cpu")
        self.host_request_types = torch.ones(args.batch_size, dtype=torch.int32, device="cpu")
        self.workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=self.device)

        qkv_hidden = (args.num_q_heads + 2 * args.num_kv_heads) * args.head_dim
        self.qkv = torch.randn(args.batch_size, qkv_hidden, dtype=self.dtype, device=self.device)
        self.output = torch.empty(
            args.batch_size,
            args.num_q_heads * args.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.wrapper = TrtllmAttentionWrapper(
            num_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_dim,
        )
        self.wrapper.plan(
            layer_idx=0,
            tokens_per_block=args.page_size,
            max_num_requests=args.batch_size,
            max_sequence_length=args.context_len + 1,
            max_context_length=args.context_len,
            attention_window_size=args.context_len + 1,
            sink_token_length=0,
            beam_width=1,
            sequence_length=self.sequence_length,
            host_past_key_value_lengths=self.host_past_key_value_lengths,
            host_total_kv_lens=self.host_total_kv_lens,
            context_lengths=self.context_lengths,
            host_context_lengths=self.host_context_lengths,
            host_request_types=self.host_request_types,
            kv_cache_block_offsets=self.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=self.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=self.host_kv_cache_pool_mapping,
            workspace=self.workspace,
            use_paged_context_fmha=False,
            attention_input_type=AttentionInputType.generation_only,
        )

    def run(self):
        return self.wrapper.run(
            self.qkv,
            self.output,
            is_fused_qkv=True,
            update_kv_cache=False,
            attention_mask=self.PredefinedAttentionMask.CAUSAL,
            num_contexts=0,
            num_ctx_tokens=0,
        )


def main() -> int:
    args = parse_args()
    kernel = TRTLLMNativeDecodeKernel(args)
    avg_us, p50_us, p95_us = time_kernel(kernel.torch, kernel.run, args.warmup_steps, args.repeat)
    dtype_bytes = 2
    kv_bytes = args.batch_size * args.context_len * args.num_kv_heads * args.head_dim * 2 * dtype_bytes
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": kernel.torch.cuda.get_device_name(0),
        "backend": "tensorrt_llm_native",
        "kernel_path": "tensorrt_llm._torch.attention_backend.trtllm.TrtllmAttentionWrapper native thop.attention",
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
        "peak_allocated_gb": float(kernel.torch.cuda.max_memory_allocated()) / 1e9,
        "selected_backend": "trtllm_native_thop_attention",
        "fallback": False,
        "fallback_reason": None,
        "notes": "TensorRT-LLM native PyTorch attention wrapper path. Forces TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=0 to avoid FlashInfer trtllm-gen dispatch.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
