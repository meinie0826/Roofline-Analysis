#!/usr/bin/env python3
"""Tensor-core projection baseline for the cluster decode megakernel.

This benchmark intentionally uses PyTorch matmul/cuBLAS as the tensor-core
target line for the two fixed-cost projections in the megakernel:

  QKV: hidden_norm @ w_qkv.T   -> (1, 3 * hidden_dim)
  WO:  attention_out @ w_o.T   -> (1, hidden_dim)

It does not replace the fused CuTeDSL path.  The point is to quantify how much
room exists before wiring a CuTeDSL UMMA/TMA projection tile into the kernel.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for this benchmark.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _sync() -> None:
    torch.cuda.synchronize()


def _time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters


def _make_inputs(hidden_dim: int, num_heads: int, dtype, seed: int):
    torch.manual_seed(seed)
    hidden = torch.randn((1, hidden_dim), device="cuda", dtype=dtype)
    rms_weight = torch.randn((hidden_dim,), device="cuda", dtype=dtype)
    w_qkv = torch.randn((3 * hidden_dim, hidden_dim), device="cuda", dtype=dtype)
    w_o = torch.randn((hidden_dim, hidden_dim), device="cuda", dtype=dtype)
    attn = torch.randn((1, hidden_dim), device="cuda", dtype=dtype)

    h_f = hidden.float()
    rms = torch.rsqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    hidden_norm = (h_f * rms * rms_weight.float()).to(dtype)
    return hidden_norm, w_qkv, attn, w_o


def run(args) -> int:
    _require_torch()
    dtype = getattr(torch, args.dtype)
    hidden_norm, w_qkv, attn, w_o = _make_inputs(
        args.hidden_dim,
        args.num_heads,
        dtype,
        args.seed,
    )

    # Enable TF32 only affects fp32 matmul; fp16/bf16 uses tensor cores directly.
    torch.backends.cuda.matmul.allow_tf32 = True

    def qkv():
        return hidden_norm @ w_qkv.t()

    def wo():
        return attn @ w_o.t()

    def qkv_wo():
        qkv_out = hidden_norm @ w_qkv.t()
        # Use a prebuilt attention vector for WO; this keeps the benchmark about
        # projection cost rather than attention semantics.
        wo_out = attn @ w_o.t()
        return qkv_out, wo_out

    qkv_out, wo_out = qkv_wo()
    _sync()

    qkv_ms = _time_cuda(qkv, args.warmup, args.iters)
    wo_ms = _time_cuda(wo, args.warmup, args.iters)
    both_ms = _time_cuda(qkv_wo, args.warmup, args.iters)

    qkv_flops = 2 * args.hidden_dim * (3 * args.hidden_dim)
    wo_flops = 2 * args.hidden_dim * args.hidden_dim
    both_flops = qkv_flops + wo_flops

    def tflops(flops: int, ms: float) -> float:
        return flops / (ms * 1e-3) / 1e12

    print("=== Tensor-core projection baseline (torch/cuBLAS) ===")
    print(f"D={args.hidden_dim} H={args.num_heads} dtype={args.dtype}")
    print(f"qkv:    {qkv_ms:.4f} ms  {tflops(qkv_flops, qkv_ms):.2f} TFLOP/s  shape={tuple(qkv_out.shape)}")
    print(f"wo:     {wo_ms:.4f} ms  {tflops(wo_flops, wo_ms):.2f} TFLOP/s  shape={tuple(wo_out.shape)}")
    print(f"qkv+wo: {both_ms:.4f} ms  {tflops(both_flops, both_ms):.2f} TFLOP/s")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tensor-core QKV/WO projection baseline.")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
