#!/usr/bin/env python3
"""Benchmark megakernel against local and SGLang dense subgraph references."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]

from .cluster_megakernel import cluster_megakernel_forward
from .common import MegakernelConfig, available_backends
from .external_reference import (
    probe_sglang_import,
    sglang_layer_reference_forward,
    sglang_subgraph_reference_forward,
)
from .megakernel_reference import (
    make_random_megakernel_inputs,
    megakernel_reference_forward,
)


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for this benchmark.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _sync():
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


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a, b) -> float:
    return (a.float() - b.float()).abs().mean().item()


def run_benchmark(args) -> int:
    _require_torch()

    if not available_backends()["cute"]:
        raise RuntimeError("CuTe DSL is required for the megakernel benchmark.")

    status = probe_sglang_import()
    if not status.available:
        raise RuntimeError(f"SGLang unavailable: {status.error}")

    dtype = getattr(torch, args.dtype)
    config = MegakernelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.hidden_dim // args.num_heads,
        cluster_size=args.cluster_size,
        num_threads=args.num_threads,
    )
    config.validate()

    inputs = make_random_megakernel_inputs(
        config,
        seq_len=args.seq_len,
        device="cuda",
        dtype=dtype,
        seed=args.seed,
    )

    def local_ref():
        return megakernel_reference_forward(**inputs, config=config)

    def sglang_subgraph_ref():
        return sglang_subgraph_reference_forward(**inputs, config=config)

    def sglang_layer_ref():
        return sglang_layer_reference_forward(**inputs, config=config)

    def cute_kernel():
        return cluster_megakernel_forward(**inputs, config=config)

    # Compile/warm any lazy paths before correctness/benchmark reporting.
    local_out = local_ref()
    sglang_subgraph_out = sglang_subgraph_ref()
    sglang_out = sglang_layer_ref()
    cute_out = cute_kernel()
    _sync()

    print("=== Correctness vs SGLang reference ===")
    for name, actual, expected in [
        ("output", cute_out[0], sglang_out[0]),
        ("k_new", cute_out[1], sglang_out[1]),
        ("v_new", cute_out[2], sglang_out[2]),
    ]:
        print(
            f"{name}: max_abs={_max_abs(actual, expected):.6g}, "
            f"mean_abs={_mean_abs(actual, expected):.6g}"
        )

    print("\n=== Reference agreement ===")
    print(
        "local_ref vs sglang_layer_ref output: "
        f"max_abs={_max_abs(local_out[0], sglang_out[0]):.6g}, "
        f"mean_abs={_mean_abs(local_out[0], sglang_out[0]):.6g}"
    )
    print(
        "sglang_subgraph_ref vs sglang_layer_ref output: "
        f"max_abs={_max_abs(sglang_subgraph_out[0], sglang_out[0]):.6g}, "
        f"mean_abs={_mean_abs(sglang_subgraph_out[0], sglang_out[0]):.6g}"
    )

    print("\n=== Latency ===")
    for name, fn in [
        ("local_pytorch_ref", local_ref),
        ("sglang_subgraph_ref", sglang_subgraph_ref),
        ("sglang_layer_ref", sglang_layer_ref),
        ("cute_megakernel", cute_kernel),
    ]:
        ms = _time_cuda(fn, warmup=args.warmup, iters=args.iters)
        print(f"{name}: {ms:.4f} ms")

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster megakernel against SGLang dense subgraph reference."
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--cluster-size", type=int, default=2, choices=[2, 4])
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
