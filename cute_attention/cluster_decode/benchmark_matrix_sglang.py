#!/usr/bin/env python3
"""Run a fixed SGLang-reference benchmark matrix for the cluster megakernel."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]

from .cluster_megakernel import cluster_megakernel_forward
from .common import MegakernelConfig, available_backends
from .external_reference import (
    SGLangLayerRunner,
    SGLangSubgraphRunner,
    probe_sglang_import,
    sglang_layer_reference_forward,
    sglang_subgraph_reference_forward,
)
from .megakernel_reference import (
    make_random_megakernel_inputs,
    megakernel_reference_forward,
)


MATRIX_SHAPES = (
    (256, 4),
    (4096, 32),
)
MATRIX_SEQ_LENS = (128, 512, 2048, 4096)
MATRIX_CLUSTER_SIZES = (2, 4)


CSV_FIELDS = (
    "status",
    "error",
    "hidden_dim",
    "num_heads",
    "head_dim",
    "seq_len",
    "cluster_size",
    "num_threads",
    "dtype",
    "seed",
    "output_max_abs_vs_sglang_layer",
    "output_mean_abs_vs_sglang_layer",
    "output_rel_l2_vs_sglang_layer",
    "k_max_abs_vs_sglang_layer",
    "k_mean_abs_vs_sglang_layer",
    "k_rel_l2_vs_sglang_layer",
    "v_max_abs_vs_sglang_layer",
    "v_mean_abs_vs_sglang_layer",
    "v_rel_l2_vs_sglang_layer",
    "local_output_max_abs_vs_sglang_layer",
    "local_output_mean_abs_vs_sglang_layer",
    "local_output_rel_l2_vs_sglang_layer",
    "subgraph_output_max_abs_vs_sglang_layer",
    "subgraph_output_mean_abs_vs_sglang_layer",
    "subgraph_output_rel_l2_vs_sglang_layer",
    "persistent_subgraph_output_max_abs_vs_sglang_layer",
    "persistent_subgraph_output_mean_abs_vs_sglang_layer",
    "persistent_subgraph_output_rel_l2_vs_sglang_layer",
    "persistent_layer_output_max_abs_vs_sglang_layer",
    "persistent_layer_output_mean_abs_vs_sglang_layer",
    "persistent_layer_output_rel_l2_vs_sglang_layer",
    "tc_output_max_abs_vs_sglang_layer",
    "tc_output_mean_abs_vs_sglang_layer",
    "tc_output_rel_l2_vs_sglang_layer",
    "tc_k_max_abs_vs_sglang_layer",
    "tc_k_mean_abs_vs_sglang_layer",
    "tc_k_rel_l2_vs_sglang_layer",
    "tc_v_max_abs_vs_sglang_layer",
    "tc_v_mean_abs_vs_sglang_layer",
    "tc_v_rel_l2_vs_sglang_layer",
    "local_pytorch_ref_ms",
    "sglang_subgraph_ref_ms",
    "sglang_subgraph_persistent_ms",
    "sglang_layer_ref_ms",
    "sglang_layer_persistent_ms",
    "cute_megakernel_ms",
    "tc_megakernel_ms",
)


def _require_runtime() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for this benchmark.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not available_backends()["cute"]:
        raise RuntimeError("CuTe DSL is required for this benchmark.")

    status = probe_sglang_import()
    if not status.available:
        raise RuntimeError(f"SGLang unavailable: {status.error}")


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


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a, b) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _rel_l2(a, b) -> float:
    diff = a.float() - b.float()
    denom = b.float().norm()
    if denom.item() == 0.0:
        return diff.norm().item()
    return (diff.norm() / denom).item()


def _compare(prefix: str, actual, expected) -> dict[str, float]:
    return {
        f"{prefix}_max_abs_vs_sglang_layer": _max_abs(actual, expected),
        f"{prefix}_mean_abs_vs_sglang_layer": _mean_abs(actual, expected),
        f"{prefix}_rel_l2_vs_sglang_layer": _rel_l2(actual, expected),
    }


def _run_case(hidden_dim: int, num_heads: int, seq_len: int, cluster_size: int, args) -> dict[str, object]:
    dtype = getattr(torch, args.dtype)
    config = MegakernelConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=hidden_dim // num_heads,
        cluster_size=cluster_size,
        num_threads=args.num_threads,
    )
    config.validate()

    row: dict[str, object] = {
        "status": "ok",
        "error": "",
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": config.head_dim,
        "seq_len": seq_len,
        "cluster_size": cluster_size,
        "num_threads": args.num_threads,
        "dtype": args.dtype,
        "seed": args.seed,
    }

    inputs = make_random_megakernel_inputs(
        config,
        seq_len=seq_len,
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

    subgraph_runner = None if args.skip_persistent_sglang else SGLangSubgraphRunner(**inputs, config=config)
    layer_runner = None if args.skip_persistent_sglang else SGLangLayerRunner(**inputs, config=config)

    def cute_kernel():
        return cluster_megakernel_forward(**inputs, config=config)

    def tc_kernel():
        return cluster_megakernel_forward(**inputs, config=config, use_tensor_core=True)

    local_out = None if args.skip_local_ref else local_ref()
    subgraph_out = None if args.skip_subgraph_ref else sglang_subgraph_ref()
    layer_out = sglang_layer_ref()
    cute_out = cute_kernel()
    tc_out = None if args.skip_tc else tc_kernel()
    persistent_subgraph_out = None if subgraph_runner is None else subgraph_runner()
    persistent_layer_out = None if layer_runner is None else layer_runner()
    _sync()

    row.update(_compare("output", cute_out[0], layer_out[0]))
    row.update(_compare("k", cute_out[1], layer_out[1]))
    row.update(_compare("v", cute_out[2], layer_out[2]))
    if local_out is not None:
        row.update(_compare("local_output", local_out[0], layer_out[0]))
    if subgraph_out is not None:
        row.update(_compare("subgraph_output", subgraph_out[0], layer_out[0]))
    if persistent_subgraph_out is not None:
        row.update(_compare("persistent_subgraph_output", persistent_subgraph_out[0], layer_out[0]))
    if persistent_layer_out is not None:
        row.update(_compare("persistent_layer_output", persistent_layer_out[0], layer_out[0]))
    if tc_out is not None:
        row.update(_compare("tc_output", tc_out[0], layer_out[0]))
        row.update(_compare("tc_k", tc_out[1], layer_out[1]))
        row.update(_compare("tc_v", tc_out[2], layer_out[2]))

    row["local_pytorch_ref_ms"] = "" if args.skip_local_ref else _time_cuda(local_ref, args.warmup, args.iters)
    row["sglang_subgraph_ref_ms"] = "" if args.skip_subgraph_ref else _time_cuda(
        sglang_subgraph_ref, args.warmup, args.iters
    )
    row["sglang_subgraph_persistent_ms"] = (
        "" if subgraph_runner is None else _time_cuda(subgraph_runner, args.warmup, args.iters)
    )
    row["sglang_layer_ref_ms"] = _time_cuda(sglang_layer_ref, args.warmup, args.iters)
    row["sglang_layer_persistent_ms"] = (
        "" if layer_runner is None else _time_cuda(layer_runner, args.warmup, args.iters)
    )
    row["cute_megakernel_ms"] = _time_cuda(cute_kernel, args.warmup, args.iters)
    row["tc_megakernel_ms"] = "" if args.skip_tc else _time_cuda(tc_kernel, args.warmup, args.iters)
    return row


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_shape_list(value: str) -> tuple[tuple[int, int], ...]:
    shapes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        hidden_dim, num_heads = item.lower().split("x", maxsplit=1)
        shapes.append((int(hidden_dim), int(num_heads)))
    return tuple(shapes)


def _make_output_path(args) -> Path:
    if args.output:
        return Path(args.output)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "result" / f"sglang_matrix_{stamp}.csv"


def run_matrix(args) -> int:
    _require_runtime()

    shapes = _parse_shape_list(args.shapes) if args.shapes else MATRIX_SHAPES
    seq_lens = _parse_int_list(args.seq_lens) if args.seq_lens else MATRIX_SEQ_LENS
    cluster_sizes = _parse_int_list(args.cluster_sizes) if args.cluster_sizes else MATRIX_CLUSTER_SIZES
    output = _make_output_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        for hidden_dim, num_heads in shapes:
            for seq_len in seq_lens:
                for cluster_size in cluster_sizes:
                    label = f"D={hidden_dim} H={num_heads} S={seq_len} C={cluster_size}"
                    print(f"[RUN] {label}", flush=True)
                    try:
                        row = _run_case(hidden_dim, num_heads, seq_len, cluster_size, args)
                        tc_ms = row["tc_megakernel_ms"]
                        tc_text = "tc=skipped" if tc_ms == "" else f"tc={tc_ms:.4f}ms"
                        tc_rel_l2 = row.get("tc_output_rel_l2_vs_sglang_layer", "")
                        tc_rel_l2_text = "tc_rel_l2=skipped" if tc_rel_l2 == "" else f"tc_rel_l2={tc_rel_l2:.6g}"
                        subgraph_ms = row["sglang_subgraph_ref_ms"]
                        persistent_subgraph_ms = row["sglang_subgraph_persistent_ms"]
                        persistent_layer_ms = row["sglang_layer_persistent_ms"]
                        persistent_text = (
                            "persist_subgraph=skipped persist_layer=skipped"
                            if persistent_subgraph_ms == ""
                            else (
                                f"persist_subgraph={persistent_subgraph_ms:.4f}ms "
                                f"persist_layer={persistent_layer_ms:.4f}ms"
                            )
                        )
                        subgraph_text = (
                            "subgraph=skipped speedup=skipped"
                            if subgraph_ms == ""
                            else (
                                f"subgraph={subgraph_ms:.4f}ms "
                                f"speedup={subgraph_ms / row['cute_megakernel_ms']:.3f}x"
                            )
                        )
                        print(
                            "[OK] "
                            f"{label} cute={row['cute_megakernel_ms']:.4f}ms "
                            f"{tc_text} "
                            f"{subgraph_text} "
                            f"{persistent_text} "
                            f"sglang_layer={row['sglang_layer_ref_ms']:.4f}ms "
                            f"out_max_abs={row['output_max_abs_vs_sglang_layer']:.6g} "
                            f"out_rel_l2={row['output_rel_l2_vs_sglang_layer']:.6g} "
                            f"{tc_rel_l2_text}",
                            flush=True,
                        )
                    except Exception as exc:
                        row = {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "hidden_dim": hidden_dim,
                            "num_heads": num_heads,
                            "head_dim": hidden_dim // num_heads,
                            "seq_len": seq_len,
                            "cluster_size": cluster_size,
                            "num_threads": args.num_threads,
                            "dtype": args.dtype,
                            "seed": args.seed,
                        }
                        print(f"[FAIL] {label} {row['error']}", flush=True)
                        if args.fail_fast:
                            writer.writerow(row)
                            f.flush()
                            raise

                    writer.writerow(row)
                    f.flush()

    print(f"\nWrote {output}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGLang-reference benchmark matrix for cluster megakernel.")
    parser.add_argument("--shapes", default="", help="Comma list of hidden_dim x num_heads, e.g. 256x4,4096x32.")
    parser.add_argument("--seq-lens", default="", help="Comma list of sequence lengths, e.g. 128,512,2048.")
    parser.add_argument("--cluster-sizes", default="", help="Comma list of cluster sizes, e.g. 2,4.")
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default="", help="CSV output path.")
    parser.add_argument("--skip-local-ref", action="store_true", help="Skip local PyTorch reference timing.")
    parser.add_argument("--skip-subgraph-ref", action="store_true", help="Skip lightweight SGLang subgraph timing.")
    parser.add_argument("--skip-persistent-sglang", action="store_true", help="Skip persistent SGLang runner timing.")
    parser.add_argument("--skip-tc", action="store_true", help="Skip experimental tensor-core path timing.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failing case.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return run_matrix(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
