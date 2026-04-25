#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CheckResult:
    name: str
    max_abs_err: float
    max_rel_err: float


def _assert_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for cluster decode correctness verification.")


def _decode_reference(q, k, v, scale: float):
    scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.to(torch.float32)).to(dtype=q.dtype)


def _error(out, ref) -> tuple[float, float]:
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    denom = ref.to(torch.float32).abs().clamp_min(1e-6)
    return diff.max().item(), (diff / denom).max().item()


def _check_close(name: str, out, ref, rtol: float, atol: float) -> CheckResult:
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    max_abs, max_rel = _error(out, ref)
    return CheckResult(name=name, max_abs_err=max_abs, max_rel_err=max_rel)


def verify_reduce_contract(args) -> CheckResult:
    _assert_torch()
    try:
        from .cluster_decode_reduce import leader_reduce_payload_floats, split_kv_decode_reference
    except ImportError:
        from cluster_decode_reduce import leader_reduce_payload_floats, split_kv_decode_reference

    torch.manual_seed(args.seed)

    q = torch.randn(args.batch_heads, 1, args.head_dim, dtype=torch.float32)
    k = torch.randn(args.batch_heads, args.seq_len, args.head_dim, dtype=torch.float32)
    v = torch.randn(args.batch_heads, args.seq_len, args.head_dim, dtype=torch.float32)
    scale = args.softmax_scale

    ref = _decode_reference(q, k, v, scale)
    out = split_kv_decode_reference(q, k, v, args.cluster_size, scale)
    expected_payload = args.head_dim + 2
    actual_payload = leader_reduce_payload_floats(args.head_dim)
    if actual_payload != expected_payload:
        raise AssertionError(f"Unexpected leader payload: {actual_payload} != {expected_payload}")

    return _check_close("reduce_contract_cpu", out, ref, rtol=1e-5, atol=1e-5)


def verify_kernel_stage(stage_name: str, args) -> CheckResult:
    _assert_torch()
    try:
        from .cluster_decode import cluster_decode_forward
        from .cluster_decode_split import cluster_decode_split_forward
        from .common import ClusterDecodeConfig, available_backends
    except ImportError:
        from cluster_decode import cluster_decode_forward
        from cluster_decode_split import cluster_decode_split_forward
        from common import ClusterDecodeConfig, available_backends

    if not available_backends()["cute"]:
        raise RuntimeError("CuTe DSL is required for kernel correctness verification.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel correctness verification.")

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.heads, 1, args.head_dim, device="cuda", dtype=args.dtype)
    k = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device="cuda", dtype=args.dtype)
    v = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device="cuda", dtype=args.dtype)

    config = ClusterDecodeConfig(num_threads=args.num_threads, cluster_size=args.cluster_size)
    ref = _decode_reference(q, k, v, config.resolve_scale(args.head_dim))
    if stage_name == "cluster_decode":
        out = cluster_decode_forward(q, k, v, config)
    elif stage_name == "cluster_decode_split":
        out = cluster_decode_split_forward(q, k, v, config)
    else:
        raise ValueError(f"Unknown cluster decode stage: {stage_name}")
    torch.cuda.synchronize()
    return _check_close(stage_name, out, ref, rtol=args.rtol, atol=args.atol)


def verify_megakernel(args) -> list[CheckResult]:
    """Verify the full decode megakernel against the PyTorch reference."""
    _assert_torch()
    try:
        from .cluster_megakernel import cluster_megakernel_forward
        from .common import MegakernelConfig, available_backends
        from .megakernel_reference import make_random_megakernel_inputs, megakernel_reference_forward
    except ImportError:
        from cluster_megakernel import cluster_megakernel_forward
        from common import MegakernelConfig, available_backends
        from megakernel_reference import make_random_megakernel_inputs, megakernel_reference_forward

    if not available_backends()["cute"]:
        raise RuntimeError("CuTe DSL is required for megakernel correctness verification.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for megakernel correctness verification.")

    config = MegakernelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.hidden_dim // args.num_heads,
        cluster_size=args.cluster_size,
        num_threads=args.num_threads,
    )
    config.validate()

    inputs = make_random_megakernel_inputs(
        config, seq_len=args.seq_len, device="cuda", dtype=args.dtype, seed=args.seed
    )

    ref_out, ref_k, ref_v = megakernel_reference_forward(**inputs, config=config)
    cuda_out, cuda_k, cuda_v = cluster_megakernel_forward(**inputs, config=config)
    torch.cuda.synchronize()

    results = [
        _check_close("megakernel_output", cuda_out, ref_out, rtol=args.rtol, atol=args.atol),
        _check_close("megakernel_k_new",  cuda_k,   ref_k,   rtol=args.rtol, atol=args.atol),
        _check_close("megakernel_v_new",  cuda_v,   ref_v,   rtol=args.rtol, atol=args.atol),
    ]
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correctness checks for experimental cluster decode kernels.")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "reduce", "cluster_decode", "cluster_decode_split", "megakernel"],
    )
    # Attention-only stages
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--batch-heads", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cluster-size", type=int, default=2, choices=[2, 4])
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    # Megakernel-specific
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads",  type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _assert_torch()
    args.dtype = getattr(torch, args.dtype)
    args.softmax_scale = 1.0 / (args.head_dim**0.5)

    results: list[CheckResult] = []
    if args.stage in ("all", "reduce"):
        results.append(verify_reduce_contract(args))
    if args.stage in ("all", "cluster_decode"):
        results.append(verify_kernel_stage("cluster_decode", args))
    if args.stage in ("all", "cluster_decode_split"):
        results.append(verify_kernel_stage("cluster_decode_split", args))
    if args.stage in ("all", "megakernel"):
        results.extend(verify_megakernel(args))

    for result in results:
        print(
            f"PASS {result.name}: "
            f"max_abs_err={result.max_abs_err:.6g}, max_rel_err={result.max_rel_err:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
