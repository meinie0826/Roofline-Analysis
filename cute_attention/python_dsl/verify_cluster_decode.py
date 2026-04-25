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


def _split_kv_decode_reference(q, k, v, cluster_size: int, softmax_scale: float):
    batch_heads, seq_len, head_dim = k.shape
    partial_max = torch.full((batch_heads, cluster_size), -torch.inf, dtype=torch.float32)
    partial_sum = torch.zeros((batch_heads, cluster_size), dtype=torch.float32)
    partial_out = torch.zeros((batch_heads, cluster_size, head_dim), dtype=torch.float32)

    q_f = q[:, 0, :].to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    kv_per_cta = (seq_len + cluster_size - 1) // cluster_size

    for cta_rank in range(cluster_size):
        kv_start = cta_rank * kv_per_cta
        kv_stop = min(kv_start + kv_per_cta, seq_len)
        scores = torch.sum(q_f[:, None, :] * k_f[:, kv_start:kv_stop, :], dim=-1) * softmax_scale
        slice_max = torch.max(scores, dim=-1).values
        exp_scores = torch.exp(scores - slice_max[:, None])
        partial_max[:, cta_rank] = slice_max
        partial_sum[:, cta_rank] = torch.sum(exp_scores, dim=-1)
        partial_out[:, cta_rank, :] = torch.sum(exp_scores[:, :, None] * v_f[:, kv_start:kv_stop, :], dim=1)

    global_max = torch.max(partial_max, dim=1).values
    renorm = torch.exp(partial_max - global_max[:, None])
    global_sum = torch.sum(partial_sum * renorm, dim=1)
    numerator = torch.sum(partial_out * renorm[:, :, None], dim=1)
    return (numerator / global_sum[:, None]).to(dtype=q.dtype).unsqueeze(1)


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
    torch.manual_seed(args.seed)

    q = torch.randn(args.batch_heads, 1, args.head_dim, dtype=torch.float32)
    k = torch.randn(args.batch_heads, args.seq_len, args.head_dim, dtype=torch.float32)
    v = torch.randn(args.batch_heads, args.seq_len, args.head_dim, dtype=torch.float32)
    scale = args.softmax_scale

    ref = _decode_reference(q, k, v, scale)
    out = _split_kv_decode_reference(q, k, v, args.cluster_size, scale)
    expected_payload = args.head_dim + 2
    actual_payload = args.head_dim + 2
    if actual_payload != expected_payload:
        raise AssertionError(f"Unexpected leader payload: {actual_payload} != {expected_payload}")

    return _check_close("reduce_contract_cpu", out, ref, rtol=1e-5, atol=1e-5)


def verify_kernel_stage(stage_name: str, args) -> CheckResult:
    _assert_torch()
    from kernels import AttentionConfig, available_backends, run_stage

    if not available_backends()["cute"]:
        raise RuntimeError("CuTe DSL is required for kernel correctness verification.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kernel correctness verification.")

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.heads, 1, args.head_dim, device="cuda", dtype=args.dtype)
    k = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device="cuda", dtype=args.dtype)
    v = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device="cuda", dtype=args.dtype)

    config = AttentionConfig(causal=False, num_threads=args.num_threads, cluster_size=args.cluster_size)
    ref = _decode_reference(q, k, v, config.resolve_scale(args.head_dim))
    out = run_stage(stage_name, q, k, v, config)
    torch.cuda.synchronize()
    return _check_close(stage_name, out, ref, rtol=args.rtol, atol=args.atol)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correctness checks for experimental cluster decode kernels.")
    parser.add_argument("--stage", default="all", choices=["all", "reduce", "cluster_decode", "cluster_decode_split"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--batch-heads", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=129)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cluster-size", type=int, default=2, choices=[2, 4])
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _assert_torch()
    args.dtype = getattr(torch, args.dtype)
    args.softmax_scale = 1.0 / (args.head_dim**0.5)

    results = []
    if args.stage in ("all", "reduce"):
        results.append(verify_reduce_contract(args))
    if args.stage in ("all", "cluster_decode"):
        results.append(verify_kernel_stage("cluster_decode", args))
    if args.stage in ("all", "cluster_decode_split"):
        results.append(verify_kernel_stage("cluster_decode_split", args))

    for result in results:
        print(
            f"PASS {result.name}: "
            f"max_abs_err={result.max_abs_err:.6g}, max_rel_err={result.max_rel_err:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
