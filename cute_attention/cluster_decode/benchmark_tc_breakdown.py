#!/usr/bin/env python3
"""Break down the experimental tensor-core decode path.

This is a diagnostic benchmark for deciding what to port into CuTeDSL first.
It times the same semantic pieces used by ``cluster_megakernel_tc_forward``:

  RMSNorm, QKV matmul, RoPE/KV exposure, dense attention, WO matmul, full path.

It also times persistent SGLang layer/subgraph runners on the same inputs so the
numbers are compared against a reusable SGLang baseline, not the stateless
reference construction path.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None  # type: ignore[assignment]

from .cluster_megakernel_tc import _apply_rope_gptj_batched, cluster_megakernel_tc_forward
from .common import MegakernelConfig
from .external_reference import SGLangLayerRunner, SGLangSubgraphRunner, probe_sglang_import
from .megakernel_reference import make_random_megakernel_inputs, rms_norm


def _require_runtime() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for this benchmark.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
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


class TensorCoreBreakdown:
    def __init__(self, inputs: dict, config: MegakernelConfig):
        self.inputs = inputs
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.resolve_scale()
        self.dtype = inputs["hidden_states"].dtype

        self.h_norm = None
        self.q = None
        self.q_rot = None
        self.k_new = None
        self.v_new = None
        self.attn_out = None

    def rms(self):
        self.h_norm = rms_norm(
            self.inputs["hidden_states"].float(),
            self.inputs["rms_weight"],
            eps=1e-6,
        ).to(self.dtype)
        return self.h_norm

    def qkv(self):
        h_norm = self.h_norm if self.h_norm is not None else self.rms()
        qkv = h_norm @ self.inputs["w_qkv"].t()
        qkv = qkv.reshape(1, 3, self.num_heads, self.head_dim)
        self.q = qkv[:, 0]
        self.k = qkv[:, 1]
        self.v = qkv[:, 2]
        return qkv

    def rope_kv(self):
        if self.q is None:
            self.qkv()
        cos = self.inputs["cos_rope"].to(self.dtype)
        sin = self.inputs["sin_rope"].to(self.dtype)
        self.q_rot, k_rot = _apply_rope_gptj_batched(self.q, self.k, cos, sin)
        self.k_new = k_rot.to(self.dtype)
        self.v_new = self.v.to(self.dtype)
        return self.q_rot, self.k_new, self.v_new

    def attention(self):
        if self.q_rot is None:
            self.rope_kv()
        k_f = torch.cat(
            [self.inputs["k_cache"].to(torch.float32), self.k_new.to(torch.float32)],
            dim=0,
        )
        v_f = torch.cat(
            [self.inputs["v_cache"].to(torch.float32), self.v_new.to(torch.float32)],
            dim=0,
        )
        q_bh = self.q_rot[0].to(torch.float32).unsqueeze(1)
        k_bh = k_f.permute(1, 2, 0)
        v_bh = v_f.permute(1, 0, 2)
        scores = torch.bmm(q_bh, k_bh) * self.scale
        probs = torch.softmax(scores, dim=-1)
        self.attn_out = torch.bmm(probs, v_bh).squeeze(1).to(self.dtype)
        return self.attn_out

    def wo(self):
        attn_out = self.attn_out if self.attn_out is not None else self.attention()
        return attn_out.reshape(1, self.hidden_dim) @ self.inputs["w_o"].t()

    def full(self):
        return cluster_megakernel_tc_forward(**self.inputs, config=self.config)


def run(args) -> int:
    _require_runtime()
    dtype = getattr(torch, args.dtype)
    config = MegakernelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.hidden_dim // args.num_heads,
        cluster_size=args.cluster_size,
        num_threads=args.num_threads,
    )
    inputs = make_random_megakernel_inputs(
        config,
        seq_len=args.seq_len,
        device="cuda",
        dtype=dtype,
        seed=args.seed,
    )

    breakdown = TensorCoreBreakdown(inputs, config)
    # Build dependencies once so isolated stage timings do not include previous stages.
    breakdown.rms()
    breakdown.qkv()
    breakdown.rope_kv()
    breakdown.attention()
    _sync()

    sglang_subgraph = SGLangSubgraphRunner(**inputs, config=config)
    sglang_layer = SGLangLayerRunner(**inputs, config=config)

    timings = [
        ("rms", _time_cuda(breakdown.rms, args.warmup, args.iters)),
        ("qkv_tc", _time_cuda(breakdown.qkv, args.warmup, args.iters)),
        ("rope_kv", _time_cuda(breakdown.rope_kv, args.warmup, args.iters)),
        ("attention", _time_cuda(breakdown.attention, args.warmup, args.iters)),
        ("wo_tc", _time_cuda(breakdown.wo, args.warmup, args.iters)),
        ("tc_full", _time_cuda(breakdown.full, args.warmup, args.iters)),
        ("persist_subgraph", _time_cuda(sglang_subgraph, args.warmup, args.iters)),
        ("persist_layer", _time_cuda(sglang_layer, args.warmup, args.iters)),
    ]

    print("=== Tensor-core path breakdown ===")
    print(
        f"D={args.hidden_dim} H={args.num_heads} S={args.seq_len} "
        f"C={args.cluster_size} dtype={args.dtype}"
    )
    for name, ms in timings:
        print(f"{name:16s} {ms:.4f} ms")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down tensor-core decode target-line latency.")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--cluster-size", type=int, default=4, choices=[2, 4])
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
