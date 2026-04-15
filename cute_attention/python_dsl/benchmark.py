#!/usr/bin/env python3
"""
Benchmark runner for cute_attention stages.

Default shape: (batch=1, heads=16, seqlen=2048, headdim=128, dtype=float16)
These match a typical LLM decode-phase / prefill configuration and stay within
the seq-len limits of all CuTe stages (max 4096).

Each stage is benchmarked with its most appropriate default tuning config:
  - stage12/stage13 → autotune enabled
  - stage16         → autotune enabled (block_m/block_n)
  - stage15/stage16 → num_threads always forced to 256 by the kernel
  - stage14         → num_threads=256 (128 consumer + 128 producer)
  - stage0-stage11  → num_threads=128 (can be changed with --num-threads)
"""

import argparse
import time

from kernels import (
    AttentionConfig,
    autotune_stage12_config,
    autotune_stage13_config,
    autotune_stage16_config,
    available_backends,
    run_stage,
)


torch = None
if available_backends()["torch"]:
    import torch


# ---------------------------------------------------------------------------
# Stages that benefit from a higher default num_threads (producer/consumer)
# ---------------------------------------------------------------------------
_WARPSPEC_STAGES = {"stage14", "stage15", "stage16"}

# Stages with built-in autotune
_AUTOTUNE_STAGES = {"stage12", "stage13", "stage16"}


def benchmark(stage_name, q, k, v, config, warmup=5, repeat=20):
    for _ in range(warmup):
        run_stage(stage_name, q, k, v, config)

    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_stage(stage_name, q, k, v, config)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


def _config_status_suffix(config: AttentionConfig) -> str | None:
    parts = []
    if config.block_m:
        parts.append(f"block_m={config.block_m}")
    if config.block_n:
        parts.append(f"block_n={config.block_n}")
    if config.num_stages_kv:
        parts.append(f"stages={config.num_stages_kv}")
    return ",".join(parts) if parts else None


def _make_config_for_stage(stage_name: str, base: AttentionConfig) -> AttentionConfig:
    """Return a config tailored to the given stage from the base config."""
    from dataclasses import replace
    if stage_name in _WARPSPEC_STAGES:
        return replace(base, num_threads=256)
    return base


def benchmark_stage_with_fallback(stage_name, q, k, v, config, warmup=5, repeat=20):
    """Benchmark a stage, applying autotune or block_m fallback as appropriate."""
    if stage_name == "stage12":
        tuned = autotune_stage12_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage13":
        tuned = autotune_stage13_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage16":
        tuned = autotune_stage16_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)

    if stage_name not in {
        "stage1", "stage4", "stage5", "stage6", "stage7", "stage8",
        "stage9", "stage10", "stage11", "stage14", "stage15",
    }:
        return benchmark(stage_name, q, k, v, config, warmup=warmup, repeat=repeat), None

    block_m = config.block_m
    while block_m >= 1:
        from dataclasses import replace
        cfg = replace(config, block_m=block_m)
        try:
            t = benchmark(stage_name, q, k, v, cfg, warmup=warmup, repeat=repeat)
            return t, _config_status_suffix(cfg) if block_m != config.block_m else None
        except ValueError as exc:
            if "shared memory footprint too large" not in str(exc):
                raise
            block_m //= 2
    raise ValueError(f"{stage_name} failed all fallback block_m candidates for shared memory.")


def parse_stage_list(stages_arg: str) -> list[str]:
    if stages_arg == "all":
        return [
            "stage0",
            "stage1",
            "stage3",
            "stage4",
            "stage5",
            "stage6",
            "stage7",
            "stage8",
            "stage9",
            "stage10",
            "stage11",
            "stage12",
            "stage13",
            "stage14",
            "stage15",
            "stage16",
            "baseline_fa4",
            "baseline_sdpa",
        ]
    return [name.strip() for name in stages_arg.split(",") if name.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stage names, or 'all' (default).",
    )
    parser.add_argument("--batch",       type=int,   default=1,
                        help="Batch size (default: 1).")
    parser.add_argument("--heads",       type=int,   default=16,
                        help="Number of attention heads (default: 16).")
    parser.add_argument("--seqlen",      type=int,   default=2048,
                        help="Sequence length (default: 2048; max 4096 for CuTe stages).")
    parser.add_argument("--headdim",     type=int,   default=128,
                        help="Head dimension (default: 128).")
    parser.add_argument("--dtype",       default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Data type (default: float16; CuTe stages require float16).")
    parser.add_argument("--block-m",     type=int,   default=64,
                        help="Q block size (default: 64). Overridden by autotune for stage12/13/16.")
    parser.add_argument("--block-n",     type=int,   default=128,
                        help="KV block size (default: 128). Overridden by autotune for stage12/13/16.")
    parser.add_argument("--num-threads", type=int,   default=128,
                        help="Thread count (default: 128). Forced to 256 for stage14/15/16.")
    parser.add_argument("--causal",      action="store_true", default=True,
                        help="Use causal masking (default: True).")
    parser.add_argument("--warmup",      type=int,   default=5,
                        help="Warmup iterations (default: 5).")
    parser.add_argument("--repeat",      type=int,   default=20,
                        help="Benchmark iterations (default: 20).")
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = {
        "float16":  torch.float16,
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    q = torch.randn(args.batch, args.heads, args.seqlen, args.headdim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    base_config = AttentionConfig(
        causal=args.causal,
        block_m=args.block_m,
        block_n=args.block_n,
        num_threads=args.num_threads,
    )
    stages = parse_stage_list(args.stages)

    device_name = torch.cuda.get_device_name(0)
    print({
        "device":      device_name,
        "shape":       tuple(q.shape),
        "dtype":       args.dtype,
        "causal":      args.causal,
        "block_m":     args.block_m,
        "block_n":     args.block_n,
        "num_threads": args.num_threads,
        "warmup":      args.warmup,
        "repeat":      args.repeat,
        "stages":      stages,
    })
    print("stage,time_ms,status")
    for stage_name in stages:
        config = _make_config_for_stage(stage_name, base_config)
        try:
            time_ms, status_suffix = benchmark_stage_with_fallback(
                stage_name, q, k, v, config, warmup=args.warmup, repeat=args.repeat
            )
            if status_suffix:
                print(f"{stage_name},{time_ms:.3f},ok:{status_suffix}")
            else:
                print(f"{stage_name},{time_ms:.3f},ok")
        except Exception as exc:
            print(f"{stage_name},nan,failed:{type(exc).__name__}:{exc}")


if __name__ == "__main__":
    main()
