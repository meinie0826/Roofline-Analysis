#!/usr/bin/env python3

import argparse
import time

from kernels import (
    AttentionConfig,
    autotune_stage12_config,
    autotune_stage13_config,
    autotune_stage15_config,
    available_backends,
    run_stage,
)


torch = None
if available_backends()["torch"]:
    import torch


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
    if getattr(config, "producer_warps", 0):
        parts.append(f"producer_warps={config.producer_warps}")
    if config.num_stages_kv:
        parts.append(f"stages={config.num_stages_kv}")
    return ",".join(parts) if parts else None


def benchmark_stage_with_fallback(stage_name, q, k, v, config, warmup=5, repeat=20):
    if stage_name == "stage12":
        tuned = autotune_stage12_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage13":
        tuned = autotune_stage13_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage15":
        tuned = autotune_stage15_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)

    if stage_name not in {"stage1", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11", "stage12", "stage13", "stage14", "stage15"}:
        return benchmark(stage_name, q, k, v, config, warmup=warmup, repeat=repeat), None

    block_m = config.block_m
    while block_m >= 1:
        cfg = AttentionConfig(
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            block_m=block_m,
            block_n=config.block_n,
            num_threads=config.num_threads,
            producer_warps=config.producer_warps,
        )
        try:
            t = benchmark(stage_name, q, k, v, cfg, warmup=warmup, repeat=repeat)
            return t, _config_status_suffix(cfg) if block_m != config.block_m else None
        except ValueError as exc:
            if "shared memory footprint too large" not in str(exc):
                raise
            block_m //= 2
    raise ValueError("stage1 failed all fallback block_m candidates for shared memory.")


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
            "baseline_fa4",
            "baseline_sdpa",
        ]
    return [name.strip() for name in stages_arg.split(",") if name.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stage names, or 'all'.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    q = torch.randn(args.batch, args.heads, args.seqlen, args.headdim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = AttentionConfig(block_m=args.block_m, block_n=args.block_n, num_threads=args.num_threads)
    stages = parse_stage_list(args.stages)

    print(
        {
            "shape": tuple(q.shape),
            "dtype": args.dtype,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "num_threads": args.num_threads,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "stages": stages,
        }
    )
    print("stage,time_ms,status")
    for stage_name in stages:
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
