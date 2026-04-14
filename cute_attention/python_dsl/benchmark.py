#!/usr/bin/env python3

import argparse
import time

from kernels import AttentionConfig, available_backends, run_stage


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=128)
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
    time_ms = benchmark(args.stage, q, k, v, config)
    print({"stage": args.stage, "time_ms": time_ms, "shape": tuple(q.shape), "dtype": args.dtype})


if __name__ == "__main__":
    main()
