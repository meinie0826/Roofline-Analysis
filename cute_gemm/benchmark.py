import argparse
import time
from typing import Iterable, Tuple

import torch

from mma_gemm_cutedsl import run_dense_gemm, validate_mnk
from ref import check_close, make_inputs, torch_gemm


SMALL_SHAPES = [
    (128, 256, 64),
    (256, 256, 64),
    (256, 512, 128),
]

LARGE_SHAPES = [
    (1024, 1024, 256),
    (2048, 2048, 256),
    (4096, 2048, 512),
]


def _parse_mnk(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return parts[0], parts[1], parts[2]


def _iter_shapes(shape_set: str, shapes: list[Tuple[int, int, int]] | None):
    if shapes:
        return shapes
    if shape_set == "small":
        return SMALL_SHAPES
    if shape_set == "large":
        return LARGE_SHAPES
    return SMALL_SHAPES + LARGE_SHAPES


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _tflops(mnk: Tuple[int, int, int], ms: float) -> float:
    m, n, k = mnk
    flops = 2.0 * m * n * k
    return flops / (ms / 1e3) / 1e12


def benchmark_shape(
    mnk: Tuple[int, int, int],
    atol: float,
    warmup: int,
    iters: int,
) -> dict:
    validate_mnk(mnk)
    a, b = make_inputs(mnk)

    got = run_dense_gemm(a, b)
    ref = torch_gemm(a, b)
    check_close(got, ref, atol=atol)

    cute_ms = _time_cuda(lambda: run_dense_gemm(a, b), warmup, iters)
    torch_ms = _time_cuda(lambda: torch_gemm(a, b), warmup, iters)

    return {
        "mnk": mnk,
        "cute_ms": cute_ms,
        "torch_ms": torch_ms,
        "cute_tflops": _tflops(mnk, cute_ms),
        "torch_tflops": _tflops(mnk, torch_ms),
        "speedup_vs_torch": torch_ms / cute_ms,
    }


def print_results(rows: Iterable[dict]) -> None:
    print("m,n,k,cute_ms,torch_ms,cute_tflops,torch_tflops,speedup_vs_torch")
    for row in rows:
        m, n, k = row["mnk"]
        print(
            f"{m},{n},{k},"
            f"{row['cute_ms']:.6f},"
            f"{row['torch_ms']:.6f},"
            f"{row['cute_tflops']:.6f},"
            f"{row['torch_tflops']:.6f},"
            f"{row['speedup_vs_torch']:.6f}"
        )


def main():
    from cuda.bindings import driver as cu_driver

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=["small", "large", "all"],
        default="all",
    )
    parser.add_argument(
        "--shapes",
        type=_parse_mnk,
        nargs="*",
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    cu_driver.cuInit(0)

    rows = []
    for mnk in _iter_shapes(args.shape_set, args.shapes):
        row = benchmark_shape(mnk, args.atol, args.warmup, args.iters)
        rows.append(row)
        print(
            "RESULT",
            {
                "mnk": mnk,
                "cute_ms": round(row["cute_ms"], 6),
                "torch_ms": round(row["torch_ms"], 6),
                "speedup_vs_torch": round(row["speedup_vs_torch"], 6),
            },
        )
    print_results(rows)


if __name__ == "__main__":
    main()
