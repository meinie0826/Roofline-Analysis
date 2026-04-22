import argparse
import time
from typing import Iterable, Tuple

import torch

import mma_gemm_cutedsl as gemm_1cta
import mma_gemm_2cta_cutedsl as gemm_2cta
from ref import check_close, make_inputs, torch_gemm_with_dtype

VARIANTS: dict[str, dict] = {
    "1cta": {
        "module": gemm_1cta,
        "small_shapes": [
            (128, 256, 64),
            (256, 256, 64),
            (256, 512, 128),
        ],
        "large_shapes": [
            (1024, 1024, 256),
            (2048, 2048, 256),
            (4096, 2048, 512),
        ],
        "torch_out_dtype": torch.float32,
        "default_atol": 1e-3,
    },
    "2cta": {
        "module": gemm_2cta,
        "small_shapes": [
            (256, 256, 64),
            (512, 256, 64),
            (512, 512, 128),
        ],
        "large_shapes": [
            (1024, 1024, 256),
            (2048, 2048, 256),
            (4096, 2048, 512),
        ],
        "torch_out_dtype": torch.float16,
        "default_atol": 1e-1,
    },
}


def _parse_mnk(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return parts[0], parts[1], parts[2]


def _iter_shapes(variant: str, shape_set: str, shapes: list[Tuple[int, int, int]] | None):
    if shapes:
        return shapes
    cfg = VARIANTS[variant]
    if shape_set == "small":
        return cfg["small_shapes"]
    if shape_set == "large":
        return cfg["large_shapes"]
    return cfg["small_shapes"] + cfg["large_shapes"]


def _iter_variants(variant: str):
    if variant == "all":
        return list(VARIANTS.keys())
    return [variant]


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


def _time_cuda_events(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / iters


def _tflops(mnk: Tuple[int, int, int], ms: float) -> float:
    m, n, k = mnk
    flops = 2.0 * m * n * k
    return flops / (ms / 1e3) / 1e12


def benchmark_shape(
    variant: str,
    mnk: Tuple[int, int, int],
    atol: float | None,
    warmup: int,
    iters: int,
) -> dict:
    cfg = VARIANTS[variant]
    module = cfg["module"]
    atol = cfg["default_atol"] if atol is None else atol

    module.validate_mnk(mnk)
    a, b = make_inputs(mnk)
    c, a_tensor, b_tensor, c_tensor = module.prepare_cute_gemm(a, b)

    got = module.run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor)
    ref = torch_gemm_with_dtype(a, b, cfg["torch_out_dtype"])
    check_close(got, ref, atol=atol, rtol=1e-5)

    cute_wall_ms = _time_cuda(
        lambda: module.run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor),
        warmup,
        iters,
    )
    cute_cuda_ms = _time_cuda_events(
        lambda: module.run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor),
        warmup,
        iters,
    )
    torch_wall_ms = _time_cuda(
        lambda: torch_gemm_with_dtype(a, b, cfg["torch_out_dtype"]),
        warmup,
        iters,
    )
    torch_cuda_ms = _time_cuda_events(
        lambda: torch_gemm_with_dtype(a, b, cfg["torch_out_dtype"]),
        warmup,
        iters,
    )

    return {
        "variant": variant,
        "mnk": mnk,
        "cute_wall_ms": cute_wall_ms,
        "cute_cuda_ms": cute_cuda_ms,
        "torch_wall_ms": torch_wall_ms,
        "torch_cuda_ms": torch_cuda_ms,
        "cute_wall_tflops": _tflops(mnk, cute_wall_ms),
        "cute_cuda_tflops": _tflops(mnk, cute_cuda_ms),
        "torch_wall_tflops": _tflops(mnk, torch_wall_ms),
        "torch_cuda_tflops": _tflops(mnk, torch_cuda_ms),
        "speedup_vs_torch_wall": torch_wall_ms / cute_wall_ms,
        "speedup_vs_torch_cuda": torch_cuda_ms / cute_cuda_ms,
    }


def print_results(rows: Iterable[dict]) -> None:
    print(
        "variant,m,n,k,"
        "cute_wall_ms,cute_cuda_ms,torch_wall_ms,torch_cuda_ms,"
        "cute_wall_tflops,cute_cuda_tflops,torch_wall_tflops,torch_cuda_tflops,"
        "speedup_vs_torch_wall,speedup_vs_torch_cuda"
    )
    for row in rows:
        m, n, k = row["mnk"]
        print(
            f"{row['variant']},{m},{n},{k},"
            f"{row['cute_wall_ms']:.6f},"
            f"{row['cute_cuda_ms']:.6f},"
            f"{row['torch_wall_ms']:.6f},"
            f"{row['torch_cuda_ms']:.6f},"
            f"{row['cute_wall_tflops']:.6f},"
            f"{row['cute_cuda_tflops']:.6f},"
            f"{row['torch_wall_tflops']:.6f},"
            f"{row['torch_cuda_tflops']:.6f},"
            f"{row['speedup_vs_torch_wall']:.6f},"
            f"{row['speedup_vs_torch_cuda']:.6f}"
        )


def main():
    from cuda.bindings import driver as cu_driver

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["1cta", "2cta", "all"],
        default="all",
    )
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
    parser.add_argument("--atol", type=float, default=None)
    args = parser.parse_args()

    cu_driver.cuInit(0)

    rows = []
    for variant in _iter_variants(args.variant):
        for mnk in _iter_shapes(variant, args.shape_set, args.shapes):
            row = benchmark_shape(variant, mnk, args.atol, args.warmup, args.iters)
            rows.append(row)
            print(
                "RESULT",
                {
                    "variant": variant,
                    "mnk": mnk,
                    "cute_wall_ms": round(row["cute_wall_ms"], 6),
                    "cute_cuda_ms": round(row["cute_cuda_ms"], 6),
                    "torch_wall_ms": round(row["torch_wall_ms"], 6),
                    "torch_cuda_ms": round(row["torch_cuda_ms"], 6),
                    "speedup_vs_torch_wall": round(
                        row["speedup_vs_torch_wall"], 6
                    ),
                    "speedup_vs_torch_cuda": round(
                        row["speedup_vs_torch_cuda"], 6
                    ),
                },
            )
    print_results(rows)


if __name__ == "__main__":
    main()
