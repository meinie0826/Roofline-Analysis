import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import cutlass.cute as cute
import cutlass.cute.testing as cute_testing
import torch

import mma_gemm_1cta_cutedsl as gemm_1cta
import mma_gemm_2cta_commit_cutedsl as gemm_2cta_commit
import mma_gemm_2cta_manual_phase_mask_cutedsl as gemm_2cta_manual
import mma_gemm_2cta_pipeline_cutedsl as gemm_2cta_pipeline
import mma_gemm_2cta_tma_2stage_cutedsl as gemm_2cta_tma_2stage
import mma_gemm_2cta_tma_3stage_cutedsl as gemm_2cta_tma_3stage
import mma_gemm_2cta_tma_6stage_cutedsl as gemm_2cta_tma_6stage
import mma_gemm_2cta_tma_nopipeline_cutedsl as gemm_2cta_tma_nopipeline
import mma_gemm_2cta_tma_pipeline_cutedsl as gemm_2cta_tma_pipeline
import mma_gemm_2cta_tma_pipeline_tma_store_cutedsl as gemm_2cta_tma_pipeline_tma_store
from configs import CANDIDATE_GROUPS
from ref import (
    check_close,
    make_torch_cublas_runner,
    make_inputs,
    torch_perf_gemm_with_dtype,
    torch_reference_gemm_with_dtype,
)


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
    "2cta_pipeline": {
        "module": gemm_2cta_pipeline,
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
    "2cta_commit": {
        "module": gemm_2cta_commit,
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
    "2cta_manual": {
        "module": gemm_2cta_manual,
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
    "2cta_tma_pipeline": {
        "module": gemm_2cta_tma_pipeline,
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
    "2cta_tma_pipeline_tma_store": {
        "module": gemm_2cta_tma_pipeline_tma_store,
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
    "2cta_tma_2stage": {
        "module": gemm_2cta_tma_2stage,
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
    "2cta_tma_3stage": {
        "module": gemm_2cta_tma_3stage,
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
    "2cta_tma_6stage": {
        "module": gemm_2cta_tma_6stage,
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
    "2cta_tma_nopipeline": {
        "module": gemm_2cta_tma_nopipeline,
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



def _flops(mnk: Tuple[int, int, int]) -> float:
    m, n, k = mnk
    return 2.0 * m * n * k


def _tflops(mnk: Tuple[int, int, int], ms: float | None) -> float | None:
    if ms is None or ms <= 0.0:
        return None
    return _flops(mnk) / (ms * 1e-3) / 1e12


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _run_cublaslt_baseline(
    binary: Path | None,
    mnk: Tuple[int, int, int],
    warmup: int,
    iters: int,
    algos: int,
    workspace_mb: int,
) -> tuple[float | None, float | None]:
    if binary is None:
        return None, None
    if not binary.exists():
        raise FileNotFoundError(f"cuBLASLt benchmark binary not found: {binary}")

    proc = subprocess.run(
        [
            str(binary),
            "--mnk",
            ",".join(str(x) for x in mnk),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--algos",
            str(algos),
            "--workspace-mb",
            str(workspace_mb),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"RESULT benchmark=cublaslt .*? ms=([0-9.]+) tflops=([0-9.]+)", proc.stdout)
    if not match:
        raise RuntimeError(f"failed to parse cuBLASLt output:\n{proc.stdout}")
    return float(match.group(1)), float(match.group(2))

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


def _iter_variants(variant: str) -> list[str]:
    if variant == "all":
        return list(VARIANTS.keys())
    return [variant]


def _time_torch_kernel(fn, warmup: int, iters: int) -> float:
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


def _time_cute_kernel(
    module,
    a,
    b,
    c,
    a_tensor,
    b_tensor,
    c_tensor,
    warmup: int,
    iters: int,
) -> float:
    compiled_func = cute.compile(module.host_function, a_tensor, b_tensor, c_tensor)
    args = cute_testing.JitArguments(a_tensor, b_tensor, c_tensor)
    args.add_to_scope([a, b, c, a_tensor, b_tensor, c_tensor])
    time_us = cute_testing.benchmark(
        compiled_func,
        kernel_arguments=args,
        warmup_iterations=warmup,
        iterations=iters,
    )
    return time_us / 1e3


def benchmark_shape(
    variant: str,
    mnk: Tuple[int, int, int],
    atol: float | None,
    warmup: int,
    iters: int,
    cublaslt_bin: Path | None = None,
    cublaslt_algos: int = 32,
    cublaslt_workspace_mb: int = 64,
) -> dict:
    cfg = VARIANTS[variant]
    module = cfg["module"]
    atol = cfg["default_atol"] if atol is None else atol

    module.validate_mnk(mnk)
    a, b = make_inputs(mnk)
    c, a_tensor, b_tensor, c_tensor = module.prepare_cute_gemm(a, b)

    got = module.run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor)
    ref = torch_reference_gemm_with_dtype(a, b, cfg["torch_out_dtype"])
    check_close(got, ref, atol=atol, rtol=1e-5)

    cute_ms = _time_cute_kernel(
        module,
        a,
        b,
        c,
        a_tensor,
        b_tensor,
        c_tensor,
        warmup,
        iters,
    )
    torch_ms = _time_torch_kernel(
        lambda: torch_perf_gemm_with_dtype(a, b, cfg["torch_out_dtype"]),
        warmup,
        iters,
    )
    cublas_runner = make_torch_cublas_runner(a, b, cfg["torch_out_dtype"])
    cublas_ms = _time_torch_kernel(cublas_runner, warmup, iters)
    cublaslt_ms, cublaslt_tflops = _run_cublaslt_baseline(
        cublaslt_bin,
        mnk,
        warmup,
        iters,
        cublaslt_algos,
        cublaslt_workspace_mb,
    )

    cute_tflops = _tflops(mnk, cute_ms)
    torch_tflops = _tflops(mnk, torch_ms)
    cublas_tflops = _tflops(mnk, cublas_ms)
    return {
        "variant": variant,
        "mnk": mnk,
        "flops": _flops(mnk),
        "cute_ms": cute_ms,
        "torch_ms": torch_ms,
        "cublas_ms": cublas_ms,
        "cublaslt_ms": cublaslt_ms,
        "cute_tflops": cute_tflops,
        "torch_tflops": torch_tflops,
        "cublas_tflops": cublas_tflops,
        "cublaslt_tflops": cublaslt_tflops,
        "speedup_vs_torch": torch_ms / cute_ms,
        "speedup_vs_cublas": cublas_ms / cute_ms,
        "speedup_vs_cublaslt": None if cublaslt_ms is None else cublaslt_ms / cute_ms,
    }


def _candidate_names(group: str) -> list[str]:
    return [candidate.name for candidate in CANDIDATE_GROUPS[group]]


def benchmark_shape_autotuned(
    mnk: Tuple[int, int, int],
    group: str,
    atol: float | None,
    warmup: int,
    iters: int,
    cublaslt_bin: Path | None = None,
    cublaslt_algos: int = 32,
    cublaslt_workspace_mb: int = 64,
) -> dict:
    rows = []
    for candidate in CANDIDATE_GROUPS[group]:
        row = benchmark_shape(
            candidate.variant,
            mnk,
            atol,
            warmup,
            iters,
            cublaslt_bin,
            cublaslt_algos,
            cublaslt_workspace_mb,
        )
        row["candidate"] = candidate.to_dict()
        rows.append(row)
        print(
            "AUTOTUNE_CANDIDATE",
            {
                "mnk": mnk,
                "name": candidate.name,
                "variant": candidate.variant,
                "cute_ms": round(row["cute_ms"], 6),
                "cute_tflops": None if row["cute_tflops"] is None else round(row["cute_tflops"], 3),
            },
        )

    best = min(rows, key=lambda row: row["cute_ms"])
    best = dict(best)
    best["variant"] = "autotuned"
    best["selected_variant"] = best["candidate"]["variant"]
    best["selected_name"] = best["candidate"]["name"]
    best["autotune_group"] = group
    best["autotune_candidates"] = [row["candidate"] for row in rows]
    print(
        "AUTOTUNE_BEST",
        {
            "mnk": mnk,
            "name": best["selected_name"],
            "variant": best["selected_variant"],
            "cute_ms": round(best["cute_ms"], 6),
            "cute_tflops": None if best["cute_tflops"] is None else round(best["cute_tflops"], 3),
        },
    )
    return best


def print_results(rows: Iterable[dict]) -> None:
    print(
        "variant,selected_variant,selected_name,m,n,k,flops,"
        "cute_ms,torch_ms,cublas_ms,cublaslt_ms,"
        "cute_tflops,torch_tflops,cublas_tflops,cublaslt_tflops,"
        "speedup_vs_torch,speedup_vs_cublas,speedup_vs_cublaslt"
    )
    for row in rows:
        m, n, k = row["mnk"]
        print(
            f"{row['variant']},"
            f"{row.get('selected_variant', '')},"
            f"{row.get('selected_name', '')},"
            f"{m},{n},{k},{row['flops']:.0f},"
            f"{row['cute_ms']:.6f},"
            f"{row['torch_ms']:.6f},"
            f"{row['cublas_ms']:.6f},"
            f"{_format_optional(row['cublaslt_ms'])},"
            f"{_format_optional(row['cute_tflops'])},"
            f"{_format_optional(row['torch_tflops'])},"
            f"{_format_optional(row['cublas_tflops'])},"
            f"{_format_optional(row['cublaslt_tflops'])},"
            f"{row['speedup_vs_torch']:.6f},"
            f"{row['speedup_vs_cublas']:.6f},"
            f"{_format_optional(row['speedup_vs_cublaslt'])}"
        )



def print_result_row(row: dict) -> None:
    print(
        "RESULT",
        {
            "variant": row["variant"],
            "selected_variant": row.get("selected_variant"),
            "selected_name": row.get("selected_name"),
            "mnk": row["mnk"],
            "flops": row["flops"],
            "cute_ms": round(row["cute_ms"], 6),
            "torch_ms": round(row["torch_ms"], 6),
            "cublas_ms": round(row["cublas_ms"], 6),
            "cublaslt_ms": None if row["cublaslt_ms"] is None else round(row["cublaslt_ms"], 6),
            "cute_tflops": None if row["cute_tflops"] is None else round(row["cute_tflops"], 3),
            "torch_tflops": None if row["torch_tflops"] is None else round(row["torch_tflops"], 3),
            "cublas_tflops": None if row["cublas_tflops"] is None else round(row["cublas_tflops"], 3),
            "cublaslt_tflops": None if row["cublaslt_tflops"] is None else round(row["cublaslt_tflops"], 3),
            "speedup_vs_torch": round(row["speedup_vs_torch"], 6),
            "speedup_vs_cublas": round(row["speedup_vs_cublas"], 6),
            "speedup_vs_cublaslt": None if row["speedup_vs_cublaslt"] is None else round(row["speedup_vs_cublaslt"], 6),
        },
    )

def main():
    from cuda.bindings import driver as cu_driver

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["all", "autotuned", *VARIANTS.keys()],
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--cublaslt-bin",
        type=Path,
        default=None,
        help="optional path to compiled cute_gemm/cublaslt_benchmark binary",
    )
    parser.add_argument("--cublaslt-algos", type=int, default=32)
    parser.add_argument("--cublaslt-workspace-mb", type=int, default=64)
    parser.add_argument(
        "--autotune-group",
        choices=sorted(CANDIDATE_GROUPS.keys()),
        default="default",
        help="candidate group used when --variant autotuned",
    )
    args = parser.parse_args()

    cu_driver.cuInit(0)

    rows = []
    if args.variant == "autotuned":
        print(
            "AUTOTUNE",
            {
                "group": args.autotune_group,
                "candidates": _candidate_names(args.autotune_group),
            },
        )
        shape_source = "2cta_tma_pipeline"
        for mnk in _iter_shapes(shape_source, args.shape_set, args.shapes):
            row = benchmark_shape_autotuned(
                mnk,
                args.autotune_group,
                args.atol,
                args.warmup,
                args.iters,
                args.cublaslt_bin,
                args.cublaslt_algos,
                args.cublaslt_workspace_mb,
            )
            rows.append(row)
            print_result_row(row)
    else:
        for variant in _iter_variants(args.variant):
            for mnk in _iter_shapes(variant, args.shape_set, args.shapes):
                row = benchmark_shape(
                    variant,
                    mnk,
                    args.atol,
                    args.warmup,
                    args.iters,
                    args.cublaslt_bin,
                    args.cublaslt_algos,
                    args.cublaslt_workspace_mb,
                )
                rows.append(row)
                print_result_row(row)
    print_results(rows)


if __name__ == "__main__":
    main()
