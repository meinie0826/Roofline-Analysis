import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

from cuda.bindings import driver as cu_driver

from benchmark import VARIANTS, _iter_shapes, _parse_mnk, benchmark_candidate_shape
from configs import CANDIDATE_GROUPS, GemmCandidate


RESULT_DIR = Path(__file__).resolve().parent / "autotune_results"


def _candidate_names(candidates: Iterable[GemmCandidate]) -> list[str]:
    return [candidate.name for candidate in candidates]


def _select_candidates(group: str, variants: list[str] | None) -> tuple[GemmCandidate, ...]:
    candidates = CANDIDATE_GROUPS[group]
    if not variants:
        return tuple(candidates)

    selected = []
    variant_set = set(variants)
    for candidate in candidates:
        if candidate.name in variant_set or candidate.variant in variant_set:
            selected.append(candidate)

    missing = variant_set - {candidate.name for candidate in selected} - {
        candidate.variant for candidate in selected
    }
    if missing:
        raise ValueError(f"unknown autotune candidate(s): {sorted(missing)}")
    return tuple(selected)


def _shape_key(mnk: Tuple[int, int, int]) -> str:
    return "x".join(str(x) for x in mnk)


def _best_row(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: row["cute_ms"])


def autotune_shape(
    mnk: Tuple[int, int, int],
    candidates: tuple[GemmCandidate, ...],
    atol: float | None,
    warmup: int,
    iters: int,
    cublaslt_bin=None,
    cublaslt_algos: int = 32,
    cublaslt_workspace_mb: int = 64,
) -> dict:
    rows = []
    for candidate in candidates:
        try:
            row = benchmark_candidate_shape(
                candidate,
                mnk,
                atol,
                warmup,
                iters,
                cublaslt_bin,
                cublaslt_algos,
                cublaslt_workspace_mb,
            )
        except ValueError as error:
            print(
                "AUTOTUNE_SKIP",
                {
                    "shape": mnk,
                    "name": candidate.name,
                    "variant": candidate.variant,
                    "reason": str(error),
                },
            )
            continue
        row["candidate"] = candidate.to_dict()
        rows.append(row)
        print(
            "CANDIDATE",
            {
                "shape": mnk,
                "name": candidate.name,
                "variant": candidate.variant,
                "cute_ms": round(row["cute_ms"], 6),
                "cute_tflops": None if row["cute_tflops"] is None else round(row["cute_tflops"], 3),
                "cublas_ms": round(row["cublas_ms"], 6),
                "cublas_tflops": None if row["cublas_tflops"] is None else round(row["cublas_tflops"], 3),
                "cublaslt_ms": None if row["cublaslt_ms"] is None else round(row["cublaslt_ms"], 6),
                "cublaslt_tflops": None if row["cublaslt_tflops"] is None else round(row["cublaslt_tflops"], 3),
                "speedup_vs_cublas": round(row["speedup_vs_cublas"], 6),
            },
        )

    if not rows:
        raise ValueError(f"no compatible autotune candidates for shape {mnk}")

    best = _best_row(rows)
    print(
        "BEST",
        {
            "shape": mnk,
            "name": best["candidate"]["name"],
            "variant": best["variant"],
            "cute_ms": round(best["cute_ms"], 6),
            "cute_tflops": None if best["cute_tflops"] is None else round(best["cute_tflops"], 3),
        },
    )
    return {"mnk": mnk, "best": best, "candidates": rows}


def write_results(result: dict, output: Path | None) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = RESULT_DIR / f"autotune_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latest = RESULT_DIR / "latest.json"
    latest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def print_summary(results: list[dict]) -> None:
    print("mnk,best_name,best_variant,flops,cute_ms,cute_tflops,cublas_ms,cublas_tflops,cublaslt_ms,cublaslt_tflops,speedup_vs_cublas,speedup_vs_cublaslt")
    for result in results:
        best = result["best"]
        m, n, k = result["mnk"]
        print(
            f"{m}x{n}x{k},"
            f"{best['candidate']['name']},"
            f"{best['variant']},"
            f"{best['flops']:.0f},"
            f"{best['cute_ms']:.6f},"
            f"{_fmt(best['cute_tflops'])},"
            f"{best['cublas_ms']:.6f},"
            f"{_fmt(best['cublas_tflops'])},"
            f"{_fmt(best['cublaslt_ms'])},"
            f"{_fmt(best['cublaslt_tflops'])},"
            f"{best['speedup_vs_cublas']:.6f},"
            f"{_fmt(best['speedup_vs_cublaslt'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=sorted(CANDIDATE_GROUPS.keys()),
        default="default",
        help="candidate group to autotune",
    )
    parser.add_argument(
        "--base-variant",
        choices=VARIANTS.keys(),
        default="2cta_tma_pipeline",
        help="shape-set source when --shapes is not provided",
    )
    parser.add_argument(
        "--shape-set",
        choices=["small", "large", "all"],
        default="all",
    )
    parser.add_argument("--shapes", type=_parse_mnk, nargs="*", default=None)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="candidate names or variant names from the selected group",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cublaslt-bin",
        type=Path,
        default=None,
        help="optional path to compiled cute_gemm/cublaslt_benchmark binary",
    )
    parser.add_argument("--cublaslt-algos", type=int, default=32)
    parser.add_argument("--cublaslt-workspace-mb", type=int, default=64)
    args = parser.parse_args()

    candidates = _select_candidates(args.group, args.candidates)
    print("AUTOTUNE", {"group": args.group, "candidates": _candidate_names(candidates)})

    cu_driver.cuInit(0)
    shapes = list(_iter_shapes(args.base_variant, args.shape_set, args.shapes))
    results = [
        autotune_shape(
            shape,
            candidates,
            args.atol,
            args.warmup,
            args.iters,
            args.cublaslt_bin,
            args.cublaslt_algos,
            args.cublaslt_workspace_mb,
        )
        for shape in shapes
    ]
    payload = {
        "group": args.group,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "results": results,
        "best_by_shape": {
            _shape_key(result["mnk"]): result["best"]["candidate"] for result in results
        },
    }
    output = write_results(payload, args.output)
    print_summary(results)
    print("WROTE", str(output))


if __name__ == "__main__":
    main()
