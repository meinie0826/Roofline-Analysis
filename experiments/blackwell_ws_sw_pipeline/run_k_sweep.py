#!/usr/bin/env python3
"""Run K-sweep benchmarks for local WS/SW tutorial GEMM copies."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

VARIANTS = {
    "sw": {
        "label": "sw_pipeline",
        "script": THIS_DIR / "baselines" / "tutorial_fp16_gemm_1_sw_pipeline.py",
    },
    "ws": {
        "label": "warp_specialized",
        "script": THIS_DIR
        / "baselines"
        / "tutorial_fp16_gemm_2_warp_specialized.py",
    },
}


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc


def parse_variant_list(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variant(s): {','.join(unknown)}")
    return variants


def parse_result_line(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("RESULT,"):
            fields = {}
            for item in line.removeprefix("RESULT,").split(","):
                key, value = item.split("=", 1)
                fields[key] = value
            return fields
    raise RuntimeError("benchmark output did not contain a RESULT line")


def build_command(
    python: str,
    variant: str,
    m: int,
    n: int,
    k: int,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
) -> list[str]:
    command = [
        python,
        str(VARIANTS[variant]["script"]),
        "--mnk",
        f"{m},{n},{k}",
        "--do_benchmark",
        "--warmup_iterations",
        str(warmup_iterations),
        "--iterations",
        str(iterations),
    ]
    if skip_ref_check:
        command.append("--skip_ref_check")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local tutorial WS/SW GEMM copies over K values."
    )
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument(
        "--k_values",
        type=parse_int_list,
        default=parse_int_list("64,128,256,512,1024,2048,4096,8192"),
        help="Comma-separated K values. Use multiples of 64 for these kernels.",
    )
    parser.add_argument(
        "--variants",
        type=parse_variant_list,
        default=parse_variant_list("sw,ws"),
        help="Comma-separated subset of sw,ws.",
    )
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip torch reference check after the first compile launch.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch each benchmark subprocess.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to results/k_sweep_<timestamp>.csv.",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue the sweep if one point fails.",
    )
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or THIS_DIR / "results" / f"k_sweep_{timestamp}.csv"
    log_dir = THIS_DIR / "results" / f"logs_{timestamp}"
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "m",
        "n",
        "k",
        "avg_ms",
        "tflops",
        "iterations",
        "repeat",
        "returncode",
        "log",
        "command",
    ]

    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        env = os.environ.copy()
        cutlass_python_paths = [
            str(REPO_ROOT / "3rd" / "cutlass" / "python"),
            str(REPO_ROOT / "3rd" / "cutlass" / "examples" / "python"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            cutlass_python_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )

        for k in args.k_values:
            if k % 64 != 0:
                raise ValueError(f"K must be a multiple of 64 for this setup: {k}")
            for variant in args.variants:
                for repeat in range(args.repeats):
                    command = build_command(
                        args.python,
                        variant,
                        args.m,
                        args.n,
                        k,
                        args.warmup_iterations,
                        args.iterations,
                        args.skip_ref_check,
                    )
                    print(
                        f"[run] variant={variant} repeat={repeat} "
                        f"mnk={args.m},{args.n},{k}"
                    )
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    log_path = log_dir / f"{variant}_m{args.m}_n{args.n}_k{k}_r{repeat}.log"
                    log_path.write_text(
                        completed.stdout
                        + "\n\n--- STDERR ---\n"
                        + completed.stderr,
                        encoding="utf-8",
                    )

                    row = {
                        "variant": VARIANTS[variant]["label"],
                        "m": args.m,
                        "n": args.n,
                        "k": k,
                        "avg_ms": "",
                        "tflops": "",
                        "iterations": args.iterations,
                        "repeat": repeat,
                        "returncode": completed.returncode,
                        "log": str(log_path.relative_to(REPO_ROOT)),
                        "command": " ".join(command),
                    }

                    if completed.returncode == 0:
                        try:
                            row.update(parse_result_line(completed.stdout))
                        except RuntimeError as exc:
                            row["returncode"] = f"parse_error: {exc}"
                    writer.writerow(row)
                    csv_file.flush()

                    if completed.returncode != 0 and not args.keep_going:
                        print(f"[fail] see {log_path}")
                        return completed.returncode
                    if isinstance(row["returncode"], str) and not args.keep_going:
                        print(f"[fail] see {log_path}")
                        return 2

    print(f"[done] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
