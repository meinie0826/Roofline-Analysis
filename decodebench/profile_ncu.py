#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from run_matrix import build_cmd, is_supported, load_config, shell, short_backend

DEFAULT_NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_wgmma_pred_on.sum",
]

TENSOR_CORE_METRIC_PATTERNS = (
    "tensor_op",
    "pipe_tensor",
    "hmma",
    "wgmma",
    "mma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DecodeBench backend/workload pairs with Nsight Compute and summarize Tensor Core utilization.")
    parser.add_argument("--config", type=Path, default=ROOT / "matrix_b200.py")
    parser.add_argument("--workload-id", action="append", required=True, help="Workload id to profile. May be repeated.")
    parser.add_argument("--backend", action="append", required=True, help="Backend name to profile. May be repeated.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles")
    parser.add_argument("--ncu", default="ncu", help="Nsight Compute CLI path.")
    parser.add_argument("--metric", action="append", dest="metrics", help="NCU metric to collect. May be repeated. Defaults include Tensor Core/SM/DRAM/time metrics.")
    parser.add_argument("--section", action="append", dest="sections", help="Optional NCU section to collect, e.g. SpeedOfLight. May be repeated.")
    parser.add_argument("--kernel-name", default=None, help="Optional NCU --kernel-name regex filter.")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1, help="Limit profiled launches to keep runs short; increase when the first launch is setup-only.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Override benchmark warmup steps during profiling.")
    parser.add_argument("--repeat", type=int, default=5, help="Override benchmark repeat count during profiling.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing profile JSON/CSV.")
    parser.add_argument("--query-metrics", action=argparse.BooleanOptionalAction, default=True, help="Query NCU and keep only available default metrics before profiling.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_metric_value(value: str) -> float | str | None:
    text = value.strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "INF", "NAN"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return value.strip()


def load_ncu_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            if not raw_line.lstrip().startswith('"ID",') and not raw_line.lstrip().startswith("ID,"):
                continue
            header = next(csv.reader([raw_line]))
            reader = csv.DictReader(handle, fieldnames=header)
            for row in reader:
                if not row.get("ID"):
                    continue
                metric_name = row.get("Metric Name") or row.get("Metric") or ""
                if not metric_name:
                    continue
                rows.append(row)
            break
    return rows


def summarize_ncu(csv_path: Path) -> dict[str, Any]:
    rows = load_ncu_rows(csv_path)
    by_kernel: dict[str, dict[str, Any]] = {}
    for row in rows:
        kernel = row.get("Kernel Name") or row.get("Kernel") or "unknown_kernel"
        metric = row.get("Metric Name") or "unknown_metric"
        unit = row.get("Metric Unit") or ""
        value = normalize_metric_value(row.get("Metric Value", ""))
        entry = by_kernel.setdefault(kernel, {"kernel_name": kernel, "metrics": {}})
        entry["metrics"][metric] = {"value": value, "unit": unit}

    tc_metric_names = sorted({
        row.get("Metric Name", "")
        for row in rows
        if any(token in row.get("Metric Name", "").lower() for token in TENSOR_CORE_METRIC_PATTERNS)
    })
    pct_metric_names = [name for name in tc_metric_names if "pct_of_peak" in name]

    def values_for(metric_name: str) -> list[tuple[float, float | None]]:
        values: list[tuple[float, float | None]] = []
        for kernel in by_kernel.values():
            metric = kernel["metrics"].get(metric_name)
            if not metric:
                continue
            value = metric["value"]
            if not isinstance(value, float):
                continue
            time_metric = kernel["metrics"].get("gpu__time_duration.sum")
            time_value = time_metric["value"] if time_metric else None
            values.append((value, time_value if isinstance(time_value, float) else None))
        return values

    tensor_core_summary: dict[str, Any] = {}
    for metric_name in pct_metric_names:
        values = values_for(metric_name)
        if not values:
            continue
        numeric = [value for value, _ in values]
        weighted_pairs = [(value, weight) for value, weight in values if weight and weight > 0]
        if weighted_pairs:
            weighted = sum(value * weight for value, weight in weighted_pairs) / sum(weight for _, weight in weighted_pairs)
        else:
            weighted = None
        tensor_core_summary[metric_name] = {
            "max_pct": max(numeric),
            "avg_pct": sum(numeric) / len(numeric),
            "time_weighted_pct": weighted,
        }

    selected_tc = None
    for preferred in DEFAULT_NCU_METRICS:
        if preferred in tensor_core_summary:
            item = tensor_core_summary[preferred]
            selected_tc = item["time_weighted_pct"] if item["time_weighted_pct"] is not None else item["max_pct"]
            break
    if selected_tc is None and tensor_core_summary:
        first = next(iter(tensor_core_summary.values()))
        selected_tc = first["time_weighted_pct"] if first["time_weighted_pct"] is not None else first["max_pct"]

    return {
        "ncu_metric_rows": len(rows),
        "ncu_kernel_count": len(by_kernel),
        "ncu_tensor_core_metric_names": tc_metric_names,
        "ncu_tensor_core_util_pct": selected_tc,
        "ncu_tensor_core_summary": tensor_core_summary,
        "ncu_kernels": list(by_kernel.values()),
    }



def query_available_metrics(ncu: str) -> set[str]:
    commands = ([ncu, "--query-metrics"], [ncu, "--query-metrics", "--query-metrics-mode", "all"])
    for command in commands:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            continue
        return set(re.findall(r"\b[a-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)*\b", completed.stdout))
    return set()


def resolve_metrics(ncu: str, requested: list[str], should_query: bool) -> tuple[list[str], list[str]]:
    if not should_query:
        return requested, []
    available = query_available_metrics(ncu)
    if not available:
        return requested, ["Could not query NCU metrics; using requested metrics unchanged."]
    selected = [metric for metric in requested if metric in available]
    warnings: list[str] = []
    missing = [metric for metric in requested if metric not in available]
    if missing:
        warnings.append(f"NCU metrics unavailable and skipped: {', '.join(missing)}")
    if not any(any(token in metric.lower() for token in TENSOR_CORE_METRIC_PATTERNS) for metric in selected):
        candidates = sorted(
            metric for metric in available
            if "pct_of_peak" in metric and any(token in metric.lower() for token in TENSOR_CORE_METRIC_PATTERNS)
        )
        if candidates:
            selected.append(candidates[0])
            warnings.append(f"Added available Tensor Core-like metric: {candidates[0]}")
    if not selected:
        return requested, warnings + ["No requested metrics matched NCU query; using requested metrics unchanged so NCU reports the exact error."]
    return selected, warnings

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    defaults = dict(config.get("defaults", {}))
    defaults["warmup_steps"] = args.warmup_steps
    defaults["repeat"] = args.repeat
    metrics, metric_warnings = resolve_metrics(args.ncu, args.metrics or DEFAULT_NCU_METRICS, args.query_metrics)
    for warning in metric_warnings:
        print(f"# {warning}")

    backends = {backend["name"]: backend for backend in config["backends"]}
    workloads = {workload["id"]: workload for workload in config["workloads"]}
    missing_backends = [name for name in args.backend if name not in backends]
    missing_workloads = [name for name in args.workload_id if name not in workloads]
    if missing_backends or missing_workloads:
        raise SystemExit(f"Unknown backend(s)={missing_backends}, workload(s)={missing_workloads}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for backend_name in args.backend:
        backend = backends[backend_name]
        for workload_id in args.workload_id:
            workload = workloads[workload_id]
            if not is_supported(backend, workload):
                print(f"- {workload_id:<28} {short_backend(backend_name)} | unsupported")
                continue
            stem = f"{safe_name(backend_name)}__{safe_name(workload_id)}"
            csv_path = args.output_dir / f"{stem}.ncu.csv"
            result_path = args.output_dir / f"{stem}.profile.json"
            bench_result_path = args.output_dir / f"{stem}.bench.json"
            if result_path.exists() and not args.force:
                print(f"↷ {workload_id:<28} {short_backend(backend_name)} | existing profile")
                continue
            bench_cmd = build_cmd(backend, workload, defaults, args.output_dir)
            bench_cmd[-1] = str(bench_result_path)
            ncu_cmd = [
                args.ncu,
                "--target-processes", "all",
                "--csv",
                "--page", "raw",
                "--launch-skip", str(args.launch_skip),
                "--launch-count", str(args.launch_count),
                "--log-file", str(csv_path),
            ]
            for section in args.sections or []:
                ncu_cmd += ["--section", section]
            if metrics:
                ncu_cmd += ["--metrics", ",".join(metrics)]
            if args.kernel_name:
                ncu_cmd += ["--kernel-name", args.kernel_name]
            ncu_cmd += bench_cmd

            env = os.environ.copy()
            env["DECODEBENCH_RUN_ID"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            env["DECODEBENCH_WORKLOAD_ID"] = workload_id
            completed = subprocess.run(ncu_cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if args.verbose and completed.stdout:
                print(completed.stdout, end="")
            result: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": backend_name,
                "backend_label": short_backend(backend_name),
                "workload_id": workload_id,
                "workload": workload,
                "ncu_command": shell(ncu_cmd),
                "benchmark_command": shell(bench_cmd),
                "ncu_csv": str(csv_path),
                "benchmark_result": str(bench_result_path),
                "metrics_requested": metrics,
                "metric_warnings": metric_warnings,
                "sections_requested": args.sections or [],
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
            if completed.returncode == 0 and csv_path.exists():
                result.update(summarize_ncu(csv_path))
                print(f"✓ {workload_id:<28} {short_backend(backend_name)} | TC {result.get('ncu_tensor_core_util_pct')}")
            else:
                failures += 1
                print(f"✗ {workload_id:<28} {short_backend(backend_name)} | ncu failed")
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
