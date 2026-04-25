#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ncu_utils import DEFAULT_NCU_METRICS, resolve_metrics, summarize_ncu


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
