#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from summarize_results import print_summary


def shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


def flashinfer_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        "python3",
        "decodebench/flashinfer_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def flashinfer_trtllm_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        "python3",
        "decodebench/trtllm_decode_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def vllm_attention_cmd(workload: dict, defaults: dict, output: Path, backend: str) -> list[str]:
    return [
        "python3",
        "decodebench/vllm_attention_benchmark.py",
        "--backend", backend,
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def build_cmd(backend: dict, workload: dict, defaults: dict, results_dir: Path) -> list[str]:
    output = results_dir / f'{backend["name"]}__{workload["id"]}.json'
    if backend["name"] == "flashinfer_paged_decode":
        return flashinfer_cmd(workload, defaults, output)
    if backend["name"] == "flashinfer_trtllm_decode":
        return flashinfer_trtllm_cmd(workload, defaults, output)
    if backend["name"] == "vllm_flash":
        return vllm_attention_cmd(workload, defaults, output, "flash")
    if backend["name"] == "vllm_flashinfer":
        return vllm_attention_cmd(workload, defaults, output, "flashinfer")
    raise ValueError(f'Backend not implemented yet: {backend["name"]}')


def is_supported(backend: dict, workload: dict) -> bool:
    supported_kv_dtypes = backend.get("supported_kv_dtypes")
    return supported_kv_dtypes is None or workload["kv_dtype"] in supported_kv_dtypes


def output_path(backend: dict, workload: dict, results_dir: Path) -> Path:
    return results_dir / f'{backend["name"]}__{workload["id"]}.json'


def short_backend(name: str) -> str:
    return {
        "flashinfer_paged_decode": "flashinfer",
        "flashinfer_trtllm_decode": "trtllm",
        "vllm_flash": "vllm-flash",
        "vllm_flashinfer": "vllm-flashinfer",
    }.get(name, name)


def write_failure(path: Path, backend: dict, workload: dict, returncode: int, command: list[str], output: str) -> None:
    result = {
        "status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend["name"],
        "kernel_path": backend.get("kernel_path"),
        "workload_id": workload["id"],
        "attention": workload["attention"],
        "kv_dtype": workload["kv_dtype"],
        "page_size": workload["page_size"],
        "batch_size": workload["batch_size"],
        "context_len": workload["context_len"],
        "returncode": returncode,
        "command": shell(command),
        "output": output,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("decodebench_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load matrix file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="Run decode attention kernel benchmark matrix.")
    parser.add_argument("--config", type=Path, default=ROOT / "matrix_b200.py")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-backend", default="flashinfer_paged_decode")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        parser.error("choose exactly one: --dry-run or --execute")

    config = load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures = 0

    for backend in config["backends"]:
        if not backend.get("enabled", True):
            continue
        for workload in config["workloads"]:
            if not is_supported(backend, workload):
                continue
            argv = build_cmd(backend, workload, config.get("defaults", {}), args.results_dir)
            if args.dry_run:
                print(shell(argv))
            else:
                env = os.environ.copy()
                env["DECODEBENCH_RUN_ID"] = run_id
                env["DECODEBENCH_WORKLOAD_ID"] = workload["id"]
                completed = subprocess.run(argv, check=False, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if args.verbose and completed.stdout:
                    print(completed.stdout, end="")
                if completed.returncode != 0:
                    failures += 1
                    write_failure(output_path(backend, workload, args.results_dir), backend, workload, completed.returncode, argv, completed.stdout)
                    print(f"✗ {workload['id']:<28} {short_backend(backend['name'])}")
                else:
                    print(f"✓ {workload['id']:<28} {short_backend(backend['name'])}")

    if args.execute and args.report:
        print()
        rows = []
        for path in sorted(args.results_dir.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        if rows:
            print_summary(rows)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
