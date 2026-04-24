#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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


def build_cmd(backend: dict, workload: dict, defaults: dict, results_dir: Path) -> list[str]:
    output = results_dir / f'{backend["name"]}__{workload["id"]}.json'
    if backend["name"] == "flashinfer_paged_decode":
        return flashinfer_cmd(workload, defaults, output)
    raise ValueError(f'Backend not implemented yet: {backend["name"]}')


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
    args = parser.parse_args()

    if args.dry_run == args.execute:
        parser.error("choose exactly one: --dry-run or --execute")

    config = load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for backend in config["backends"]:
        if not backend.get("enabled", True):
            continue
        for workload in config["workloads"]:
            argv = build_cmd(backend, workload, config.get("defaults", {}), args.results_dir)
            print(shell(argv))
            if args.execute:
                subprocess.run(argv, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
