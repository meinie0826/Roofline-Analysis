#!/usr/bin/env python3
"""Build and run the DSMEM cluster-size microbenchmark."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "dsmem_cluster_bench"


def run(cmd: list[str], cwd: Path = ROOT, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )


def parse_jsonl(output: str) -> list[dict]:
    rows: list[dict] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def default_output(gpu_label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in gpu_label).strip("_") or "gpu"
    return ROOT / "results" / f"{safe}_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-label", default="auto", help="Label for this run, e.g. B200 or B300.")
    parser.add_argument("--arch", default=None, help="NVCC arch, e.g. sm_100a for B200 or sm_103a for B300.")
    parser.add_argument("--no-build", action="store_true", help="Skip make before running.")
    parser.add_argument("--output", type=Path, default=None, help="JSON result path.")
    parser.add_argument("--cluster-sizes", default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-iters", type=int, default=200_000)
    parser.add_argument("--bandwidth-iters", type=int, default=4096)
    parser.add_argument("--latency-elems", type=int, default=4096)
    parser.add_argument("--bandwidth-bytes", type=int, default=32768)
    parser.add_argument("--global-latency-elems", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--global-bandwidth-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--block-threads", type=int, default=128)
    parser.add_argument("--max-grid-blocks", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Short smoke-test run.")
    args = parser.parse_args()

    if args.quick:
        args.repeats = 3
        args.warmup = 1
        args.latency_iters = 10_000
        args.bandwidth_iters = 128
        args.cluster_sizes = "1,2,4"
        args.global_latency_elems = 4 * 1024 * 1024
        args.global_bandwidth_bytes = 64 * 1024 * 1024

    if not args.no_build:
        if shutil.which("nvcc") is None:
            raise SystemExit("nvcc not found. Load CUDA first or rerun with --no-build if the binary already exists.")
        make_cmd = ["make"]
        if args.arch:
            make_cmd.append(f"ARCH={args.arch}")
        run(make_cmd)

    if not BIN.exists():
        raise SystemExit(f"Benchmark binary not found: {BIN}")

    bench_cmd = [
        str(BIN),
        f"--gpu-label={args.gpu_label}",
        f"--cluster-sizes={args.cluster_sizes}",
        f"--repeats={args.repeats}",
        f"--warmup={args.warmup}",
        f"--latency-iters={args.latency_iters}",
        f"--bandwidth-iters={args.bandwidth_iters}",
        f"--latency-elems={args.latency_elems}",
        f"--bandwidth-bytes={args.bandwidth_bytes}",
        f"--global-latency-elems={args.global_latency_elems}",
        f"--global-bandwidth-bytes={args.global_bandwidth_bytes}",
        f"--block-threads={args.block_threads}",
        f"--max-grid-blocks={args.max_grid_blocks}",
    ]
    completed = run(bench_cmd, capture=True)
    assert completed.stdout is not None
    print(completed.stdout, end="")
    rows = parse_jsonl(completed.stdout)
    if not rows:
        raise SystemExit("Benchmark produced no JSON rows.")

    output = args.output or default_output(args.gpu_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "command": bench_cmd,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
