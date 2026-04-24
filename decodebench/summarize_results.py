#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["file"] = path.name
        rows.append(row)
    return rows


def short_backend(name: str | None) -> str:
    return {
        "flashinfer_paged_decode": "flashinfer",
    }.get(name or "-", name or "-")


def workload_key(row: dict) -> tuple:
    return (
        row.get("attention"),
        row.get("kv_dtype"),
        row.get("batch_size"),
        row.get("context_len"),
        row.get("page_size"),
    )


def workload_name(key: tuple) -> str:
    attention, kv_dtype, batch_size, context_len, page_size = key
    return f"{attention:<3} {kv_dtype:<4} b{batch_size:<3} ctx{context_len:<5} p{page_size:<3}"


def ok_rows(group: list[dict]) -> list[dict]:
    return [row for row in group if row.get("status") != "failed" and row.get("kernel_latency_p50_us") is not None]


def failed_rows(group: list[dict]) -> list[dict]:
    return [row for row in group if row.get("status") == "failed"]


def winner_summary(group: list[dict]) -> str:
    valid = sorted(ok_rows(group), key=lambda row: row["kernel_latency_p50_us"])
    failed = failed_rows(group)
    if not valid:
        if failed:
            names = ", ".join(short_backend(row.get("backend")) for row in failed)
            return f"FAIL ({names})"
        return "-"

    winner = valid[0]
    winner_text = (
        f"winner {short_backend(winner.get('backend'))}: "
        f"{winner['kernel_latency_p50_us']:.1f}us, "
        f"{winner['approx_effective_kv_bandwidth_gb_s']:.0f}GB/s"
    )

    if len(valid) == 1 and not failed:
        return winner_text

    parts = [winner_text]
    for row in valid[1:]:
        parts.append(f"{short_backend(row.get('backend'))} {row['kernel_latency_p50_us'] / winner['kernel_latency_p50_us']:.2f}x")
    for row in failed:
        parts.append(f"{short_backend(row.get('backend'))} FAIL")
    return " | ".join(parts)


def print_summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    print(f"{'workload':<30} summary")
    print("-" * 88)
    for key in sorted(groups):
        print(f"{workload_name(key):<30} {winner_summary(groups[key])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print compact DecodeBench result summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("decodebench/results"))
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No JSON results found in {args.results_dir}")
        return 1
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
