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
        "torch_sdpa_decode": "sdpa",
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


def row_summary(row: dict, best_us: float | None) -> str:
    if row.get("status") == "failed":
        return f"{short_backend(row.get('backend'))}: FAIL"
    p50 = row.get("kernel_latency_p50_us")
    bandwidth = row.get("approx_effective_kv_bandwidth_gb_s")
    if p50 is None:
        return f"{short_backend(row.get('backend'))}: -"
    tag = "best" if best_us == p50 else f"{p50 / best_us:.2f}x" if best_us else "-"
    return f"{short_backend(row.get('backend'))}: {p50:.1f}us, {bandwidth:.0f}GB/s, {tag}"


def print_summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    print(f"{'workload':<30} result")
    print("-" * 96)
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda row: row.get("kernel_latency_p50_us") if row.get("kernel_latency_p50_us") is not None else float("inf"),
        )
        ok_latencies = [row["kernel_latency_p50_us"] for row in group if row.get("kernel_latency_p50_us") is not None]
        best_us = min(ok_latencies) if ok_latencies else None
        result = " | ".join(row_summary(row, best_us) for row in group)
        print(f"{workload_name(key):<30} {result}")


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
