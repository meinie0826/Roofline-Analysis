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


def fmt(value, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def workload_key(row: dict) -> tuple:
    return (
        row.get("attention"),
        row.get("kv_dtype"),
        row.get("batch_size"),
        row.get("context_len"),
        row.get("page_size"),
    )


def print_table(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    for key in sorted(groups):
        attention, kv_dtype, batch_size, context_len, page_size = key
        print(f"\n[{attention} {kv_dtype} b{batch_size} ctx{context_len} p{page_size}]")
        print(f"{'backend':<24} {'status':<8} {'p50_us':>10} {'p95_us':>10} {'GB/s':>10} {'alloc_GB':>10}  notes")
        print("-" * 96)
        group = sorted(
            groups[key],
            key=lambda row: row.get("kernel_latency_p50_us") if row.get("kernel_latency_p50_us") is not None else float("inf"),
        )
        best = group[0].get("kernel_latency_p50_us") if group and group[0].get("kernel_latency_p50_us") is not None else None
        for row in group:
            p50 = row.get("kernel_latency_p50_us")
            ratio = ""
            if best and p50:
                ratio = " winner" if p50 == best else f" {p50 / best:.2f}x"
            status = row.get("status", "ok")
            notes = row.get("notes") or row.get("command") or ""
            print(
                f"{row.get('backend', '-'):<24} "
                f"{status:<8} "
                f"{fmt(p50, digits=1):>10} "
                f"{fmt(row.get('kernel_latency_p95_us'), digits=1):>10} "
                f"{fmt(row.get('approx_effective_kv_bandwidth_gb_s'), digits=1):>10} "
                f"{fmt(row.get('peak_allocated_gb'), digits=2):>10}  "
                f"{ratio}{(' | ' + notes) if notes else ''}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print DecodeBench JSON results as readable tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("decodebench/results"))
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No JSON results found in {args.results_dir}")
        return 1
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
