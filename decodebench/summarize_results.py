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
        "flashinfer_trtllm_decode": "trtllm",
        "vllm_flash": "vllm-flash",
        "vllm_flashinfer": "vllm-flashinfer",
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


def compare_latency_us(row: dict) -> float | None:
    return row.get("compare_latency_us") or row.get("kernel_latency_p50_us")


def failed_rows(group: list[dict]) -> list[dict]:
    return [row for row in group if row.get("status") == "failed"]


def ok_rows(group: list[dict]) -> list[dict]:
    return [row for row in group if row.get("status") != "failed" and compare_latency_us(row) is not None]


def format_metric(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_bw(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def print_summary(rows: list[dict], reference_backend: str) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    print("Operator: decode_attention  Performance Test (mode=kernel, level=sota)")
    print(
        f"{'Status':<12}"
        f"{'Ref Latency (us)':>18}"
        f"{'Cand Latency (us)':>20}"
        f"{'Cand Speedup':>16}"
        f"{'Ref GB/s':>12}"
        f"{'Cand GB/s':>12}"
        f"{'Compare':>18}  "
        f"Workload Detail"
    )
    print("-" * 140)
    for key in sorted(groups):
        group = groups[key]
        ref = next((row for row in group if row.get("backend") == reference_backend), None)
        ref_latency = compare_latency_us(ref) if ref else None
        ref_bw = ref.get("approx_effective_kv_bandwidth_gb_s") if ref else None
        ordered = sorted(group, key=lambda row: short_backend(row.get("backend")))
        for row in ordered:
            cand_latency = compare_latency_us(row)
            cand_bw = row.get("approx_effective_kv_bandwidth_gb_s")
            status = "FAILED" if row.get("status") == "failed" else "SUCCESS"
            if status == "SUCCESS" and ref_latency and cand_latency:
                speedup = ref_latency / cand_latency
            else:
                speedup = None
            print(
                f"{status:<12}"
                f"{format_metric(ref_latency, 1):>18}"
                f"{format_metric(cand_latency, 1):>20}"
                f"{format_metric(speedup):>16}"
                f"{format_bw(ref_bw):>12}"
                f"{format_bw(cand_bw):>12}"
                f"{short_backend(row.get('backend')):>18}  "
                f"{workload_name(key)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print compact DecodeBench result summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("decodebench/results"))
    parser.add_argument("--reference-backend", default="flashinfer_paged_decode")
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No JSON results found in {args.results_dir}")
        return 1
    print_summary(rows, args.reference_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
