#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
        "flashattn_kvcache": "flash-attn",
        "flashmla_decode": "flashmla",
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
    value = row.get("compare_latency_us") or row.get("kernel_latency_p50_us")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def row_status(row: dict) -> str:
    if row.get("status") == "failed":
        return "FAILED"
    if row.get("fallback"):
        return "FALLBACK"
    if compare_latency_us(row) is None:
        return "INVALID"
    return "SUCCESS"


def format_metric(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_bw(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def print_summary(rows: list[dict], reference_backend: str | None = None) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    print("Operator: decode_attention  Performance Test (mode=kernel, level=sota)")
    print(
        f"{'Status':<12}"
        f"{'Latency (us)':>16}"
        f"{'Vs Best':>12}"
        f"{'GB/s':>10}"
        f"{'Compare':>18}  "
        f"Workload Detail"
    )
    print("-" * 108)
    for key in sorted(groups):
        group = groups[key]
        valid_rows = [row for row in group if row_status(row) == "SUCCESS"]
        best_latency = min((compare_latency_us(row) for row in valid_rows), default=None)
        ordered = sorted(
            group,
            key=lambda row: (
                row_status(row) != "SUCCESS",
                compare_latency_us(row) if compare_latency_us(row) is not None else float("inf"),
                short_backend(row.get("backend")),
            ),
        )
        for row in ordered:
            latency = compare_latency_us(row)
            cand_bw = row.get("approx_effective_kv_bandwidth_gb_s")
            status = row_status(row)
            if status == "SUCCESS" and best_latency and latency:
                speedup = best_latency / latency
            else:
                speedup = None
            print(
                f"{status:<12}"
                f"{format_metric(latency, 1):>16}"
                f"{format_metric(speedup, 3):>12}"
                f"{format_bw(cand_bw):>10}"
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
