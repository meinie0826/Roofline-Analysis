#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_rows(results_dir: Path, run_id: str | None = None, latest_run_only: bool = False) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["file"] = path.name
        rows.append(row)
    if run_id is not None:
        return [row for row in rows if row.get("run_id") == run_id]
    if latest_run_only:
        run_ids = sorted({row.get("run_id") for row in rows if row.get("run_id")})
        if run_ids:
            latest_run_id = run_ids[-1]
            return [row for row in rows if row.get("run_id") == latest_run_id]
    return rows


def short_backend(name: str | None) -> str:
    return {
        "flashinfer_paged_decode": "flashinfer",
        "flashinfer_trtllm_decode": "trtllm",
        "flashattn_kvcache": "flash-attn",
        "flashattn_mla_decode": "flash-attn-mla",
        "flashmla_decode": "flashmla",
        "flashinfer_trtllm_mla_decode": "trtllm-mla",
        "vllm_paged_decode": "vllm-paged",
        "vllm_flash": "vllm-flash",
        "vllm_flashinfer": "vllm-flashinfer",
        "torch_sdpa_auto": "torch-sdpa",
        "torch_sdpa_cudnn": "torch-cudnn",
        "torch_sdpa_flash": "torch-flash",
        "tensorrt_llm_native": "trtllm-native",
        "sglang_serving": "sglang",
    }.get(name or "-", name or "-")


def display_backend(row: dict) -> str:
    name = row.get("backend")
    selected = str(row.get("selected_backend") or "")
    if name == "flashattn_kvcache":
        if selected.startswith("fa4"):
            return "flash-attn-fa4"
        if selected.startswith("fa3"):
            return "flash-attn-fa3"
        if selected.startswith("fa2"):
            return "flash-attn-fa2"
    return short_backend(name)


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


def backend_order(name: str) -> tuple[int, str]:
    order = {
        "trtllm": 0,
        "flashinfer": 1,
        "flash-attn-fa4": 2,
        "flash-attn-fa3": 3,
        "flash-attn-fa2": 4,
        "flashmla": 5,
        "trtllm-mla": 6,
        "flash-attn-mla": 7,
        "vllm-paged": 8,
        "vllm-flash": 9,
        "vllm-flashinfer": 10,
        "trtllm-native": 11,
        "torch-cudnn": 12,
        "torch-flash": 13,
        "torch-sdpa": 14,
        "sglang": 15,
    }
    return order.get(name, 100), name


def table_cell(row: dict | None, best_latency: float | None) -> str:
    if row is None:
        return "-"
    status = row_status(row)
    if status != "SUCCESS":
        reason = str(row.get("short_reason") or row.get("fallback_reason") or status)
        if "Unsupported block size" in reason:
            return "FAIL:block"
        if "ImportError" in reason or "ModuleNotFoundError" in reason:
            return "FAIL:import"
        return "FAIL"
    latency = compare_latency_us(row)
    if latency is None:
        return "-"
    if best_latency and latency > 0:
        return f"{latency:.1f}/{best_latency / latency:.2f}x"
    return f"{latency:.1f}"


def print_pivot_summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    backends = set()
    for row in rows:
        groups[workload_key(row)].append(row)
        backends.add(display_backend(row))

    backend_names = sorted(backends, key=backend_order)
    mode = "kernel" if all(row.get("layer") == "kernel" for row in rows) else "mixed"
    print(f"Operator: decode_attention  Performance Test (mode={mode}, level=sota)")
    print("Cell: latency_us/vs_best_x; '-' means not scheduled for that backend/workload")

    workload_width = max(28, max((len(workload_name(key)) for key in groups), default=28))
    cell_width = 16
    header = f"{'Workload':<{workload_width}}" + "".join(f"{name:>{cell_width}}" for name in backend_names)
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        group = groups[key]
        valid_rows = [row for row in group if row_status(row) == "SUCCESS"]
        best_latency = min((compare_latency_us(row) for row in valid_rows), default=None)
        by_backend = {}
        for row in group:
            backend = display_backend(row)
            previous = by_backend.get(backend)
            if previous is None or (compare_latency_us(row) or float("inf")) < (compare_latency_us(previous) or float("inf")):
                by_backend[backend] = row
        line = f"{workload_name(key):<{workload_width}}" + "".join(
            f"{table_cell(by_backend.get(name), best_latency):>{cell_width}}" for name in backend_names
        )
        print(line)


def print_long_summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[workload_key(row)].append(row)

    mode = "kernel" if all(row.get("layer") == "kernel" for row in rows) else "mixed"
    print(f"Operator: decode_attention  Performance Test (mode={mode}, level=sota)")
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
                display_backend(row),
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
                f"{display_backend(row):>18}  "
                f"{workload_name(key)}"
            )


def print_summary(rows: list[dict], reference_backend: str | None = None) -> None:
    print_pivot_summary(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print compact DecodeBench result summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("decodebench/results"))
    parser.add_argument("--run-id")
    parser.add_argument("--latest-run-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-backend", default="flashinfer_paged_decode")
    parser.add_argument("--format", choices=["pivot", "long"], default="pivot")
    args = parser.parse_args()

    rows = load_rows(args.results_dir, run_id=args.run_id, latest_run_only=args.latest_run_only)
    if not rows:
        print(f"No JSON results found in {args.results_dir}")
        return 1
    if args.format == "long":
        print_long_summary(rows)
    else:
        print_summary(rows, args.reference_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
