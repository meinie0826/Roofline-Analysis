from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_wgmma_pred_on.sum",
]

TENSOR_CORE_METRIC_PATTERNS = (
    "tensor_op",
    "pipe_tensor",
    "hmma",
    "wgmma",
    "mma",
)


def normalize_metric_value(value: str) -> float | str | None:
    text = value.strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "INF", "NAN"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return value.strip()


def load_ncu_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            if not raw_line.lstrip().startswith('"ID",') and not raw_line.lstrip().startswith("ID,"):
                continue
            header = next(csv.reader([raw_line]))
            reader = csv.DictReader(handle, fieldnames=header)
            for row in reader:
                if not row.get("ID"):
                    continue
                metric_name = row.get("Metric Name") or row.get("Metric") or ""
                if not metric_name:
                    continue
                rows.append(row)
            break
    return rows


def summarize_ncu(csv_path: Path) -> dict[str, Any]:
    rows = load_ncu_rows(csv_path)
    by_kernel: dict[str, dict[str, Any]] = {}
    for row in rows:
        kernel = row.get("Kernel Name") or row.get("Kernel") or "unknown_kernel"
        metric = row.get("Metric Name") or "unknown_metric"
        unit = row.get("Metric Unit") or ""
        value = normalize_metric_value(row.get("Metric Value", ""))
        entry = by_kernel.setdefault(kernel, {"kernel_name": kernel, "metrics": {}})
        entry["metrics"][metric] = {"value": value, "unit": unit}

    tc_metric_names = sorted({
        row.get("Metric Name", "")
        for row in rows
        if any(token in row.get("Metric Name", "").lower() for token in TENSOR_CORE_METRIC_PATTERNS)
    })
    pct_metric_names = [name for name in tc_metric_names if "pct_of_peak" in name]

    def values_for(metric_name: str) -> list[tuple[float, float | None]]:
        values: list[tuple[float, float | None]] = []
        for kernel in by_kernel.values():
            metric = kernel["metrics"].get(metric_name)
            if not metric:
                continue
            value = metric["value"]
            if not isinstance(value, float):
                continue
            time_metric = kernel["metrics"].get("gpu__time_duration.sum")
            time_value = time_metric["value"] if time_metric else None
            values.append((value, time_value if isinstance(time_value, float) else None))
        return values

    tensor_core_summary: dict[str, Any] = {}
    for metric_name in pct_metric_names:
        values = values_for(metric_name)
        if not values:
            continue
        numeric = [value for value, _ in values]
        weighted_pairs = [(value, weight) for value, weight in values if weight and weight > 0]
        if weighted_pairs:
            weighted = sum(value * weight for value, weight in weighted_pairs) / sum(weight for _, weight in weighted_pairs)
        else:
            weighted = None
        tensor_core_summary[metric_name] = {
            "max_pct": max(numeric),
            "avg_pct": sum(numeric) / len(numeric),
            "time_weighted_pct": weighted,
        }

    selected_tc = None
    for preferred in DEFAULT_NCU_METRICS:
        if preferred in tensor_core_summary:
            item = tensor_core_summary[preferred]
            selected_tc = item["time_weighted_pct"] if item["time_weighted_pct"] is not None else item["max_pct"]
            break
    if selected_tc is None and tensor_core_summary:
        first = next(iter(tensor_core_summary.values()))
        selected_tc = first["time_weighted_pct"] if first["time_weighted_pct"] is not None else first["max_pct"]

    return {
        "ncu_metric_rows": len(rows),
        "ncu_kernel_count": len(by_kernel),
        "ncu_tensor_core_metric_names": tc_metric_names,
        "ncu_tensor_core_util_pct": selected_tc,
        "ncu_tensor_core_summary": tensor_core_summary,
        "ncu_kernels": list(by_kernel.values()),
    }


def query_available_metrics(ncu: str) -> set[str]:
    commands = ([ncu, "--query-metrics"], [ncu, "--query-metrics", "--query-metrics-mode", "all"])
    for command in commands:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            continue
        return set(re.findall(r"\b[a-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)*\b", completed.stdout))
    return set()


def resolve_metrics(ncu: str, requested: list[str], should_query: bool) -> tuple[list[str], list[str]]:
    if not should_query:
        return requested, []
    available = query_available_metrics(ncu)
    if not available:
        return requested, ["Could not query NCU metrics; using requested metrics unchanged."]
    selected = [metric for metric in requested if metric in available]
    warnings: list[str] = []
    missing = [metric for metric in requested if metric not in available]
    if missing:
        warnings.append(f"NCU metrics unavailable and skipped: {', '.join(missing)}")
    if not any(any(token in metric.lower() for token in TENSOR_CORE_METRIC_PATTERNS) for metric in selected):
        candidates = sorted(
            metric for metric in available
            if "pct_of_peak" in metric and any(token in metric.lower() for token in TENSOR_CORE_METRIC_PATTERNS)
        )
        if candidates:
            selected.append(candidates[0])
            warnings.append(f"Added available Tensor Core-like metric: {candidates[0]}")
    if not selected:
        return requested, warnings + ["No requested metrics matched NCU query; using requested metrics unchanged so NCU reports the exact error."]
    return selected, warnings
