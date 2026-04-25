#!/usr/bin/env python3
"""Plot Figure-5-style DSMEM cluster results for one or more GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    label = path.stem
    for row in rows:
        if row.get("metric") == "metadata":
            label = row.get("gpu_label") or row.get("gpu_name") or label
            if label == "auto":
                label = row.get("gpu_name") or path.stem
            break
    return label, rows


def by_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row.get("metric") == metric and row.get("supported", True)],
        key=lambda row: row.get("cluster_size", 0),
    )


def metric_scalar(rows: list[dict[str, Any]], metric: str, key: str) -> float | None:
    values = by_metric(rows, metric)
    if not values:
        return None
    value = values[0].get(key)
    return float(value) if value is not None else None


def plot_one(axs, label: str, rows: list[dict[str, Any]]) -> None:
    latency = by_metric(rows, "dsmem_latency")
    bandwidth = by_metric(rows, "dsmem_bandwidth")
    active = by_metric(rows, "active_sm")
    global_latency = metric_scalar(rows, "global_latency", "cycles_per_load_median")
    global_bandwidth = metric_scalar(rows, "global_bandwidth", "bandwidth_tb_s_median")
    local_smem_bandwidth = metric_scalar(rows, "local_smem_bandwidth", "bandwidth_tb_s_median")

    x_lat = [row["cluster_size"] for row in latency]
    y_lat = [row["cycles_per_load_median"] for row in latency]
    axs[0].bar([str(x) for x in x_lat], y_lat, color="#9ecae1", edgecolor="#4d4d4d")
    if global_latency is not None:
        axs[0].axhline(global_latency, linestyle="--", color="black", linewidth=1.2, label="Global Memory")
        axs[0].legend(frameon=False, fontsize=9)
    axs[0].set_title(f"{label}: Latency")
    axs[0].set_xlabel("Cluster Size")
    axs[0].set_ylabel("Latency (cycles)")

    x_bw = [row["cluster_size"] for row in bandwidth]
    y_bw = [row["bandwidth_tb_s_median"] for row in bandwidth]
    axs[1].bar([str(x) for x in x_bw], y_bw, color="#9ecae1", edgecolor="#4d4d4d")
    if global_bandwidth is not None:
        axs[1].axhline(global_bandwidth, linestyle="--", color="black", linewidth=1.2, label="Global Memory")
    if local_smem_bandwidth is not None:
        axs[1].axhline(local_smem_bandwidth, linestyle=":", color="#555555", linewidth=1.2, label="Local SMEM")
    if global_bandwidth is not None or local_smem_bandwidth is not None:
        axs[1].legend(frameon=False, fontsize=9)
    axs[1].set_title(f"{label}: Bandwidth")
    axs[1].set_xlabel("Cluster Size")
    axs[1].set_ylabel("Bandwidth (TB/s)")

    x_sm = [row["cluster_size"] for row in active]
    y_sm = [row["active_sms_estimate"] for row in active]
    axs[2].bar([str(x) for x in x_sm], y_sm, color="#9ecae1", edgecolor="#4d4d4d")
    axs[2].set_title(f"{label}: Active SM")
    axs[2].set_xlabel("Cluster Size")
    axs[2].set_ylabel("Active SM")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Result JSON files from run_dsmem_cluster.py.")
    parser.add_argument("--output", type=Path, default=Path("figures/dsmem_cluster.png"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with: python3 -m pip install matplotlib") from exc

    datasets = [load_rows(path) for path in args.results]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(11.0, 3.0 * len(datasets)), squeeze=False)
    for row_idx, (label, rows) in enumerate(datasets):
        plot_one(axes[row_idx], label, rows)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
