#!/usr/bin/env python3
"""Plot WS/SW K-sweep results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

STYLE = {
    "sw_stage6": {
        "label": "SW stage 6",
        "color": "#7fb6d6",
        "marker": "o",
        "linestyle": "-",
    },
    "sw_stage7": {
        "label": "SW stage 7",
        "color": "#2f79b7",
        "marker": "s",
        "linestyle": "-",
    },
    "ws_stage6": {
        "label": "WS stage 6",
        "color": "#d68a3a",
        "marker": "^",
        "linestyle": "-",
    },
    "ws_stage7": {
        "label": "WS stage 7",
        "color": "#8f5fbf",
        "marker": "D",
        "linestyle": "-",
    },
    "ws_stage6_regular_store": {
        "label": "WS stage 6 regular",
        "color": "#c66c33",
        "marker": "v",
        "linestyle": "--",
    },
    "ws_stage7_regular_store": {
        "label": "WS stage 7 regular",
        "color": "#6c4fa3",
        "marker": "P",
        "linestyle": "--",
    },
}


def latest_csv() -> Path:
    files = sorted(RESULTS_DIR.glob("k_sweep_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No k_sweep_*.csv files under {RESULTS_DIR}")
    return files[-1]


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("returncode", "")) != "0":
                continue
            if not row.get("avg_ms") or not row.get("tflops"):
                continue
            row = dict(row)
            row["k"] = int(row["k"])
            row["avg_ms"] = float(row["avg_ms"])
            row["tflops"] = float(row["tflops"])
            row["repeat"] = int(row["repeat"])
            row["ab_stages"] = int(row["ab_stages"])
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["schedule"], row["ab_stages"], row["k"])].append(row)

    summary = []
    for (variant, schedule, ab_stages, k), items in grouped.items():
        ms = [item["avg_ms"] for item in items]
        tflops = [item["tflops"] for item in items]
        summary.append(
            {
                "variant": variant,
                "schedule": schedule,
                "ab_stages": ab_stages,
                "k": k,
                "k_tiles": k / 64.0,
                "avg_ms": mean(ms),
                "std_ms": stdev(ms),
                "tflops": mean(tflops),
                "count": len(items),
            }
        )
    return sorted(summary, key=lambda r: (r["variant"], r["k"]))


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return intercept, slope


def variant_points(summary: list[dict]) -> dict[str, list[dict]]:
    by_variant = defaultdict(list)
    for row in summary:
        by_variant[row["variant"]].append(row)
    return {
        variant: sorted(points, key=lambda r: r["k"])
        for variant, points in by_variant.items()
    }


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "font.family": "DejaVu Sans",
        }
    )
    return plt


def draw_plot(summary: list[dict], output: Path, title_prefix: str) -> None:
    plt = setup_matplotlib()
    variants = variant_points(summary)
    ordered_variants = [v for v in STYLE if v in variants]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), dpi=180)
    axes = axes.ravel()

    ax = axes[0]
    for variant in ordered_variants:
        points = variants[variant]
        style = STYLE[variant]
        ax.errorbar(
            [p["k"] for p in points],
            [p["avg_ms"] for p in points],
            yerr=[p["std_ms"] for p in points],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6.0,
            capsize=3,
        )
    ax.set_title(f"{title_prefix}: Latency")
    ax.set_xlabel("K")
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("log", base=2)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False)

    ax = axes[1]
    for variant in ordered_variants:
        points = variants[variant]
        style = STYLE[variant]
        ax.plot(
            [p["k"] for p in points],
            [p["tflops"] for p in points],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6.0,
        )
    ax.set_title(f"{title_prefix}: Throughput")
    ax.set_xlabel("K")
    ax.set_ylabel("TFLOP/s")
    ax.set_xscale("log", base=2)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False)

    baseline = {p["k"]: p["avg_ms"] for p in variants.get("sw_stage7", [])}
    ax = axes[2]
    for variant in ordered_variants:
        if variant == "sw_stage7" or not baseline:
            continue
        points = [p for p in variants[variant] if p["k"] in baseline]
        style = STYLE[variant]
        ax.plot(
            [p["k"] for p in points],
            [baseline[p["k"]] / p["avg_ms"] for p in points],
            label=f"{style['label']} vs SW stage 7",
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6.0,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3, label="Parity")
    ax.set_title(f"{title_prefix}: Speedup")
    ax.set_xlabel("K")
    ax.set_ylabel("Speedup over SW stage 7")
    ax.set_xscale("log", base=2)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False)

    ax = axes[3]
    x_positions = list(range(len(ordered_variants)))
    intercepts = []
    slopes = []
    labels = []
    colors = []
    for variant in ordered_variants:
        points = variants[variant]
        intercept, slope = fit_line([(p["k_tiles"], p["avg_ms"]) for p in points])
        intercepts.append(intercept * 1000.0)
        slopes.append(slope * 1000.0)
        labels.append(STYLE[variant]["label"])
        colors.append(STYLE[variant]["color"])
    width = 0.36
    ax.bar(
        [x - width / 2 for x in x_positions],
        intercepts,
        width=width,
        label="Intercept A",
        color=colors,
        alpha=0.85,
        edgecolor="#444444",
        linewidth=1.0,
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        slopes,
        width=width,
        label="Slope B",
        color=colors,
        alpha=0.45,
        edgecolor="#444444",
        linewidth=1.0,
        hatch="//",
    )
    ax.set_title(f"{title_prefix}: Linear Fit")
    ax.set_ylabel("Time (us)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False)

    fig.tight_layout(pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot WS/SW K-sweep CSV results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Defaults to latest results/k_sweep_*.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG. Defaults to results/<input-stem>.png.",
    )
    parser.add_argument("--title", default="Blackwell GEMM WS vs SW")
    args = parser.parse_args()

    input_path = args.input or latest_csv()
    output_path = args.output or input_path.with_suffix(".png")
    summary = summarize(read_rows(input_path))
    if not summary:
        raise RuntimeError(f"No successful benchmark rows in {input_path}")
    draw_plot(summary, output_path, args.title)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
