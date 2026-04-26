#!/usr/bin/env python3
"""Summarize benchmark_matrix_sglang CSVs.

The matrix benchmark measures end-to-end layer-shaped paths.  This helper fits a
simple line over seq_len for each (hidden_dim, num_heads, cluster_size) group:

    latency_ms ~= fixed_ms + slope_us_per_token * seq_len / 1000

The intercept is a useful proxy for projection/RMS/output fixed work, while the
slope is a proxy for decode-attention work over KV length.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return mean_y, 0.0
    slope_ms_per_token = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept_ms = mean_y - slope_ms_per_token * mean_x
    return intercept_ms, slope_ms_per_token * 1000.0


def analyze(path: Path) -> int:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            key = (row["hidden_dim"], row["num_heads"], row["cluster_size"])
            groups[key].append(row)

    if not groups:
        raise RuntimeError(f"No ok rows found in {path}")

    print(f"File: {path}")
    print(
        "D,H,C | points | cute_fixed_ms | cute_slope_us/token | "
        "tc_fixed_ms | tc_slope_us/token | persist_layer_fixed_ms | "
        "persist_layer_slope_us/token | subgraph_fixed_ms | subgraph_slope_us/token | "
        "cute/persist_layer@maxS | tc/persist_layer@maxS | rel_l2@maxS"
    )
    print("-" * 128)

    for key in sorted(groups, key=lambda item: tuple(int(x) for x in item)):
        rows = sorted(groups[key], key=lambda row: int(row["seq_len"]))
        seq_lens = [float(row["seq_len"]) for row in rows]
        cute_ms = [_float(row, "cute_megakernel_ms") for row in rows]
        tc_ms = [_float(row, "tc_megakernel_ms") for row in rows]
        persistent_layer_ms = [_float(row, "sglang_layer_persistent_ms") for row in rows]
        subgraph_ms = [_float(row, "sglang_subgraph_ref_ms") for row in rows]
        if any(value is None for value in cute_ms):
            continue

        cute_fixed, cute_slope = _fit_line(seq_lens, [value for value in cute_ms if value is not None])

        tc_text = "n/a | n/a"
        if not any(value is None for value in tc_ms):
            tc_fixed, tc_slope = _fit_line(seq_lens, [value for value in tc_ms if value is not None])
            tc_text = f"{tc_fixed:.4f} | {tc_slope:.4f}"

        subgraph_text = "n/a | n/a"
        persistent_layer_text = "n/a | n/a"
        max_row = rows[-1]
        max_cute = _float(max_row, "cute_megakernel_ms")
        max_tc = _float(max_row, "tc_megakernel_ms")
        max_persistent_layer = _float(max_row, "sglang_layer_persistent_ms")
        max_subgraph = _float(max_row, "sglang_subgraph_ref_ms")
        ratio_text = "n/a"
        if max_cute is not None and max_persistent_layer is not None:
            ratio_text = f"{max_cute / max_persistent_layer:.3f}x"
        tc_ratio_text = "n/a"
        if max_tc is not None and max_persistent_layer is not None:
            tc_ratio_text = f"{max_tc / max_persistent_layer:.3f}x"
        if not any(value is None for value in persistent_layer_ms):
            persistent_fixed, persistent_slope = _fit_line(
                seq_lens, [value for value in persistent_layer_ms if value is not None]
            )
            persistent_layer_text = f"{persistent_fixed:.4f} | {persistent_slope:.4f}"
        if not any(value is None for value in subgraph_ms):
            sub_fixed, sub_slope = _fit_line(
                seq_lens, [value for value in subgraph_ms if value is not None]
            )
            subgraph_text = f"{sub_fixed:.4f} | {sub_slope:.4f}"

        rel_l2 = _float(max_row, "output_rel_l2_vs_sglang_layer")
        rel_l2_text = "n/a" if rel_l2 is None else f"{rel_l2:.6g}"
        d, h, c = key
        print(
            f"{d},{h},{c} | {len(rows)} | {cute_fixed:.4f} | {cute_slope:.4f} | "
            f"{tc_text} | {persistent_layer_text} | {subgraph_text} | "
            f"{ratio_text} | {tc_ratio_text} | {rel_l2_text}"
        )

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark_matrix_sglang CSV output.")
    parser.add_argument("csv", type=Path, help="CSV produced by benchmark_matrix_sglang.py.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return analyze(args.csv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
