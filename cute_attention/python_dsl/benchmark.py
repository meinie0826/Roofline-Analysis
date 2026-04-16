#!/usr/bin/env python3
"""
Benchmark runner for cute_attention stages.

Default shape: (batch=1, heads=16, seqlen=2048, headdim=128, dtype=float16)
These match a typical LLM decode-phase / prefill configuration and stay within
the seq-len limits of all CuTe stages (max 4096).

Each stage is benchmarked with its most appropriate default tuning config:
  - stage12/stage13 → dedicated autotune enabled
  - stage16/stage17 → dedicated autotune enabled
  - non-autotuned stages use generic tile search by default to reduce
    tile-size bias in comparisons; pass --no-generic-tile-autotune to disable
  - stage15/stage16/stage17 → num_threads always forced to 256 by the kernel
  - stage14                 → num_threads defaults to 256 in benchmark mode
  - stage0-stage11          → num_threads=128 by default
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from kernels import (
    AttentionConfig,
    autotune_stage12_config,
    autotune_stage13_config,
    autotune_stage16_config,
    autotune_stage17_config,
    autotune_stage18_config,
    autotune_stage19_config,
    autotune_stage20_config,
    autotune_stage21_config,
    autotune_stage22_config,
    available_backends,
    run_stage,
)


torch = None
if available_backends()["torch"]:
    import torch


# ---------------------------------------------------------------------------
# Stages that benefit from a higher default num_threads (producer/consumer)
# ---------------------------------------------------------------------------
_WARPSPEC_STAGES = {"stage14", "stage15", "stage16", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22"}

_DEDICATED_AUTOTUNE_STAGES = {"stage12", "stage13", "stage16", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22"}
_MULTISTAGE_STAGES = {"stage12", "stage13", "stage16", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22"}
_GENERIC_TILE_TUNABLE_STAGES = {
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
    "stage8",
    "stage9",
    "stage10",
    "stage11",
    "stage14",
    "stage15",
}
_THREAD_SEARCH_STAGES = {"stage14"}
_STAGE_TUNING_AXES = {
    "stage0": "none",
    "stage1": "benchmark fallback only",
    "stage2": "benchmark fallback only",
    "stage3": "benchmark fallback only",
    "stage4": "benchmark fallback only",
    "stage5": "benchmark fallback only",
    "stage6": "benchmark fallback only",
    "stage7": "benchmark fallback only",
    "stage8": "benchmark fallback only",
    "stage9": "benchmark fallback only",
    "stage10": "benchmark fallback only",
    "stage11": "benchmark fallback only",
    "stage12": "block_m,block_n,num_stages_kv",
    "stage13": "block_m,block_n,num_stages_kv",
    "stage14": "benchmark fallback only",
    "stage15": "benchmark fallback only",
    "stage16": "block_m,block_n",
    "stage17": "block_m,block_n,num_stages_kv",
    "stage18": "block_m,block_n,num_stages_kv",
    "stage19": "block_m,block_n,num_stages_kv",
    "stage20": "block_m,block_n,num_stages_kv",
    "stage21": "block_m,block_n,num_stages_kv",
    "stage22": "block_m,block_n,num_stages_kv",
    "baseline_fa4": "none",
    "baseline_sdpa": "none",
}
_STAGE_NOTES = {
    "stage12": "independent two-stage cp.async pipeline kernel; autotunes block sizes within its own stage-2 design",
    "stage13": "independent MMA multistage kernel; autotunes tile sizes and num_stages_kv",
    "stage2": "column-blocked reference CuTe kernel; benchmark uses generic tile fallback only",
    "stage14": "warp-specialized producer/consumer kernel; no dedicated autotune yet",
    "stage15": "SM90-style warp specialization; no dedicated autotune yet",
    "stage16": "fixed double-buffer warp-specialized kernel; current autotune is conservative block search only",
    "stage17": "independent warp-specialized multistage kernel; fixed 256-thread producer/consumer schedule with autotuned tiles and stage depth",
    "stage18": "SM90-oriented experimental backend; independent multistage path with stage-state mainloop, stage-aware cp.async waits, and broader tile search",
    "stage19": "warpgroup-layout experimental backend; independent multistage path that swaps to Hopper-style warpgroup shared-memory layout atoms",
    "stage20": "aggressive warpspec experimental backend; circular-buffer steady-state mainloop with full-slot prefetch and dedicated multistage autotune",
    "stage21": "explicit producer/consumer state-machine backend; stage18-derived mainline with dedicated prefetch, advance, and wait helpers",
    "stage22": "independent Blackwell tcgen05+TMA backend under active bring-up; stage22 keeps its own autotune and compile-cache plumbing",
}


def _candidate_values(preferred: int, values: list[int], *, limit: int) -> list[int]:
    ordered = []
    for value in [preferred, *values]:
        if value <= 0 or value > limit or value in ordered:
            continue
        ordered.append(value)
    return ordered


def get_stage_metadata() -> list[dict[str, str]]:
    rows = []
    for stage_name in parse_stage_list("all"):
        rows.append(
            {
                "stage": stage_name,
                "autotune": str(stage_name in _DEDICATED_AUTOTUNE_STAGES),
                "multistage": str(stage_name in _MULTISTAGE_STAGES),
                "tuning_axes": _STAGE_TUNING_AXES.get(stage_name, "unknown"),
                "notes": _STAGE_NOTES.get(stage_name, ""),
            }
        )
    return rows


def _estimate_attention_flops(shape: tuple[int, int, int, int], causal: bool) -> float:
    batch, heads, seqlen, head_dim = shape
    if causal:
        pair_count = seqlen * (seqlen + 1) / 2.0
    else:
        pair_count = float(seqlen * seqlen)
    # QK and PV GEMMs only, counted as multiply-add FLOPs.
    return 4.0 * batch * heads * head_dim * pair_count


def _estimate_tflops(time_ms: float, shape: tuple[int, int, int, int], causal: bool) -> float:
    if time_ms <= 0:
        return 0.0
    return _estimate_attention_flops(shape, causal) / (time_ms / 1000.0) / 1e12


def _thread_candidates(stage_name: str, preferred: int) -> list[int]:
    if stage_name not in _THREAD_SEARCH_STAGES:
        return [preferred]
    ordered = []
    for value in [preferred, 256, 128]:
        if value <= 0 or value in ordered:
            continue
        ordered.append(value)
    return ordered


def benchmark(stage_name, q, k, v, config, warmup=5, repeat=20):
    for _ in range(warmup):
        run_stage(stage_name, q, k, v, config)

    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_stage(stage_name, q, k, v, config)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


def _config_status_suffix(config: AttentionConfig) -> str | None:
    parts = []
    if config.block_m:
        parts.append(f"block_m={config.block_m}")
    if config.block_n:
        parts.append(f"block_n={config.block_n}")
    if config.num_threads:
        parts.append(f"num_threads={config.num_threads}")
    if config.num_stages_kv:
        parts.append(f"stages={config.num_stages_kv}")
    return ",".join(parts) if parts else None


def _make_config_for_stage(stage_name: str, base: AttentionConfig) -> AttentionConfig:
    """Return a config tailored to the given stage from the base config."""
    if stage_name == "stage22":
        return replace(base, block_m=128, block_n=128, num_threads=256, num_stages_kv=3)
    if stage_name in {"stage17", "stage18", "stage19", "stage20", "stage21"}:
        return replace(base, block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    if stage_name in _WARPSPEC_STAGES:
        return replace(base, num_threads=256)
    return base


def _benchmark_with_generic_tile_search(stage_name, q, k, v, config, warmup=5, repeat=20):
    seq_len = q.shape[2]
    block_m_values = _candidate_values(config.block_m, [128, 96, 64, 48, 32, 16], limit=seq_len)
    block_n_values = _candidate_values(config.block_n, [256, 192, 128, 96, 64, 32], limit=seq_len)
    best_time = None
    best_config = None
    last_exc = None

    for num_threads in _thread_candidates(stage_name, config.num_threads):
        for block_m in block_m_values:
            for block_n in block_n_values:
                trial = replace(config, block_m=block_m, block_n=block_n, num_threads=num_threads)
                try:
                    elapsed = benchmark(stage_name, q, k, v, trial, warmup=warmup, repeat=repeat)
                except ValueError as exc:
                    last_exc = exc
                    continue
                if best_time is None or elapsed < best_time:
                    best_time = elapsed
                    best_config = trial

    if best_time is None or best_config is None:
        if last_exc is not None:
            raise last_exc
        raise ValueError(f"{stage_name} failed all generic tile-search candidates.")
    return best_time, _config_status_suffix(best_config)


def benchmark_stage_with_fallback(stage_name, q, k, v, config, warmup=5, repeat=20, generic_tile_autotune=False):
    """Benchmark a stage, applying autotune or block_m fallback as appropriate."""
    if stage_name == "stage12":
        tuned = autotune_stage12_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage13":
        tuned = autotune_stage13_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage16":
        tuned = autotune_stage16_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage17":
        tuned = autotune_stage17_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage18":
        tuned = autotune_stage18_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage19":
        tuned = autotune_stage19_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage20":
        tuned = autotune_stage20_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage21":
        tuned = autotune_stage21_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)
    if stage_name == "stage22":
        tuned = autotune_stage22_config(q, k, v, config)
        return benchmark(stage_name, q, k, v, tuned, warmup=warmup, repeat=repeat), _config_status_suffix(tuned)

    if generic_tile_autotune and stage_name in _GENERIC_TILE_TUNABLE_STAGES:
        return _benchmark_with_generic_tile_search(stage_name, q, k, v, config, warmup=warmup, repeat=repeat)

    if stage_name not in _GENERIC_TILE_TUNABLE_STAGES:
        return benchmark(stage_name, q, k, v, config, warmup=warmup, repeat=repeat), None

    block_m = config.block_m
    while block_m >= 1:
        cfg = replace(config, block_m=block_m)
        try:
            t = benchmark(stage_name, q, k, v, cfg, warmup=warmup, repeat=repeat)
            return t, _config_status_suffix(cfg) if block_m != config.block_m else None
        except ValueError as exc:
            if "shared memory footprint too large" not in str(exc):
                raise
            block_m //= 2
    raise ValueError(f"{stage_name} failed all fallback block_m candidates for shared memory.")


def parse_stage_list(stages_arg: str) -> list[str]:
    if stages_arg == "all":
        return [
            "stage0",
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
            "stage6",
            "stage7",
            "stage8",
            "stage9",
            "stage10",
            "stage11",
            "stage12",
            "stage13",
            "stage14",
            "stage15",
            "stage16",
            "stage17",
            "stage18",
            "stage19",
            "stage20",
            "stage21",
            "stage22",
            "baseline_fa4",
            "baseline_sdpa",
        ]
    return [name.strip() for name in stages_arg.split(",") if name.strip()]


def plot_results(
    results: list[dict],
    shape: tuple,
    dtype: str,
    causal: bool,
    device_name: str,
    report_tensorcore: bool,
    output_dir: str = "result",
) -> str:
    """Save a bar chart of benchmark results to output_dir. Returns the saved file path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("# matplotlib not available, skipping plot", file=sys.stderr)
        return ""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ok_results = [r for r in results if r["time_ms"] is not None and r["time_ms"] > 0]
    if not ok_results:
        return ""

    labels = [r["stage"] for r in ok_results]
    metric = "tflops" if report_tensorcore else "time_ms"
    values = [r.get("tflops_est", 0.0) if report_tensorcore else r["time_ms"] for r in ok_results]
    ylabel = "Estimated TFLOP/s" if report_tensorcore else "Latency (ms)"
    title_metric = "TFLOP/s" if report_tensorcore else "Latency"

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))

    bars = ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, rotation=45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Attention Benchmark — {title_metric}\n"
        f"shape={shape}  dtype={dtype}  causal={causal}\n"
        f"device: {device_name}",
        fontsize=9,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shape_str = "x".join(str(s) for s in shape)
    fname = f"bench_{shape_str}_{dtype}_{ts}.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stage names, or 'all' (default).",
    )
    parser.add_argument("--batch",       type=int,   default=1,
                        help="Batch size (default: 1).")
    parser.add_argument("--heads",       type=int,   default=16,
                        help="Number of attention heads (default: 16).")
    parser.add_argument("--seqlen",      type=int,   default=2048,
                        help="Sequence length (default: 2048; max 4096 for CuTe stages).")
    parser.add_argument("--headdim",     type=int,   default=128,
                        help="Head dimension (default: 128).")
    parser.add_argument("--dtype",       default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Data type (default: float16; CuTe stages require float16).")
    parser.add_argument("--block-m",     type=int,   default=64,
                        help="Q block size (default: 64). Overridden by autotune for stage12/13/16/17.")
    parser.add_argument("--block-n",     type=int,   default=128,
                        help="KV block size (default: 128). Overridden by autotune for stage12/13/16/17.")
    parser.add_argument("--num-threads", type=int,   default=128,
                        help="Thread count (default: 128). Forced to 256 for stage14/15/16/17.")
    parser.add_argument("--causal",      action="store_true", default=True,
                        help="Use causal masking (default: True).")
    parser.add_argument("--warmup",      type=int,   default=5,
                        help="Warmup iterations (default: 5).")
    parser.add_argument("--repeat",      type=int,   default=20,
                        help="Benchmark iterations (default: 20).")
    parser.add_argument("--generic-tile-autotune", action=argparse.BooleanOptionalAction, default=True,
                        help="For stages without dedicated autotune, search a small set of block_m/block_n candidates (default: enabled).")
    parser.add_argument("--report-tensorcore", action="store_true",
                        help="Add estimated attention TFLOP/s and optional Tensor Core utilization columns.")
    parser.add_argument("--tensorcore-peak-tflops", type=float, default=None,
                        help="Peak Tensor Core TFLOP/s for utilization estimates. Used only with --report-tensorcore.")
    parser.add_argument("--print-stage-metadata", action="store_true",
                        help="Print autotune/multistage coverage for each stage before running benchmarks.")
    parser.add_argument("--result-dir", default="result",
                        help="Directory to save benchmark plots (default: result).")
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = {
        "float16":  torch.float16,
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    q = torch.randn(args.batch, args.heads, args.seqlen, args.headdim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    base_config = AttentionConfig(
        causal=args.causal,
        block_m=args.block_m,
        block_n=args.block_n,
        num_threads=args.num_threads,
    )
    stages = parse_stage_list(args.stages)

    device_name = torch.cuda.get_device_name(0)
    print({
        "device":      device_name,
        "shape":       tuple(q.shape),
        "dtype":       args.dtype,
        "causal":      args.causal,
        "block_m":     args.block_m,
        "block_n":     args.block_n,
        "num_threads": args.num_threads,
        "warmup":      args.warmup,
        "repeat":      args.repeat,
        "generic_tile_autotune": args.generic_tile_autotune,
        "report_tensorcore": args.report_tensorcore,
        "tensorcore_peak_tflops": args.tensorcore_peak_tflops,
        "stages":      stages,
    })
    if args.print_stage_metadata:
        print("stage_metadata")
        metadata_writer = csv.writer(sys.stdout)
        metadata_writer.writerow(["stage", "autotune", "multistage", "tuning_axes", "notes"])
        for row in get_stage_metadata():
            if row["stage"] in stages:
                metadata_writer.writerow(
                    [row["stage"], row["autotune"], row["multistage"], row["tuning_axes"], row["notes"]]
                )
    if args.report_tensorcore:
        print("stage,time_ms,tflops_est,tc_util_pct,status")
    else:
        print("stage,time_ms,status")
    _plot_results: list[dict] = []
    for stage_name in stages:
        config = _make_config_for_stage(stage_name, base_config)
        try:
            time_ms, status_suffix = benchmark_stage_with_fallback(
                stage_name,
                q,
                k,
                v,
                config,
                warmup=args.warmup,
                repeat=args.repeat,
                generic_tile_autotune=args.generic_tile_autotune,
            )
            if args.report_tensorcore:
                tflops_est = _estimate_tflops(time_ms, tuple(q.shape), args.causal)
                tc_util_pct = "na"
                if args.tensorcore_peak_tflops:
                    tc_util_pct = f"{100.0 * tflops_est / args.tensorcore_peak_tflops:.2f}"
                status = f"ok:{status_suffix}" if status_suffix else "ok"
                print(f"{stage_name},{time_ms:.3f},{tflops_est:.3f},{tc_util_pct},{status}")
                _plot_results.append({"stage": stage_name, "time_ms": time_ms, "tflops_est": tflops_est})
            elif status_suffix:
                print(f"{stage_name},{time_ms:.3f},ok:{status_suffix}")
                _plot_results.append({"stage": stage_name, "time_ms": time_ms})
            else:
                print(f"{stage_name},{time_ms:.3f},ok")
                _plot_results.append({"stage": stage_name, "time_ms": time_ms})
        except Exception as exc:
            if args.report_tensorcore:
                print(f"{stage_name},nan,nan,na,failed:{type(exc).__name__}:{exc}")
            else:
                print(f"{stage_name},nan,failed:{type(exc).__name__}:{exc}")
            _plot_results.append({"stage": stage_name, "time_ms": None})

    saved = plot_results(
        _plot_results,
        shape=tuple(q.shape),
        dtype=args.dtype,
        causal=args.causal,
        device_name=device_name,
        report_tensorcore=args.report_tensorcore,
        output_dir=args.result_dir,
    )
    if saved:
        print(f"# plot saved to {saved}", file=sys.stderr)


if __name__ == "__main__":
    main()
