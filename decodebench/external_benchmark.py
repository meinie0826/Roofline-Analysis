#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an external DecodeBench-compatible benchmark command template.")
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--layer", default="framework_reference")
    parser.add_argument("--command-template", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--head-dim-v", type=int, required=True)
    parser.add_argument("--qk-nope-head-dim", type=int, required=True)
    parser.add_argument("--kv-lora-rank", type=int, required=True)
    parser.add_argument("--qk-rope-head-dim", type=int, required=True)
    parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def attention_name(num_q_heads: int, num_kv_heads: int) -> str:
    if num_q_heads == num_kv_heads:
        return "MHA"
    if num_kv_heads == 1:
        return "MQA"
    return "GQA"


def load_external_result(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"external command did not create result JSON: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if "compare_latency_us" not in row and "kernel_latency_p50_us" not in row:
        raise RuntimeError(f"external result JSON must contain compare_latency_us or kernel_latency_p50_us: {path}")
    return row


def main() -> int:
    args = parse_args()
    if args.command_template.startswith("__missing_env_"):
        env_name = args.command_template.removeprefix("__missing_env_").removesuffix("__")
        raise RuntimeError(f"set {env_name} to an external benchmark command template before running {args.backend_name}")

    external_output = args.output.with_suffix(".external.json")
    values = {
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "head_dim_v": args.head_dim_v,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "kv_dtype": args.kv_dtype,
        "page_size": args.page_size,
        "warmup_steps": args.warmup_steps,
        "repeat": args.repeat,
        "output": str(external_output),
    }
    command = args.command_template.format(**values)
    completed = subprocess.run(shlex.split(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or f"external benchmark failed: {completed.returncode}")

    external = load_external_result(external_output)
    latency = external.get("compare_latency_us") or external.get("kernel_latency_p50_us")
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": external.get("gpu"),
        "backend": args.backend_name,
        "kernel_path": args.kernel_path,
        "layer": args.layer,
        "workload_id": os.environ.get("DECODEBENCH_WORKLOAD_ID"),
        "attention": attention_name(args.num_q_heads, args.num_kv_heads),
        "kv_dtype": args.kv_dtype,
        "page_size": args.page_size,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "decode_steps": 1,
        "compare_latency_us": latency,
        "kernel_latency_avg_us": external.get("kernel_latency_avg_us", latency),
        "kernel_latency_p50_us": external.get("kernel_latency_p50_us", latency),
        "kernel_latency_p95_us": external.get("kernel_latency_p95_us"),
        "approx_kv_bytes_read": external.get("approx_kv_bytes_read"),
        "approx_effective_kv_bandwidth_gb_s": external.get("approx_effective_kv_bandwidth_gb_s"),
        "peak_allocated_gb": external.get("peak_allocated_gb"),
        "selected_backend": external.get("selected_backend", args.backend_name),
        "fallback": bool(external.get("fallback", False)),
        "fallback_reason": external.get("fallback_reason"),
        "command": command,
        "stdout": completed.stdout,
        "notes": "External benchmark adapter. The command template must write DecodeBench-compatible JSON to {output}.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
