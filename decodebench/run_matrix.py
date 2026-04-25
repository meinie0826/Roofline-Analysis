#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PYTHON_BIN = sys.executable or "python3"
REPO_ROOT = ROOT.parent

from summarize_results import print_summary
from ncu_utils import DEFAULT_NCU_METRICS, resolve_metrics, summarize_ncu


def shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


DEFAULT_BENCH_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRTLLM_BENCH_PYTHON = REPO_ROOT / ".venv-trtllm" / "bin" / "python"

# Keep interpreter selection inside the benchmark runner so mixed environments
# are reproducible without shell exports. Most backends use the main bench venv;
# TensorRT-LLM native uses its own venv because its precompiled extension is
# tied to a different PyTorch ABI.
BACKEND_PYTHON_PATHS = {
    "flashinfer_paged_decode": DEFAULT_BENCH_PYTHON,
    "flashinfer_trtllm_decode": DEFAULT_BENCH_PYTHON,
    "flashattn_kvcache": DEFAULT_BENCH_PYTHON,
    "flashmla_decode": DEFAULT_BENCH_PYTHON,
    "flashattn_mla_decode": DEFAULT_BENCH_PYTHON,
    "flashinfer_trtllm_mla_decode": DEFAULT_BENCH_PYTHON,
    "vllm_paged_decode": DEFAULT_BENCH_PYTHON,
    "vllm_flash": DEFAULT_BENCH_PYTHON,
    "vllm_flashinfer": DEFAULT_BENCH_PYTHON,
    "torch_sdpa_auto": DEFAULT_BENCH_PYTHON,
    "torch_sdpa_cudnn": DEFAULT_BENCH_PYTHON,
    "torch_sdpa_flash": DEFAULT_BENCH_PYTHON,
    "sglang_serving": DEFAULT_BENCH_PYTHON,
    "tensorrt_llm_native": TRTLLM_BENCH_PYTHON,
}


def python_for_backend(backend: dict) -> str:
    configured = backend.get("python")
    if configured:
        return str(configured)
    candidate = BACKEND_PYTHON_PATHS.get(backend["name"], DEFAULT_BENCH_PYTHON)
    return str(candidate if candidate.exists() else PYTHON_BIN)


def apply_backend_python(argv: list[str], backend: dict) -> list[str]:
    if not argv:
        return argv
    return [python_for_backend(backend), *argv[1:]]


def flashinfer_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/flashinfer_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def flashinfer_trtllm_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/trtllm_decode_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def flashattn_kvcache_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/flashattn_kvcache_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def flashmla_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/flashmla_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--head-dim-v", str(workload["head_dim_v"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def flashattn_mla_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/flashattn_mla_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--head-dim-v", str(workload["head_dim_v"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def trtllm_mla_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/trtllm_mla_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload.get("num_kv_heads", 1)),
        "--qk-nope-head-dim", str(workload.get("qk_nope_head_dim", 128)),
        "--kv-lora-rank", str(workload.get("kv_lora_rank", workload.get("head_dim_v", 512))),
        "--qk-rope-head-dim", str(workload.get("qk_rope_head_dim", workload.get("head_dim", 64))),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def vllm_attention_cmd(workload: dict, defaults: dict, output: Path, backend: str) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/vllm_attention_benchmark.py",
        "--backend", backend,
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def vllm_paged_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/vllm_paged_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def torch_sdpa_cmd(workload: dict, defaults: dict, output: Path, backend: str) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/torch_sdpa_benchmark.py",
        "--backend", backend,
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def external_cmd(backend: dict, workload: dict, defaults: dict, output: Path) -> list[str]:
    template_key = backend["command_template_env"]
    command_template = os.environ.get(template_key, backend.get("command_template", ""))
    if not command_template:
        command_template = f"__missing_env_{template_key}__"
    return [
        PYTHON_BIN,
        "decodebench/external_benchmark.py",
        "--backend-name", backend["name"],
        "--kernel-path", backend.get("kernel_path", backend["name"]),
        "--layer", backend.get("layer", "framework_reference"),
        "--command-template", command_template,
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--head-dim-v", str(workload.get("head_dim_v", workload["head_dim"])),
        "--qk-nope-head-dim", str(workload.get("qk_nope_head_dim", 128)),
        "--kv-lora-rank", str(workload.get("kv_lora_rank", workload.get("head_dim_v", workload["head_dim"]))),
        "--qk-rope-head-dim", str(workload.get("qk_rope_head_dim", workload["head_dim"])),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def trtllm_native_cmd(workload: dict, defaults: dict, output: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "decodebench/trtllm_native_benchmark.py",
        "--batch-size", str(workload["batch_size"]),
        "--context-len", str(workload["context_len"]),
        "--num-q-heads", str(workload["num_q_heads"]),
        "--num-kv-heads", str(workload["num_kv_heads"]),
        "--head-dim", str(workload["head_dim"]),
        "--kv-dtype", workload["kv_dtype"],
        "--page-size", str(workload["page_size"]),
        "--warmup-steps", str(defaults.get("warmup_steps", 10)),
        "--repeat", str(defaults.get("repeat", 50)),
        "--output", str(output),
    ]


def build_cmd(backend: dict, workload: dict, defaults: dict, results_dir: Path) -> list[str]:
    output = results_dir / f'{backend["name"]}__{workload["id"]}.json'
    if backend["name"] == "flashinfer_paged_decode":
        return apply_backend_python(flashinfer_cmd(workload, defaults, output), backend)
    if backend["name"] == "flashinfer_trtllm_decode":
        return apply_backend_python(flashinfer_trtllm_cmd(workload, defaults, output), backend)
    if backend["name"] == "flashattn_kvcache":
        return apply_backend_python(flashattn_kvcache_cmd(workload, defaults, output), backend)
    if backend["name"] == "flashmla_decode":
        return apply_backend_python(flashmla_cmd(workload, defaults, output), backend)
    if backend["name"] == "flashattn_mla_decode":
        return apply_backend_python(flashattn_mla_cmd(workload, defaults, output), backend)
    if backend["name"] == "flashinfer_trtllm_mla_decode":
        return apply_backend_python(trtllm_mla_cmd(workload, defaults, output), backend)
    if backend["name"] == "vllm_paged_decode":
        return apply_backend_python(vllm_paged_cmd(workload, defaults, output), backend)
    if backend["name"] == "vllm_flash":
        return apply_backend_python(vllm_attention_cmd(workload, defaults, output, "flash"), backend)
    if backend["name"] == "vllm_flashinfer":
        return apply_backend_python(vllm_attention_cmd(workload, defaults, output, "flashinfer"), backend)
    if backend["name"] == "torch_sdpa_auto":
        return apply_backend_python(torch_sdpa_cmd(workload, defaults, output, "auto"), backend)
    if backend["name"] == "torch_sdpa_cudnn":
        return apply_backend_python(torch_sdpa_cmd(workload, defaults, output, "cudnn"), backend)
    if backend["name"] == "torch_sdpa_flash":
        return apply_backend_python(torch_sdpa_cmd(workload, defaults, output, "flash"), backend)
    if backend["name"] == "tensorrt_llm_native":
        return apply_backend_python(trtllm_native_cmd(workload, defaults, output), backend)
    if backend["name"] == "sglang_serving":
        return apply_backend_python(external_cmd(backend, workload, defaults, output), backend)
    raise ValueError(f'Backend not implemented yet: {backend["name"]}')


def is_supported(backend: dict, workload: dict) -> bool:
    supported_attention = backend.get("supported_attention")
    if supported_attention is not None and workload["attention"] not in supported_attention:
        return False
    supported_kv_dtypes = backend.get("supported_kv_dtypes")
    if supported_kv_dtypes is not None and workload["kv_dtype"] not in supported_kv_dtypes:
        return False
    supported_page_sizes = backend.get("supported_page_sizes")
    if supported_page_sizes is not None and workload["page_size"] not in supported_page_sizes:
        return False
    supported_workload_ids = backend.get("supported_workload_ids")
    if supported_workload_ids is not None and workload["id"] not in supported_workload_ids:
        return False
    return True


def output_path(backend: dict, workload: dict, results_dir: Path) -> Path:
    return results_dir / f'{backend["name"]}__{workload["id"]}.json'


def short_backend(name: str) -> str:
    return {
        "flashinfer_paged_decode": "flashinfer",
        "flashinfer_trtllm_decode": "trtllm",
        "flashattn_kvcache": "flash-attn",
        "flashmla_decode": "flashmla",
        "flashattn_mla_decode": "flash-attn-mla",
        "flashinfer_trtllm_mla_decode": "trtllm-mla",
        "vllm_paged_decode": "vllm-paged",
        "vllm_flash": "vllm-flash",
        "vllm_flashinfer": "vllm-flashinfer",
        "torch_sdpa_auto": "torch-sdpa",
        "torch_sdpa_cudnn": "torch-cudnn",
        "torch_sdpa_flash": "torch-flash",
        "tensorrt_llm_native": "trtllm-native",
        "sglang_serving": "sglang",
    }.get(name, name)


def safe_profile_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in text)


def ncu_csv_path(profile_dir: Path, backend: dict, workload: dict) -> Path:
    return profile_dir / f'{safe_profile_name(backend["name"])}__{safe_profile_name(workload["id"])}.ncu.csv'


def wrap_ncu_cmd(
    argv: list[str],
    csv_path: Path,
    ncu_bin: str,
    metrics: list[str],
    sections: list[str],
    launch_skip: int,
    launch_count: int,
    kernel_name: str | None,
) -> list[str]:
    wrapped = [
        ncu_bin,
        "--target-processes", "all",
        "--csv",
        "--page", "raw",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--log-file", str(csv_path),
    ]
    for section in sections:
        wrapped += ["--section", section]
    if metrics:
        wrapped += ["--metrics", ",".join(metrics)]
    if kernel_name:
        wrapped += ["--kernel-name", kernel_name]
    return wrapped + argv


def merge_ncu_result(result_path: Path, csv_path: Path, command: list[str], metrics: list[str], warnings: list[str]) -> None:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = {}
    result.update(summarize_ncu(csv_path))
    result["ncu_profiled"] = True
    result["ncu_csv"] = str(csv_path)
    result["ncu_command"] = shell(command)
    result["ncu_metrics_requested"] = metrics
    result["ncu_metric_warnings"] = warnings
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_failure(path: Path, backend: dict, workload: dict, returncode: int, command: list[str], output: str) -> None:
    short_reason = ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if any(token in line for token in ("Error:", "RuntimeError:", "ImportError:", "ModuleNotFoundError:", "FileNotFoundError:", "ValueError:", "Unknown backend:")):
            short_reason = line
            break
    result = {
        "run_id": os.environ.get("DECODEBENCH_RUN_ID"),
        "status": "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_executable": command[0] if command else PYTHON_BIN,
        "backend": backend["name"],
        "kernel_path": backend.get("kernel_path"),
        "layer": backend.get("layer"),
        "workload_id": workload["id"],
        "attention": workload["attention"],
        "kv_dtype": workload["kv_dtype"],
        "page_size": workload["page_size"],
        "batch_size": workload["batch_size"],
        "context_len": workload["context_len"],
        "returncode": returncode,
        "command": shell(command),
        "output": output,
        "short_reason": short_reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("decodebench_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load matrix file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def short_failure_reason(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if any(token in line for token in ("Error:", "RuntimeError:", "ImportError:", "ModuleNotFoundError:", "FileNotFoundError:", "ValueError:", "Unknown backend:")):
            return line
    return lines[-1] if lines else "unknown error"


def result_has_ncu_profile(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return row.get("ncu_profiled") is True and isinstance(row.get("ncu_tensor_core_summary"), dict)


def result_is_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if row.get("status") == "failed" or row.get("fallback"):
        return False
    value = row.get("compare_latency_us") or row.get("kernel_latency_p50_us")
    return isinstance(value, (int, float)) and value > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run decode attention kernel benchmark matrix.")
    parser.add_argument("--config", type=Path, default=ROOT / "matrix_b200.py")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-backend", default="flashinfer_paged_decode")
    parser.add_argument("--resume", action="store_true", help="Skip backend/workload pairs with an existing successful result JSON.")
    parser.add_argument("--profile-ncu", action="store_true", help="Wrap each benchmark with Nsight Compute and merge Tensor Core metrics into the normal result JSON.")
    parser.add_argument("--ncu", default="ncu", help="Nsight Compute CLI path used with --profile-ncu.")
    parser.add_argument("--ncu-profile-dir", type=Path, default=None, help="Directory for raw NCU CSV files. Defaults to <results-dir>/ncu.")
    parser.add_argument("--ncu-metric", action="append", dest="ncu_metrics", help="NCU metric to collect. May be repeated. Defaults include TC/SM/DRAM/time metrics.")
    parser.add_argument("--ncu-section", action="append", dest="ncu_sections", default=[], help="Optional NCU section. May be repeated.")
    parser.add_argument("--ncu-kernel-name", default=None, help="Optional NCU --kernel-name regex filter.")
    parser.add_argument("--ncu-launch-skip", type=int, default=0)
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    parser.add_argument("--ncu-query-metrics", action=argparse.BooleanOptionalAction, default=True, help="Query NCU and keep only available default metrics before profiling.")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        parser.error("choose exactly one: --dry-run or --execute")

    config = load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ncu_metrics: list[str] = []
    ncu_metric_warnings: list[str] = []
    ncu_profile_dir = args.ncu_profile_dir or (args.results_dir / "ncu")
    if args.profile_ncu:
        ncu_profile_dir.mkdir(parents=True, exist_ok=True)
        ncu_metrics, ncu_metric_warnings = resolve_metrics(args.ncu, args.ncu_metrics or DEFAULT_NCU_METRICS, args.ncu_query_metrics)
        for warning in ncu_metric_warnings:
            print(f"# {warning}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures = 0
    executed_paths: list[Path] = []
    skipped = 0

    for backend in config["backends"]:
        if not backend.get("enabled", True):
            continue
        for workload in config["workloads"]:
            if not is_supported(backend, workload):
                continue
            path = output_path(backend, workload, args.results_dir)
            executed_paths.append(path)
            argv = build_cmd(backend, workload, config.get("defaults", {}), args.results_dir)
            run_argv = argv
            csv_path = ncu_csv_path(ncu_profile_dir, backend, workload)
            if args.profile_ncu:
                run_argv = wrap_ncu_cmd(
                    argv,
                    csv_path,
                    args.ncu,
                    ncu_metrics,
                    args.ncu_sections,
                    args.ncu_launch_skip,
                    args.ncu_launch_count,
                    args.ncu_kernel_name,
                )
            if args.resume and result_is_success(path) and (not args.profile_ncu or result_has_ncu_profile(path)):
                skipped += 1
                if args.dry_run:
                    print(f"# skip existing success: {short_backend(backend['name'])} {workload['id']}")
                else:
                    print(f"↷ {workload['id']:<28} {short_backend(backend['name'])}  | existing result")
                continue
            if args.dry_run:
                print(shell(run_argv))
            else:
                env = os.environ.copy()
                env["DECODEBENCH_RUN_ID"] = run_id
                env["DECODEBENCH_WORKLOAD_ID"] = workload["id"]
                completed = subprocess.run(run_argv, check=False, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if args.verbose and completed.stdout:
                    print(completed.stdout, end="")
                if completed.returncode != 0:
                    failures += 1
                    write_failure(path, backend, workload, completed.returncode, run_argv, completed.stdout)
                    print(f"✗ {workload['id']:<28} {short_backend(backend['name'])}  | {short_failure_reason(completed.stdout)}")
                else:
                    if args.profile_ncu:
                        if csv_path.exists():
                            merge_ncu_result(path, csv_path, run_argv, ncu_metrics, ncu_metric_warnings)
                            try:
                                tc_util = json.loads(path.read_text(encoding="utf-8")).get("ncu_tensor_core_util_pct")
                            except (OSError, json.JSONDecodeError):
                                tc_util = None
                            print(f"✓ {workload['id']:<28} {short_backend(backend['name'])}  | TC {tc_util}")
                        else:
                            failures += 1
                            write_failure(path, backend, workload, completed.returncode, run_argv, completed.stdout + "\nNCU CSV was not produced.")
                            print(f"✗ {workload['id']:<28} {short_backend(backend['name'])}  | NCU CSV was not produced")
                    else:
                        print(f"✓ {workload['id']:<28} {short_backend(backend['name'])}")

    if args.resume and skipped:
        print(f"Skipped existing successful results: {skipped}")

    if args.execute and args.report:
        print()
        rows = []
        for path in executed_paths:
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if not args.resume and row.get("run_id") not in (None, run_id):
                continue
            rows.append(row)
        if rows:
            print_summary(rows)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
