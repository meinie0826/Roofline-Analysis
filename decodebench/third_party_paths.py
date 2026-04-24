from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ROOT = REPO_ROOT / "3rd"


def _existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_vllm_benchmark_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()

    env_path = os.environ.get("VLLM_BENCH_DIR")
    if env_path:
        return Path(env_path).expanduser()

    vllm_root = THIRD_PARTY_ROOT / "vllm"
    candidates = [
        vllm_root / "benchmarks" / "attention_benchmarks",
        vllm_root / "benchmarks",
    ]
    path = _existing_path(candidates)
    if path is not None:
        benchmark_py = path / "benchmark.py"
        if benchmark_py.exists():
            return path

    if vllm_root.exists():
        for benchmark_py in sorted(vllm_root.rglob("benchmark.py")):
            try:
                text = benchmark_py.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            markers = ("--backend", "--batch-specs", "--output-json")
            if all(marker in text for marker in markers):
                return benchmark_py.parent

    return candidates[0]


def find_vllm_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit

    env_python = os.environ.get("VLLM_BENCH_PYTHON")
    if env_python:
        return env_python

    candidates = [THIRD_PARTY_ROOT / "vllm" / ".venv" / "bin" / "python"]
    path = _existing_path(candidates)
    return str(path) if path is not None else sys.executable


def find_flash_attention_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()

    env_path = os.environ.get("FLASH_ATTN_SRC_DIR")
    if env_path:
        return Path(env_path).expanduser()

    return THIRD_PARTY_ROOT / "flash-attention"
