from __future__ import annotations

import os
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

    candidates = [THIRD_PARTY_ROOT / "vllm" / "benchmarks" / "attention_benchmarks"]
    path = _existing_path(candidates)
    return path if path is not None else candidates[0]


def find_vllm_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit

    env_python = os.environ.get("VLLM_BENCH_PYTHON")
    if env_python:
        return env_python

    candidates = [THIRD_PARTY_ROOT / "vllm" / ".venv" / "bin" / "python"]
    path = _existing_path(candidates)
    return str(path) if path is not None else "python3"
