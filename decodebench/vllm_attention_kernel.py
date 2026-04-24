from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecodeShape:
    batch_size: int
    context_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    kv_dtype: str
    page_size: int

    @property
    def attention(self) -> str:
        if self.num_q_heads == self.num_kv_heads:
            return "MHA"
        if self.num_kv_heads == 1:
            return "MQA"
        return "GQA"


def compact_k(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def make_batch_spec(shape: DecodeShape) -> str:
    return f"{shape.batch_size}q1s{compact_k(shape.context_len)}"


class VLLMAttentionBenchmark:
    def __init__(self, backend: str, shape: DecodeShape, benchmark_dir: Path, python_bin: str):
        self.backend = backend
        self.shape = shape
        self.benchmark_dir = benchmark_dir
        self.python_bin = python_bin

    def command(self, output_json: Path, repeats: int, warmup_steps: int) -> list[str]:
        backend_name = {
            "flash": "FLASH_ATTN",
            "flashinfer": "FLASHINFER",
        }.get(self.backend, self.backend)
        kv_cache_dtype = "fp8" if self.shape.kv_dtype == "fp8" else "auto"
        return [
            self.python_bin,
            "benchmark.py",
            "--backend",
            backend_name,
            "--batch-specs",
            make_batch_spec(self.shape),
            "--num-layers",
            "1",
            "--head-dim",
            str(self.shape.head_dim),
            "--num-q-heads",
            str(self.shape.num_q_heads),
            "--num-kv-heads",
            str(self.shape.num_kv_heads),
            "--block-size",
            str(self.shape.page_size),
            "--repeats",
            str(repeats),
            "--warmup-iters",
            str(warmup_steps),
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--output-json",
            str(output_json),
        ]

    def run(self, repeats: int, warmup_steps: int) -> dict:
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(
                "vLLM benchmark dir not found. Expected benchmark source under "
                f"{self.benchmark_dir}. Put the vLLM repo under 3rd/vllm or set "
                "VLLM_BENCH_DIR explicitly."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "vllm_attention_results.json"
            cmd = self.command(output_json, repeats, warmup_steps)
            completed = subprocess.run(
                cmd,
                cwd=self.benchmark_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout.strip() or f"vLLM benchmark failed: {completed.returncode}")
            rows = json.loads(output_json.read_text(encoding="utf-8"))
            if not rows:
                raise RuntimeError("vLLM benchmark produced empty JSON output")
            return rows[0]
