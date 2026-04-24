from __future__ import annotations

import math
import sys
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

    @property
    def kv_bytes(self) -> int:
        dtype_bytes = 1 if self.kv_dtype == "fp8" else 2
        return self.batch_size * self.context_len * self.num_kv_heads * self.head_dim * 2 * dtype_bytes


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp8", "fp8_e4m3", "float8_e4m3fn"}:
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported kv dtype: {name}")


class VLLMPagedDecodeKernel:
    PARTITION_SIZE = 512

    def __init__(self, shape: DecodeShape):
        import torch

        repo_root = Path(__file__).resolve().parent.parent
        vllm_root = repo_root / "3rd" / "vllm"
        if vllm_root.exists() and str(vllm_root) not in sys.path:
            sys.path.insert(0, str(vllm_root))

        try:
            from vllm import _custom_ops as ops
        except ImportError as error:
            raise ImportError(
                "vLLM is not installed. Install it as a package or editable source under 3rd/vllm."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        if shape.kv_dtype == "fp8":
            raise ValueError(
                "Direct vLLM paged_attention benchmark currently supports bf16/fp16 only. "
                "vLLM fp8 decode needs extra cache-scale plumbing and is not wired here yet."
            )

        self.torch = torch
        self.ops = ops
        self.shape = shape
        self.device = torch.device("cuda")
        self.q_dtype = torch.bfloat16 if shape.kv_dtype == "bf16" else torch.float16
        self.kv_dtype = dtype_from_name(torch, shape.kv_dtype)
        self.kv_cache_dtype = "auto"
        self.pages_per_seq = math.ceil(shape.context_len / shape.page_size)
        self.num_blocks = shape.batch_size * self.pages_per_seq
        self.max_seq_len = shape.context_len
        self.scale = float(1.0 / (shape.head_dim**0.5))

        self.query = torch.randn(
            shape.batch_size,
            shape.num_q_heads,
            shape.head_dim,
            dtype=self.q_dtype,
            device=self.device,
        )
        self.key_cache, self.value_cache = self._make_kv_cache()
        self.block_tables = self._make_block_tables()
        self.seq_lens = torch.full(
            (shape.batch_size,),
            shape.context_len,
            dtype=torch.int32,
            device=self.device,
        )
        self._populate_cache()
        self.output, self.exp_sums, self.max_logits, self.tmp_out = self._make_outputs()
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

    def _make_kv_cache(self):
        torch = self.torch
        shape = self.shape
        x = 16 // torch.tensor([], dtype=self.kv_dtype).element_size()
        key_cache = torch.empty(
            self.num_blocks,
            shape.num_kv_heads,
            shape.head_dim // x,
            shape.page_size,
            x,
            dtype=self.kv_dtype,
            device=self.device,
        )
        value_cache = torch.empty(
            self.num_blocks,
            shape.num_kv_heads,
            shape.head_dim,
            shape.page_size,
            dtype=self.kv_dtype,
            device=self.device,
        )
        return key_cache, value_cache

    def _make_block_tables(self):
        torch = self.torch
        return torch.arange(
            self.num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).reshape(self.shape.batch_size, self.pages_per_seq)

    def _populate_cache(self):
        torch = self.torch
        shape = self.shape

        key = torch.randn(
            shape.batch_size,
            shape.context_len,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=self.kv_dtype,
            device=self.device,
        )
        value = torch.randn(
            shape.batch_size,
            shape.context_len,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=self.kv_dtype,
            device=self.device,
        )

        token_slots = []
        for batch_idx in range(shape.batch_size):
            base_block = batch_idx * self.pages_per_seq
            for token_idx in range(shape.context_len):
                block_idx = base_block + token_idx // shape.page_size
                offset = token_idx % shape.page_size
                token_slots.append(block_idx * shape.page_size + offset)
        slot_mapping = torch.tensor(token_slots, dtype=torch.int64, device=self.device)

        self.ops.reshape_and_cache(
            key.view(-1, shape.num_kv_heads, shape.head_dim),
            value.view(-1, shape.num_kv_heads, shape.head_dim),
            self.key_cache,
            self.value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            self.k_scale,
            self.v_scale,
        )

    def _make_outputs(self):
        torch = self.torch
        shape = self.shape
        num_partitions = math.ceil(shape.context_len / self.PARTITION_SIZE)
        output = torch.empty(
            shape.batch_size,
            shape.num_q_heads,
            shape.head_dim,
            dtype=self.q_dtype,
            device=self.device,
        )
        exp_sums = torch.empty(
            shape.batch_size,
            shape.num_q_heads,
            num_partitions,
            dtype=torch.float32,
            device=self.device,
        )
        max_logits = torch.empty_like(exp_sums)
        tmp_out = torch.empty(
            shape.batch_size,
            shape.num_q_heads,
            num_partitions,
            shape.head_dim,
            dtype=self.q_dtype,
            device=self.device,
        )
        return output, exp_sums, max_logits, tmp_out

    def run(self):
        self.ops.paged_attention_v2(
            self.output,
            self.exp_sums,
            self.max_logits,
            self.tmp_out,
            self.query,
            self.key_cache,
            self.value_cache,
            self.shape.num_kv_heads,
            self.scale,
            self.block_tables,
            self.seq_lens,
            self.shape.page_size,
            self.max_seq_len,
            None,
            self.kv_cache_dtype,
            self.k_scale,
            self.v_scale,
        )
        return self.output
