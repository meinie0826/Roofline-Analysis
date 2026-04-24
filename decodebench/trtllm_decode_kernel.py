from __future__ import annotations

import math
from dataclasses import dataclass


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


class FlashInferTRTLLMDecodeKernel:
    def __init__(self, shape: DecodeShape):
        import torch

        try:
            import flashinfer
        except ImportError as error:
            raise ImportError(
                "FlashInfer is not installed. Install it with: "
                "python3 -m pip install flashinfer-python flashinfer-cubin"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.torch = torch
        self.flashinfer = flashinfer
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16 if shape.kv_dtype == "bf16" else torch.float16
        self.kv_dtype = dtype_from_name(torch, shape.kv_dtype)
        self.sm_scale = float(1.0 / (shape.head_dim**0.5))
        self.q_scale = 1.0
        self.k_scale = 1.0
        self.v_scale = 1.0
        self.o_scale = 1.0

        self.query = torch.randn(
            shape.batch_size,
            shape.num_q_heads,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.seq_lens = torch.full((shape.batch_size,), shape.context_len, dtype=torch.int32, device=self.device)
        self.max_seq_len = shape.context_len
        self.block_tables = self._make_block_tables()
        self.workspace_buffer = torch.zeros(1024 * 1024 * 1024, dtype=torch.int8, device=self.device)
        self.kv_cache = self._make_kv_cache()
        self.output = torch.empty(self.query.shape, dtype=self.dtype, device=self.device)

    def _make_block_tables(self):
        torch = self.torch
        shape = self.shape
        num_blocks = math.ceil(shape.context_len / shape.page_size)
        return torch.arange(
            0,
            shape.batch_size * num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).reshape(shape.batch_size, num_blocks)

    def _make_kv_cache(self):
        torch = self.torch
        shape = self.shape
        num_blocks = self.block_tables.numel()
        cache_shape = (num_blocks, 2, shape.num_kv_heads, shape.page_size, shape.head_dim)
        if self.kv_dtype == torch.float8_e4m3fn:
            return torch.randn(*cache_shape, dtype=torch.float16, device=self.device).to(self.kv_dtype)
        return torch.randn(*cache_shape, dtype=self.kv_dtype, device=self.device)

    def run(self):
        return self.flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=self.query,
            kv_cache=self.kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            max_seq_len=self.max_seq_len,
            bmm1_scale=self.q_scale * self.k_scale * self.sm_scale,
            bmm2_scale=self.v_scale / self.o_scale,
            out=self.output,
        )
