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


class FlashInferPagedDecodeKernel:
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
        self.q_dtype = torch.bfloat16 if shape.kv_dtype == "bf16" else torch.float16
        self.kv_dtype = dtype_from_name(torch, shape.kv_dtype)
        self.pages_per_seq = math.ceil(shape.context_len / shape.page_size)
        self.total_pages = shape.batch_size * self.pages_per_seq

        self.q = torch.randn(
            shape.batch_size,
            shape.num_q_heads,
            shape.head_dim,
            dtype=self.q_dtype,
            device=self.device,
        )
        self.k_cache, self.v_cache = self._make_kv_cache()
        self.wrapper = self._make_wrapper()

    def _make_kv_cache(self):
        torch = self.torch
        shape = self.shape
        cache_shape = (self.total_pages, shape.page_size, shape.num_kv_heads, shape.head_dim)
        if self.kv_dtype == torch.float8_e4m3fn:
            k_cache = torch.randn(*cache_shape, dtype=torch.float16, device=self.device).to(self.kv_dtype)
            v_cache = torch.randn(*cache_shape, dtype=torch.float16, device=self.device).to(self.kv_dtype)
            return k_cache, v_cache
        return (
            torch.randn(*cache_shape, dtype=self.kv_dtype, device=self.device),
            torch.randn(*cache_shape, dtype=self.kv_dtype, device=self.device),
        )

    def _make_wrapper(self):
        torch = self.torch
        shape = self.shape
        kv_indptr = torch.arange(0, self.total_pages + 1, self.pages_per_seq, dtype=torch.int32, device=self.device)
        kv_indices = torch.arange(self.total_pages, dtype=torch.int32, device=self.device)
        last_page_len = torch.full(
            (shape.batch_size,),
            shape.context_len % shape.page_size or shape.page_size,
            dtype=torch.int32,
            device=self.device,
        )
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        wrapper = self.flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            shape.num_q_heads,
            shape.num_kv_heads,
            shape.head_dim,
            shape.page_size,
            data_type=self.q_dtype,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
        )
        return wrapper

    def run(self):
        return self.wrapper.run(self.q, (self.k_cache, self.v_cache), return_lse=False)
