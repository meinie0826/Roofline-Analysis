from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from third_party_paths import find_flash_attention_root


@dataclass(frozen=True)
class FlashAttnMLADecodeShape:
    batch_size: int
    context_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_v: int
    kv_dtype: str
    page_size: int

    @property
    def kv_bytes(self) -> int:
        dtype_bytes = 2
        return self.batch_size * self.context_len * self.num_kv_heads * (self.head_dim + self.head_dim_v) * dtype_bytes


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"FlashAttention MLA benchmark supports bf16/fp16, got: {name}")


class FlashAttnMLAKernel:
    def __init__(self, shape: FlashAttnMLADecodeShape):
        import torch

        flash_attn_root = find_flash_attention_root()
        hopper_dir = flash_attn_root / "hopper"
        for path in (flash_attn_root, hopper_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
        except ImportError as error:
            raise ImportError(
                "Tri Dao flash-attention Hopper MLA path is not importable. Build/install flash-attention hopper interface first."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if shape.page_size != 64:
            raise ValueError("FlashAttention MLA benchmark currently uses page_size=64")

        self.torch = torch
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(torch, shape.kv_dtype)
        self.pages_per_seq = math.ceil(shape.context_len / shape.page_size)
        self.num_pages = shape.batch_size * self.pages_per_seq

        self.q = torch.randn(shape.batch_size, 1, shape.num_q_heads, shape.head_dim, dtype=self.dtype, device=self.device)
        self.qv = torch.randn(shape.batch_size, 1, shape.num_q_heads, shape.head_dim_v, dtype=self.dtype, device=self.device)
        self.k_cache = torch.randn(self.num_pages, shape.page_size, shape.num_kv_heads, shape.head_dim, dtype=self.dtype, device=self.device)
        self.v_cache = torch.randn(self.num_pages, shape.page_size, shape.num_kv_heads, shape.head_dim_v, dtype=self.dtype, device=self.device)
        self.page_table = torch.arange(self.num_pages, dtype=torch.int32, device=self.device).reshape(shape.batch_size, self.pages_per_seq)
        self.cache_seqlens = torch.full((shape.batch_size,), shape.context_len, dtype=torch.int32, device=self.device)
        self.scheduler_metadata = get_scheduler_metadata(
            shape.batch_size,
            1,
            shape.context_len,
            shape.num_q_heads,
            shape.num_kv_heads,
            shape.head_dim,
            self.cache_seqlens,
            self.dtype,
            head_dim_v=shape.head_dim_v,
            page_size=shape.page_size,
            causal=True,
        )

    def run(self):
        return self.flash_attn_with_kvcache(
            self.q,
            self.k_cache,
            self.v_cache,
            cache_seqlens=self.cache_seqlens,
            num_splits=0,
            qv=self.qv,
            page_table=self.page_table,
            causal=True,
            scheduler_metadata=self.scheduler_metadata,
        )
