from __future__ import annotations

import sys
from dataclasses import dataclass

from third_party_paths import find_flash_attention_root


@dataclass(frozen=True)
class MLADecodeShape:
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


class FlashMLAKernel:
    def __init__(self, shape: MLADecodeShape):
        import torch

        flash_attn_root = find_flash_attention_root()
        hopper_dir = flash_attn_root / "hopper"
        for path in (flash_attn_root, hopper_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        except ImportError as error:
            raise ImportError(
                "FlashMLA is not importable. Build/install flash_mla first, then rerun the MLA suite."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if shape.page_size != 64:
            raise ValueError("FlashMLA suite currently only supports page_size=64")

        self.torch = torch
        self.flash_mla_with_kvcache = flash_mla_with_kvcache
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16 if shape.kv_dtype == "bf16" else torch.float16

        self.q = torch.randn(shape.batch_size, 1, shape.num_q_heads, shape.head_dim, dtype=self.dtype, device=self.device)
        self.qv = torch.randn(shape.batch_size, 1, shape.num_q_heads, shape.head_dim_v, dtype=self.dtype, device=self.device)
        self.cache_seqlens = torch.full((shape.batch_size,), shape.context_len, dtype=torch.int32, device=self.device)
        self.k_cache = torch.randn(shape.batch_size, shape.context_len, shape.num_kv_heads, shape.head_dim, dtype=self.dtype, device=self.device)
        self.v_cache = torch.randn(shape.batch_size, shape.context_len, shape.num_kv_heads, shape.head_dim_v, dtype=self.dtype, device=self.device)

        num_pages_per_seq = shape.context_len // shape.page_size
        self.k_cache = self.k_cache.reshape(shape.batch_size * num_pages_per_seq, shape.page_size, shape.num_kv_heads, shape.head_dim)
        self.v_cache = self.v_cache.reshape(shape.batch_size * num_pages_per_seq, shape.page_size, shape.num_kv_heads, shape.head_dim_v)
        self.page_table = torch.arange(shape.batch_size * num_pages_per_seq, dtype=torch.int32, device=self.device).reshape(shape.batch_size, num_pages_per_seq)

        self.q_concat = torch.concat([self.q, self.qv], dim=-1)
        self.kv_cache_concat = torch.concat([self.v_cache, self.k_cache], dim=-1)
        self.mla_metadata = get_mla_metadata(
            self.cache_seqlens,
            shape.num_q_heads // shape.num_kv_heads,
            shape.num_kv_heads,
        )

    def run(self):
        return self.flash_mla_with_kvcache(
            self.q_concat,
            self.kv_cache_concat,
            self.page_table,
            self.cache_seqlens,
            self.shape.head_dim_v,
            *self.mla_metadata,
            causal=True,
        )
