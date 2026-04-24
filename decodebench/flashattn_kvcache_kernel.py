from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from third_party_paths import find_flash_attention_root


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
        dtype_bytes = 2
        return self.batch_size * self.context_len * self.num_kv_heads * self.head_dim * 2 * dtype_bytes


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"flash-attention kvcache supports bf16/fp16, got: {name}")


class FlashAttnKVCacheKernel:
    def __init__(self, shape: DecodeShape):
        import torch

        flash_attn_root = find_flash_attention_root()

        flash_attn_callable = None
        flash_attn_api: Literal["fa4_varlen", "fa3_kvcache", "fa2_kvcache"]
        cache_table_arg: Literal["page_table", "block_table"] = "block_table"
        kernel_path = "flash_attn.flash_attn_with_kvcache"
        try:
            from flash_attn.cute.interface import flash_attn_varlen_func as fa4_flash_attn_varlen_func
        except ImportError:
            if str(flash_attn_root) not in sys.path:
                sys.path.insert(0, str(flash_attn_root))
            try:
                from flash_attn.cute.interface import flash_attn_varlen_func as fa4_flash_attn_varlen_func
            except ImportError:
                fa4_flash_attn_varlen_func = None

        if fa4_flash_attn_varlen_func is not None:
            flash_attn_callable = fa4_flash_attn_varlen_func
            flash_attn_api = "fa4_varlen"
            cache_table_arg = "page_table"
            kernel_path = "flash_attn.cute.interface.flash_attn_varlen_func"

        if str(flash_attn_root) not in sys.path:
            sys.path.insert(0, str(flash_attn_root))

        if flash_attn_callable is None:
            try:
                from hopper.flash_attn_interface import flash_attn_with_kvcache as hopper_flash_attn_with_kvcache

                flash_attn_callable = hopper_flash_attn_with_kvcache
                flash_attn_api = "fa3_kvcache"
                cache_table_arg = "page_table"
                kernel_path = "hopper.flash_attn_interface.flash_attn_with_kvcache"
            except ImportError:
                pass

        if flash_attn_callable is None:
            try:
                from flash_attn import flash_attn_with_kvcache as fa2_flash_attn_with_kvcache

                flash_attn_callable = fa2_flash_attn_with_kvcache
                flash_attn_api = "fa2_kvcache"
            except ImportError as error:
                raise ImportError(
                    "flash-attention is not importable. Install/build it first, or set "
                    "FLASH_ATTN_SRC_DIR to a built flash-attention checkout."
                ) from error

        if flash_attn_api == "fa2_kvcache" and shape.page_size % 256 != 0:
            raise ImportError(
                "flash-attention FA2 paged KV cache requires page_size to be a multiple of 256. "
                "Install/use FA4 or the Hopper interface for arbitrary page sizes such as 64."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.torch = torch
        self.flash_attn_callable = flash_attn_callable
        self.flash_attn_api = flash_attn_api
        self.cache_table_arg = cache_table_arg
        self.kernel_path = kernel_path
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(torch, shape.kv_dtype)
        self.pages_per_seq = (shape.context_len + shape.page_size - 1) // shape.page_size
        self.num_blocks = shape.batch_size * self.pages_per_seq

        self.q = torch.randn(
            shape.batch_size,
            1,
            shape.num_q_heads,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.k_cache = torch.randn(
            self.num_blocks,
            shape.page_size,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache = torch.randn(
            self.num_blocks,
            shape.page_size,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.cache_table = torch.arange(
            self.num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).reshape(shape.batch_size, self.pages_per_seq)
        self.cache_seqlens = torch.full(
            (shape.batch_size,),
            shape.context_len,
            dtype=torch.int32,
            device=self.device,
        )

    def run(self):
        if self.flash_attn_api == "fa4_varlen":
            out, _ = self.flash_attn_callable(
                q=self.q,
                k=self.k_cache,
                v=self.v_cache,
                seqused_k=self.cache_seqlens,
                page_table=self.cache_table,
                causal=True,
            )
            return out
        kwargs = {
            "q": self.q,
            "k_cache": self.k_cache,
            "v_cache": self.v_cache,
            "cache_seqlens": self.cache_seqlens,
            "causal": True,
            self.cache_table_arg: self.cache_table,
        }
        return self.flash_attn_callable(**kwargs)
