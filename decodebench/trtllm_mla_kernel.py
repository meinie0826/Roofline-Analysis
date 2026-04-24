from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TRTLLMMLADecodeShape:
    batch_size: int
    context_len: int
    num_q_heads: int
    num_kv_heads: int
    qk_nope_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    kv_dtype: str
    page_size: int

    @property
    def head_dim_qk(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_bytes(self) -> int:
        dtype_bytes = 2
        return self.batch_size * self.context_len * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_bytes


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"TRTLLM MLA benchmark supports bf16/fp16, got: {name}")


class FlashInferTRTLLMMLAKernel:
    def __init__(self, shape: TRTLLMMLADecodeShape):
        import torch

        try:
            import flashinfer
        except ImportError as error:
            raise ImportError(
                "FlashInfer is not installed. Install it with flashinfer-python/flashinfer-cubin."
            ) from error

        try:
            from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
        except ImportError:
            try:
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
            except ImportError as error:
                raise ImportError(
                    "FlashInfer TRTLLM MLA API is unavailable. Need flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla."
                ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if shape.page_size not in {64, 128}:
            raise ValueError("TRTLLM MLA benchmark currently uses page_size 64/128")
        if shape.qk_rope_head_dim != 64:
            raise ValueError("TRTLLM MLA qk_rope_head_dim must be 64")
        if shape.qk_nope_head_dim not in {64, 128}:
            raise ValueError("TRTLLM MLA qk_nope_head_dim must be 64 or 128")
        if shape.kv_lora_rank not in {256, 512}:
            raise ValueError("TRTLLM MLA kv_lora_rank must be 256 or 512")

        self.torch = torch
        self.flashinfer = flashinfer
        self.run_mla = trtllm_batch_decode_with_kv_cache_mla
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(torch, shape.kv_dtype)
        self.pages_per_seq = math.ceil(shape.context_len / shape.page_size)
        self.num_pages = shape.batch_size * self.pages_per_seq
        self.sm_scale = float(1.0 / (shape.head_dim_qk**0.5))

        self.query = torch.randn(
            shape.batch_size,
            1,
            shape.num_q_heads,
            shape.kv_lora_rank + shape.qk_rope_head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_cache = torch.randn(
            self.num_pages,
            shape.page_size,
            shape.kv_lora_rank + shape.qk_rope_head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.block_tables = torch.arange(
            self.num_pages,
            dtype=torch.int32,
            device=self.device,
        ).reshape(shape.batch_size, self.pages_per_seq)
        self.seq_lens = torch.full(
            (shape.batch_size,),
            shape.context_len,
            dtype=torch.int32,
            device=self.device,
        )
        self.workspace_buffer = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.output = torch.empty(
            shape.batch_size,
            1,
            shape.num_q_heads,
            shape.kv_lora_rank,
            dtype=self.dtype,
            device=self.device,
        )

    def run(self):
        return self.run_mla(
            query=self.query,
            kv_cache=self.kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.shape.qk_nope_head_dim,
            kv_lora_rank=self.shape.kv_lora_rank,
            qk_rope_head_dim=self.shape.qk_rope_head_dim,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            max_seq_len=self.shape.context_len,
            sparse_mla_top_k=0,
            out=self.output,
            bmm1_scale=self.sm_scale,
            bmm2_scale=1.0,
            backend="auto",
            is_var_seq=False,
        )
