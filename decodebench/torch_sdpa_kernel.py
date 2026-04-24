from __future__ import annotations

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
        return self.batch_size * self.context_len * self.num_kv_heads * self.head_dim * 2 * 2


def dtype_from_name(torch, name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"torch_sdpa_decode only supports bf16/fp16, got {name}")


class TorchSDPADecodeKernel:
    def __init__(self, shape: DecodeShape):
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.torch = torch
        self.F = F
        self.shape = shape
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(torch, shape.kv_dtype)
        self.q = torch.randn(
            shape.batch_size,
            shape.num_q_heads,
            1,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.k = torch.randn(
            shape.batch_size,
            shape.num_kv_heads,
            shape.context_len,
            shape.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.v = torch.randn_like(self.k)

    def run(self):
        if self.shape.num_q_heads == self.shape.num_kv_heads:
            return self.F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=False)
        try:
            return self.F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=False, enable_gqa=True)
        except TypeError:
            repeat = self.shape.num_q_heads // self.shape.num_kv_heads
            k = self.k.repeat_interleave(repeat, dim=1)
            v = self.v.repeat_interleave(repeat, dim=1)
            return self.F.scaled_dot_product_attention(self.q, k, v, is_causal=False)
