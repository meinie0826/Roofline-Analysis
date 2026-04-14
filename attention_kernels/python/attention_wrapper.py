"""
FlashAttention Kernel Wrapper

Python bindings for all 5 stages of FlashAttention kernels.
"""

import ctypes
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# Load shared library
_LIB_PATH = Path(__file__).parent.parent / "build" / "libattention_kernels.so"

if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Shared library not found at {_LIB_PATH}\n"
        "Please build the kernels first:\n"
        "  cd attention_kernels && mkdir build && cd build\n"
        "  cmake .. && make -j$(nproc)"
    )

_LIB = ctypes.CDLL(str(_LIB_PATH))

# C function signatures
_LIB.launch_attention_naive.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_bool, ctypes.c_void_p
]
_LIB.launch_attention_naive.restype = None

_LIB.launch_attention_tiled.argtypes = _LIB.launch_attention_naive.argtypes
_LIB.launch_attention_tiled.restype = None

_LIB.launch_attention_smem.argtypes = _LIB.launch_attention_naive.argtypes
_LIB.launch_attention_smem.restype = None

_LIB.launch_attention_tensor_core.argtypes = _LIB.launch_attention_naive.argtypes
_LIB.launch_attention_tensor_core.restype = None

_LIB.launch_attention_final.argtypes = _LIB.launch_attention_naive.argtypes
_LIB.launch_attention_final.restype = None


@dataclass
class AttentionConfig:
    """Configuration for attention computation"""
    batch_size: int
    seq_len: int
    n_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    causal: bool = False
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.batch_size, self.seq_len, self.n_heads, self.head_dim)
    
    def total_flops(self, forward_only: bool = True) -> int:
        """Calculate total FLOPs for attention computation"""
        # QK^T: batch * n_heads * seq_len^2 * head_dim * 2
        # Softmax: batch * n_heads * seq_len^2 * 5 (exp, sum, div)
        # PV: batch * n_heads * seq_len^2 * head_dim * 2
        qk_flops = self.batch_size * self.n_heads * 2 * self.seq_len * self.seq_len * self.head_dim
        pv_flops = self.batch_size * self.n_heads * 2 * self.seq_len * self.seq_len * self.head_dim
        
        # For causal, average sequence length is (seq_len + 1) / 2
        if self.causal:
            factor = 0.5  # Approximate: (1 + 1/seq_len) / 2 ≈ 0.5
            qk_flops = int(qk_flops * factor)
            pv_flops = int(pv_flops * factor)
        
        return qk_flops + pv_flops


class FlashAttentionKernel:
    """Wrapper for all FlashAttention kernel stages"""
    
    STAGES = {
        0: "naive",
        1: "tiled",
        2: "shared_mem",
        3: "tensor_core",
        4: "final"
    }
    
    def __init__(self, stage: int = 4):
        """
        Initialize kernel wrapper.
        
        Args:
            stage: Optimization stage (0-4)
                0: Naive (baseline)
                1: Tiled
                2: Shared memory
                3: Tensor Core
                4: Final (online softmax + software pipeline)
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage {stage}. Must be 0-4.")
        self.stage = stage
        self._launch_fn = getattr(_LIB, f"launch_attention_{self.STAGES[stage]}")
    
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Compute attention: O = softmax(Q @ K^T / sqrt(d)) @ V
        
        Args:
            q, k, v: Input tensors [batch, seq_len, n_heads, head_dim]
            causal: Whether to use causal attention mask
            
        Returns:
            Output tensor [batch, seq_len, n_heads, head_dim]
        """
        assert q.dim() == 4, "Input must be 4D: [batch, seq_len, n_heads, head_dim]"
        assert q.dtype == torch.bfloat16, "Only BF16 is supported"
        assert q.is_cuda, "Only CUDA tensors are supported"
        
        batch, seq_len, n_heads, head_dim = q.shape
        o = torch.empty_like(q)
        
        stream = torch.cuda.current_stream().cuda_stream
        
        self._launch_fn(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            o.data_ptr(),
            batch,
            seq_len,
            n_heads,
            head_dim,
            causal,
            ctypes.c_void_p(stream)
        )
        
        return o
    
    @staticmethod
    def reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Reference implementation using PyTorch SDPA.
        """
        # Convert to [batch, n_heads, seq_len, head_dim] for SDPA
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        
        scale = q.shape[-1] ** -0.5
        o = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
        
        # Convert back to [batch, seq_len, n_heads, head_dim]
        return o.transpose(1, 2).to(torch.bfloat16)


def create_test_inputs(
    batch_size: int = 4,
    seq_len: int = 1024,
    n_heads: int = 16,
    head_dim: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random test inputs for attention."""
    torch.manual_seed(seed)
    shape = (batch_size, seq_len, n_heads, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    return q, k, v


if __name__ == "__main__":
    # Quick test
    print("Testing FlashAttention kernels...")
    
    q, k, v = create_test_inputs(batch_size=2, seq_len=512)
    
    # Reference
    ref_out = FlashAttentionKernel.reference(q, k, v, causal=False)
    print(f"Reference output shape: {ref_out.shape}")
    
    # Test each stage
    for stage in range(5):
        try:
            kernel = FlashAttentionKernel(stage)
            out = kernel(q, k, v, causal=False)
            diff = (out.float() - ref_out.float()).abs().max().item()
            print(f"Stage {stage} ({FlashAttentionKernel.STAGES[stage]}): max diff = {diff:.6f}")
        except Exception as e:
            print(f"Stage {stage}: ERROR - {e}")
