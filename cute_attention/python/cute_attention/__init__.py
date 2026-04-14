"""
FlashAttention Python Interface (CuTe Implementation)

Usage:
    from cute_attention import FlashAttention
    
    fa = FlashAttention(stage=4)  # Use final optimized version
    output = fa(q, k, v, causal=True)
"""

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class FlashAttention:
    """FlashAttention implemented in CuTe with progressive optimization stages"""
    
    STAGES = {
        0: "Naive (Baseline)",
        1: "Tiled with SMEM",
        2: "Optimized SMEM Layout",
        3: "Tensor Core MMA",
        4: "Final (Online Softmax + Pipelining)"
    }
    
    def __init__(self, stage: int = 4):
        """
        Initialize FlashAttention kernel.
        
        Args:
            stage: Optimization stage (0-4)
                   0: Naive baseline
                   1: Tiled with shared memory
                   2: Optimized SMEM layout
                   3: Tensor Core MMA
                   4: Final optimized (default)
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage {stage}. Must be 0-4.")
        
        self.stage = stage
        self._lib = self._load_library()
        self._check_cuda()
    
    def _load_library(self):
        """Load compiled library"""
        lib_path = Path(__file__).parent / "build" / "libcute_attention.so"
        if not lib_path.exists():
            raise RuntimeError(
                f"Library not found: {lib_path}\n"
                "Please build the library first:\n"
                "  cd cute_attention && mkdir build && cd build\n"
                "  cmake .. && make -j"
            )
        
        return ctypes.CDLL(str(lib_path))
    
    def _check_cuda(self):
        """Verify CUDA is available"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FlashAttention")
    
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply FlashAttention.
        
        Args:
            q: Query tensor [batch, seq_len, nheads, head_dim] (BF16)
            k: Key tensor [batch, seq_len, nheads, head_dim] (BF16)
            v: Value tensor [batch, seq_len, nheads, head_dim] (BF16)
            causal: Whether to apply causal mask (default: True)
            scale: Softmax scale (default: 1/sqrt(head_dim))
        
        Returns:
            Output tensor [batch, seq_len, nheads, head_dim] (BF16)
        """
        # Input validation
        if q.dtype != torch.bfloat16:
            raise ValueError(f"Only bfloat16 supported, got {q.dtype}")
        
        if not all(x.is_cuda for x in [q, k, v]):
            raise ValueError("All inputs must be on CUDA device")
        
        batch, seq_len, nheads, head_dim = q.shape
        
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)
        
        # Allocate output
        o = torch.empty_like(q)
        
        # Get raw pointers
        q_ptr = ctypes.c_void_p(q.data_ptr())
        k_ptr = ctypes.c_void_p(k.data_ptr())
        v_ptr = ctypes.c_void_p(v.data_ptr())
        o_ptr = ctypes.c_void_p(o.data_ptr())
        
        # Set up library function
        self._lib.launch_flash_attention.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_bool, ctypes.c_void_p
        ]
        
        stream = torch.cuda.current_stream()
        stream_ptr = ctypes.c_void_p(stream.cuda_stream)
        
        # Launch kernel
        self._lib.launch_flash_attention(
            q_ptr, k_ptr, v_ptr, o_ptr,
            batch, nheads, seq_len, head_dim,
            self.stage, causal, stream_ptr
        )
        
        return o
    
    def __repr__(self):
        return f"FlashAttention(stage={self.stage}, name='{self.STAGES[self.stage]}')"


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd function for FlashAttention"""
    
    @staticmethod
    def forward(ctx, q, k, v, stage, causal):
        fa = FlashAttention(stage=stage)
        o = fa(q, k, v, causal=causal)
        ctx.save_for_backward(q, k, v, o)
        ctx.stage = stage
        ctx.causal = causal
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward not implemented in CuTe version
        # Would need custom backward kernel
        raise NotImplementedError(
            "Backward pass not implemented. "
            "Use torch.autograd.functional for gradient computation."
        )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    stage: int = 4,
    causal: bool = True
) -> torch.Tensor:
    """
    Functional interface for FlashAttention.
    
    Args:
        q: Query [batch, seq_len, nheads, head_dim]
        k: Key [batch, seq_len, nheads, head_dim]
        v: Value [batch, seq_len, nheads, head_dim]
        stage: Optimization stage (0-4, default: 4)
        causal: Apply causal mask (default: True)
    
    Returns:
        Output [batch, seq_len, nheads, head_dim]
    """
    return FlashAttentionFunction.apply(q, k, v, stage, causal)
