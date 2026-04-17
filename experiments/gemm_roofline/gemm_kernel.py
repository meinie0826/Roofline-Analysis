"""
GEMM Kernel for Roofline Analysis using CuTeDSL

This module provides a simple GEMM kernel implementation to analyze
performance characteristics across different matrix shapes.
"""

from __future__ import annotations

import os
import warnings

# Suppress cutlass deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import cutlass
    import cutlass.cute as cute
    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False
    print("Warning: cutlass.cute not available. Install nvidia-cutlass-dsl.")

import torch

# Cache for compiled kernels
_COMPILED_KERNELS = {}


def make_gemm_kernel(M: int, N: int, K: int, 
                     block_m: int = 64, block_n: int = 64, block_k: int = 32,
                     num_threads: int = 128):
    """
    Create a compiled GEMM kernel for given dimensions.
    
    Args:
        M, N, K: Matrix dimensions (C = A @ B, where A is M×K, B is K×N)
        block_m, block_n, block_k: Tile sizes for blocking
        num_threads: Number of threads per block
    
    Returns:
        Compiled kernel function
    """
    if not HAS_CUTE:
        raise RuntimeError("CuTe DSL not available")
    
    @cute.kernel
    def gemm_kernel(
        A: cute.Tensor,  # M x K, row-major
        B: cute.Tensor,  # K x N, row-major
        C: cute.Tensor,  # M x N, row-major
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
    ):
        # Get block indices
        m_block, n_block, _ = cute.arch.block_idx()
        
        # Starting positions for this block
        m_start = m_block * block_m
        n_start = n_block * block_n
        
        # Thread index
        tidx, _, _ = cute.arch.thread_idx()
        
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        
        # Shared memory tiles
        a_smem = smem.allocate_array(cutlass.Float16, num_elems=block_m * block_k)
        b_smem = smem.allocate_array(cutlass.Float16, num_elems=block_k * block_n)
        
        # Create tensor views for shared memory
        a_tile = cute.make_tensor(a_smem, cute.make_layout((block_m, block_k)))
        b_tile = cute.make_tensor(b_smem, cute.make_layout((block_k, block_n)))
        
        # Accumulator in registers
        acc = cute.make_tensor(cutlass.Float32, cute.make_layout((block_m, block_n)))
        
        # Initialize accumulator
        for i in range(tidx, block_m * block_n, num_threads):
            acc[i // block_n, i % block_n] = cutlass.Float32(0.0)
        
        cute.arch.barrier()
        
        # Loop over K dimension in blocks
        for k_start in range(0, K, block_k):
            # Load A tile from global memory to shared memory
            for i in range(tidx, block_m * block_k, num_threads):
                row_a = i // block_k
                col_a = i % block_k
                gmem_row_a = m_start + row_a
                gmem_col_a = k_start + col_a
                
                val = cutlass.Float16(0.0)
                if gmem_row_a < M and gmem_col_a < K:
                    val = A[gmem_row_a, gmem_col_a]
                a_tile[row_a, col_a] = val
            
            # Load B tile from global memory to shared memory
            for i in range(tidx, block_k * block_n, num_threads):
                row_b = i // block_n
                col_b = i % block_n
                gmem_row_b = k_start + row_b
                gmem_col_b = n_start + col_b
                
                val = cutlass.Float16(0.0)
                if gmem_row_b < K and gmem_col_b < N:
                    val = B[gmem_row_b, gmem_col_b]
                b_tile[row_b, col_b] = val
            
            cute.arch.barrier()
            
            # Compute block matrix multiplication
            # This is a simplified version - real implementation would use Tensor Cores
            for m_idx in range(block_m):
                for n_idx in range(block_n):
                    dot = cutlass.Float32(0.0)
                    for k_idx in range(block_k):
                        gmem_m = m_start + m_idx
                        gmem_n = n_start + n_idx
                        if gmem_m < M and gmem_n < N:
                            a_val = a_tile[m_idx, k_idx].to(cutlass.Float32)
                            b_val = b_tile[k_idx, n_idx].to(cutlass.Float32)
                            dot += a_val * b_val
                    acc[m_idx, n_idx] = acc[m_idx, n_idx] + dot
        
        cute.arch.barrier()
        
        # Store results to global memory
        for i in range(tidx, block_m * block_n, num_threads):
            row_c = i // block_n
            col_c = i % block_n
            gmem_row_c = m_start + row_c
            gmem_col_c = n_start + col_c
            
            if gmem_row_c < M and gmem_col_c < N:
                result = alpha * acc[row_c, col_c]
                if beta != cutlass.Float32(0.0):
                    result = result + beta * C[gmem_row_c, gmem_col_c].to(cutlass.Float32)
                C[gmem_row_c, gmem_col_c] = result.to(C.element_type)
    
    @cute.jit
    def gemm_host(
        A: cute.Tensor,
        B: cute.Tensor, 
        C: cute.Tensor,
        alpha: cutlass.Float32,
        beta: cutlass.Float32,
    ):
        grid_m = (M + block_m - 1) // block_m
        grid_n = (N + block_n - 1) // block_n
        
        gemm_kernel(A, B, C, alpha, beta).launch(
            grid=(grid_m, grid_n, 1),
            block=(num_threads, 1, 1),
        )
    
    return gemm_host


def run_gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
             alpha: float = 1.0, beta: float = 0.0,
             block_m: int = 64, block_n: int = 64, block_k: int = 32) -> torch.Tensor:
    """
    Run GEMM: C = alpha * (A @ B) + beta * C
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
        C: Output tensor of shape (M, N)
        alpha, beta: Scaling factors
        block_m, block_n, block_k: Tile sizes
    
    Returns:
        Output tensor C
    """
    if not HAS_CUTE:
        raise RuntimeError("CuTe DSL not available")
    
    M, K = A.shape
    _, N = B.shape
    
    # Convert to contiguous tensors
    A_cont = A.contiguous()
    B_cont = B.contiguous()
    C_cont = C.contiguous()
    
    # Convert to CuTe tensors
    from cutlass.cute import from_dlpack
    A_cute = from_dlpack(A_cont, assumed_align=16).mark_layout_dynamic()
    B_cute = from_dlpack(B_cont, assumed_align=16).mark_layout_dynamic()
    C_cute = from_dlpack(C_cont, assumed_align=16).mark_layout_dynamic()
    
    # Cache key
    cache_key = (M, N, K, block_m, block_n, block_k, str(A.dtype), str(C.dtype))
    
    if cache_key not in _COMPILED_KERNELS:
        kernel = make_gemm_kernel(M, N, K, block_m, block_n, block_k)
        compiled = cute.compile(
            kernel,
            A_cute, B_cute, C_cute,
            cutlass.Float32(alpha),
            cutlass.Float32(beta),
        )
        _COMPILED_KERNELS[cache_key] = compiled
    else:
        compiled = _COMPILED_KERNELS[cache_key]
    
    # Run kernel
    compiled(A_cute, B_cute, C_cute, 
             cutlass.Float32(alpha), cutlass.Float32(beta))
    
    return C_cont


def get_reference_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Get reference GEMM result using PyTorch."""
    return torch.matmul(A, B)


if __name__ == "__main__":
    # Simple test
    if not HAS_CUTE:
        print("CuTe DSL not available, skipping test")
        exit(0)
    
    M, N, K = 512, 512, 512
    
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    
    # Warmup
    run_gemm(A, B, C)
    torch.cuda.synchronize()
    
    # Timing
    import time
    start = time.time()
    for _ in range(10):
        run_gemm(A, B, C)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 10
    flops = 2 * M * N * K
    tflops = flops / avg_time / 1e12
    
    print(f"GEMM {M}x{N}x{K}: {avg_time*1000:.3f} ms, {tflops:.2f} TFLOPS")
