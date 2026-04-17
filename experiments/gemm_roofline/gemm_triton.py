"""
GEMM Kernel using Triton

This module provides an optimized GEMM kernel implementation using Triton
for analyzing roofline characteristics across different matrix shapes.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available. Install with: pip install triton")


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """
        GEMM kernel: C = A @ B
        A: (M, K)
        B: (K, N)
        C: (M, N)
        """
        # Block indices
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Block pointers for A and B
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Loop over K
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            # Load tiles
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            
            # Matrix multiply
            acc += tl.dot(a, b)
            
            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Store result
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using Triton kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c


def benchmark_triton_matmul(M: int, N: int, K: int,
                            warmup: int = 10,
                            iterations: int = 50,
                            device: str = "cuda",
                            dtype: torch.dtype = torch.float16) -> dict:
    """
    Benchmark Triton GEMM kernel.
    
    Args:
        M, N, K: Matrix dimensions
        warmup: Warmup iterations
        iterations: Timed iterations
        device: Device to run on
        dtype: Data type
    
    Returns:
        Dictionary with timing and performance metrics
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    
    # Create tensors
    torch.manual_seed(42)
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        c = triton_matmul(a, b)
    torch.cuda.synchronize()
    
    # Timed runs
    import time
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = triton_matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    flops = 2 * M * N * K
    gflops = flops / (avg_time * 1e-3) / 1e9
    
    # Arithmetic intensity
    element_size = 2 if dtype == torch.float16 else 4
    bytes_accessed = (M * K + K * N + M * N) * element_size
    ai = flops / bytes_accessed
    
    return {
        'M': M, 'N': N, 'K': K,
        'time_ms': avg_time,
        'gflops': gflops,
        'arithmetic_intensity': ai,
        'bytes_accessed': bytes_accessed,
    }


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available, skipping test")
        exit(0)
    
    print("Testing Triton GEMM kernel...")
    
    # Test shapes
    shapes = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
    ]
    
    print(f"{'Shape':>20} | {'Time (ms)':>12} | {'GFLOPS':>12} | {'AI':>10}")
    print("-" * 60)
    
    for M, N, K in shapes:
        result = benchmark_triton_matmul(M, N, K)
        print(f"{M:>8}x{N:>8}x{K:>8} | {result['time_ms']:>12.3f} | {result['gflops']:>12.1f} | {result['arithmetic_intensity']:>10.1f}")
