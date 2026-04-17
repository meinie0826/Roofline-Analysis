"""
Simple DeepGEMM Test and Benchmark

This script tests DeepGEMM installation and runs basic benchmarks.
"""

import os
import sys
import time

import torch

print("=" * 60)
print("DeepGEMM Installation Check")
print("=" * 60)

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")

# Check DeepGEMM
print("\n" + "-" * 60)
print("DeepGEMM Import Test")
print("-" * 60)

try:
    import deep_gemm
    print("✓ DeepGEMM imported successfully!")
    print(f"  Available functions: {[x for x in dir(deep_gemm) if not x.startswith('_')][:10]}")
    HAS_DEEPGEMM = True
except ImportError as e:
    print(f"✗ DeepGEMM import failed: {e}")
    print("\nTo install DeepGEMM, run:")
    print("  git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git")
    print("  cd DeepGEMM")
    print("  ./develop.sh  # or ./install.sh")
    HAS_DEEPGEMM = False

if not HAS_DEEPGEMM:
    print("\nDeepGEMM not available. Exiting.")
    sys.exit(1)

# Check available API
print("\n" + "-" * 60)
print("Available DeepGEMM APIs")
print("-" * 60)

apis = ['fp8_gemm_nt', 'fp8_gemm_nn', 'fp8_gemm_tn', 'fp8_gemm_tt']
for api in apis:
    if hasattr(deep_gemm, api):
        print(f"  ✓ {api}")
    else:
        print(f"  ✗ {api}")

# Check BF16 API
bf16_apis = ['bf16_gemm_nt', 'bf16_gemm', 'gemm_bf16']
for api in bf16_apis:
    if hasattr(deep_gemm, api):
        print(f"  ✓ {api} (BF16)")
    else:
        print(f"  ✗ {api} (BF16 - not available)")

# Run simple FP8 test
print("\n" + "-" * 60)
print("FP8 GEMM Test")
print("-" * 60)

if hasattr(deep_gemm, 'fp8_gemm_nt'):
    try:
        M, N, K = 1024, 1024, 1024
        
        # Create FP8 tensors
        A = torch.randn(M, K, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        B = torch.randn(K, N, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        
        # Scaling factors (FP32)
        scale_a = torch.ones(M, dtype=torch.float32, device='cuda')
        scale_b = torch.ones(N, dtype=torch.float32, device='cuda')
        
        # Output (BF16)
        C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        deep_gemm.fp8_gemm_nt(C, A, scale_a, B, scale_b)
        torch.cuda.synchronize()
        
        # Benchmark
        iters = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(iters):
            start.record()
            deep_gemm.fp8_gemm_nt(C, A, scale_a, B, scale_b)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        avg_time = sum(times) / len(times)
        flops = 2 * M * N * K
        tflops = flops / avg_time / 1e9
        
        print(f"  Shape: {M}x{N}x{K}")
        print(f"  Time: {avg_time:.3f} ms")
        print(f"  Performance: {tflops:.1f} TFLOPS")
        print("  ✓ FP8 GEMM test passed!")
        
    except Exception as e:
        print(f"  ✗ FP8 GEMM test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ✗ fp8_gemm_nt API not available")

# Compare with cuBLAS BF16
print("\n" + "-" * 60)
print("cuBLAS BF16 GEMM Test (for comparison)")
print("-" * 60)

try:
    M, N, K = 1024, 1024, 1024
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    
    # Warmup
    torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    
    # Benchmark
    iters = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = sum(times) / len(times)
    flops = 2 * M * N * K
    tflops = flops / avg_time / 1e9
    
    print(f"  Shape: {M}x{N}x{K}")
    print(f"  Time: {avg_time:.3f} ms")
    print(f"  Performance: {tflops:.1f} TFLOPS")
    print("  ✓ cuBLAS BF16 test passed!")
    
except Exception as e:
    print(f"  ✗ cuBLAS test failed: {e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
