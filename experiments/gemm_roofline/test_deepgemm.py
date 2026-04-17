"""
Simple DeepGEMM Test and Benchmark

This script tests DeepGEMM installation and runs basic benchmarks.
"""

import os
import sys
import time

import torch

print("=" * 70)
print("DeepGEMM Installation Check")
print("=" * 70)

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")

# Check DeepGEMM
print("\n" + "-" * 70)
print("DeepGEMM Import Test")
print("-" * 70)

try:
    import deep_gemm
    print("✓ DeepGEMM imported successfully!")
    
    # List available APIs
    apis = [x for x in dir(deep_gemm) if not x.startswith('_')]
    print(f"\nAvailable APIs ({len(apis)}):")
    for api in apis[:20]:
        print(f"  - {api}")
    if len(apis) > 20:
        print(f"  ... and {len(apis) - 20} more")
    
    HAS_DEEPGEMM = True
except ImportError as e:
    print(f"✗ DeepGEMM import failed: {e}")
    print("\nTo install DeepGEMM, run:")
    print("  git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git")
    print("  cd DeepGEMM && ./develop.sh")
    HAS_DEEPGEMM = False

if not HAS_DEEPGEMM:
    print("\nDeepGEMM not available. Exiting.")
    sys.exit(1)

# Check specific APIs
print("\n" + "-" * 70)
print("Available DeepGEMM GEMM APIs")
print("-" * 70)

# FP8 APIs
fp8_apis = ['fp8_fp4_gemm_nt', 'fp8_gemm_nt', 'fp8_gemm_nn', 'fp8_gemm_tn', 'fp8_gemm_tt']
print("\nFP8 APIs:")
for api in fp8_apis:
    if hasattr(deep_gemm, api):
        print(f"  ✓ {api}")
    else:
        print(f"  ✗ {api}")

# BF16 APIs
bf16_apis = ['bf16_gemm_nt', 'bf16_gemm', 'gemm_bf16']
print("\nBF16 APIs:")
for api in bf16_apis:
    if hasattr(deep_gemm, api):
        print(f"  ✓ {api}")
    else:
        print(f"  ✗ {api}")

# Get the function signature
print("\n" + "-" * 70)
print("FP8 GEMM Function Signature")
print("-" + 70)

# Find any FP8 gemm function
fp8_func = None
for api in fp8_apis:
    if hasattr(deep_gemm, api):
        fp8_func = getattr(deep_gemm, api)
        print(f"\n{api} signature:")
        # Try to get function signature
        try:
            import inspect
            sig = inspect.signature(fp8_func)
            print(f"  {sig}")
        except:
            # Fallback: just print what we know
            print(f"  Function type: {type(fp8_func)}")
        break

# Run simple FP8 test with correct API
print("\n" + "-" * 70)
print("FP8 GEMM Test (New API)")
print("-" + 70)

if fp8_func is not None:
    try:
        M, N, K = 1024, 1024, 1024
        
        print(f"\n  Testing shape: {M}x{N}x{K}")
        
        # Create FP8 tensors
        A_fp8 = torch.randn(M, K, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        B_fp8 = torch.randn(K, N, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        
        # Output tensor (BF16)
        D = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        
        # Create scales - new API expects tuples
        # Per-tensor scale (scalar)
        scale_a = torch.ones((), dtype=torch.float32, device='cuda')
        scale_b = torch.ones((), dtype=torch.float32, device='cuda')
        
        # Pack into tuples: a = (A_tensor, A_scale), b = (B_tensor, B_scale)
        a_tuple = (A_fp8, scale_a)
        b_tuple = (B_fp8, scale_b)
        
        print(f"  A shape: {A_fp8.shape}, dtype: {A_fp8.dtype}")
        print(f"  B shape: {B_fp8.shape}, dtype: {B_fp8.dtype}")
        print(f"  D shape: {D.shape}, dtype: {D.dtype}")
        
        # Warmup
        for _ in range(3):
            fp8_func(a_tuple, b_tuple, D)
            torch.cuda.synchronize()
        
        print("  Warmup passed!")
        
        # Benchmark
        iters = 20
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fp8_func(a_tuple, b_tuple, D)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        avg_time = sum(times) / len(times)
        flops = 2 * M * N * K
        tflops = flops / avg_time / 1e9
        
        print(f"  Time: {avg_time:.3f} ms")
        print(f"  Performance: {tflops:.1f} TFLOPS")
        print("  ✓ FP8 GEMM test passed!")
        
    except Exception as e:
        print(f"  ✗ FP8 GEMM test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative: maybe it's block-wise scaling
        print("\n  Trying block-wise scaling format...")
        try:
            # Block size (common: 128)
            block_m = 128
            block_n = 128
            
            # Block-wise scales
            num_blocks_m = (M + block_m - 1) // block_m
            num_blocks_n = (N + block_n - 1) // block_n
            
            scale_a_block = torch.ones((num_blocks_m,), dtype=torch.float32, device='cuda')
            scale_b_block = torch.ones((num_blocks_n,), dtype=torch.float32, device='cuda')
            
            a_tuple = (A_fp8, scale_a_block)
            b_tuple = (B_fp8, scale_b_block)
            
            D = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            
            fp8_func(a_tuple, b_tuple, D)
            torch.cuda.synchronize()
            
            print("  ✓ Block-wise scaling works!")
            
        except Exception as e2:
            print(f"  ✗ Block-wise scaling also failed: {e2}")
else:
    print("  ✗ No FP8 GEMM function available")

# BF16 test with DeepGEMM
print("\n" + "-" * 70)
print("DeepGEMM BF16 GEMM Test")
print("-" + 70)

if hasattr(deep_gemm, 'bf16_gemm_nt'):
    try:
        M, N, K = 1024, 1024, 1024
        
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
        D = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        
        # Warmup
        deep_gemm.bf16_gemm_nt(A, B, D)
        torch.cuda.synchronize()
        
        # Benchmark
        iters = 20
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            deep_gemm.bf16_gemm_nt(A, B, D)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        avg_time = sum(times) / len(times)
        flops = 2 * M * N * K
        tflops = flops / avg_time / 1e9
        
        print(f"  Shape: {M}x{N}x{K}")
        print(f"  Time: {avg_time:.3f} ms")
        print(f"  Performance: {tflops:.1f} TFLOPS")
        print("  ✓ DeepGEMM BF16 test passed!")
        
    except Exception as e:
        print(f"  ✗ DeepGEMM BF16 test failed: {e}")
else:
    print("  ✗ bf16_gemm_nt API not available")

# cuBLAS BF16 reference
print("\n" + "-" * 70)
print("cuBLAS BF16 GEMM Test (for comparison)")
print("-" + 70)

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
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
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

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
