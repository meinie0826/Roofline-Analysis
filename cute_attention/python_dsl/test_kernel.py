#!/usr/bin/env python3
"""
Test kernel invocation
"""

import torch
import sys
sys.path.insert(0, '.')

print("="*60)
print(" Testing Stage 0 Attention Kernel")
print("="*60)

# Test 1: Import
print("\n[Test 1] Import kernel module...")
try:
    from kernels.stage0_attention import attention_forward, HAS_CUTE
    print(f"  ✓ Import successful")
    print(f"  CuTe available: {HAS_CUTE}")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: PyTorch baseline
print("\n[Test 2] Test PyTorch baseline...")
torch.manual_seed(42)
B, H, N, d = 1, 1, 128, 128
Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)

try:
    O = attention_forward(Q, K, V)
    print(f"  ✓ Forward pass successful")
    print(f"  Output shape: {O.shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Correctness
print("\n[Test 3] Test correctness...")
try:
    # Reference
    scale = 1.0 / (d ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    ref_weights = torch.softmax(scores, dim=-1)
    ref_O = torch.matmul(ref_weights, V)
    
    # Compare
    diff = (O - ref_O).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    if max_error < 1e-3:
        print(f"  ✓ PASSED!")
    else:
        print(f"  ✗ FAILED!")
except Exception as e:
    print(f"  ✗ Correctness test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Different shapes
print("\n[Test 4] Test different shapes...")
test_cases = [
    (1, 1, 64, 128),
    (1, 2, 128, 128),
    (2, 4, 256, 128),
]

for B, H, N, d in test_cases:
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    
    try:
        O = attention_forward(Q, K, V)
        print(f"  ✓ B={B}, H={H}, N={N}, d={d}")
    except Exception as e:
        print(f"  ✗ B={B}, H={H}, N={N}, d={d}: {e}")

print("\n" + "="*60)
print(" Tests Complete")
print("="*60)
