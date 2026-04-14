#!/usr/bin/env python3
"""
Correctness Tests for FlashAttention Stages
"""

import torch
import math


def reference_attention(Q, K, V, scale=None):
    """PyTorch reference implementation"""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    
    return output


def check_correctness(func, Q, K, V, rtol=1e-3, atol=1e-3, verbose=True):
    """
    Check correctness against PyTorch reference
    
    Args:
        func: function to test
        Q, K, V: input tensors
        rtol, atol: tolerance
        verbose: print detailed info
    
    Returns:
        passed: bool
        max_error: float
    """
    # Reference output
    ref_output = reference_attention(Q, K, V)
    
    # Test output
    test_output = func(Q, K, V)
    
    # Compare
    diff = (test_output - ref_output).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    passed = torch.allclose(test_output, ref_output, rtol=rtol, atol=atol)
    
    if verbose:
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed, max_error


def test_attention_basic(func, verbose=True):
    """Basic correctness test"""
    torch.manual_seed(42)
    
    B, H, N, d = 1, 1, 128, 64
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    
    if verbose:
        print(f"\n[Test] Basic: B={B}, H={H}, N={N}, d={d}")
    
    return check_correctness(func, Q, K, V, verbose=verbose)


def test_attention_shapes(func, verbose=True):
    """Test different shapes"""
    torch.manual_seed(42)
    
    test_cases = [
        (1, 1, 64, 64),
        (1, 2, 128, 64),
        (2, 4, 256, 64),
        (1, 8, 512, 64),
    ]
    
    all_passed = True
    
    for B, H, N, d in test_cases:
        Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
        
        if verbose:
            print(f"\n[Test] Shape: B={B}, H={H}, N={N}, d={d}")
        
        passed, _ = check_correctness(func, Q, K, V, verbose=verbose)
        all_passed = all_passed and passed
    
    return all_passed


def test_attention_dtypes(func, verbose=True):
    """Test different dtypes"""
    torch.manual_seed(42)
    
    B, H, N, d = 1, 2, 128, 64
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    all_passed = True
    
    for dtype in dtypes:
        Q = torch.randn(B, H, N, d, device='cuda', dtype=dtype)
        K = torch.randn(B, H, N, d, device='cuda', dtype=dtype)
        V = torch.randn(B, H, N, d, device='cuda', dtype=dtype)
        
        if verbose:
            print(f"\n[Test] Dtype: {dtype}")
        
        # Relax tolerance for fp16/bf16
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        atol = 1e-3 if dtype == torch.float32 else 1e-2
        
        passed, _ = check_correctness(func, Q, K, V, rtol=rtol, atol=atol, verbose=verbose)
        all_passed = all_passed and passed
    
    return all_passed


def run_all_tests(func, verbose=True):
    """Run all correctness tests"""
    print("="*60)
    print(" Running Correctness Tests")
    print("="*60)
    
    tests = [
        ("Basic", test_attention_basic),
        ("Shapes", test_attention_shapes),
        ("Dtypes", test_attention_dtypes),
    ]
    
    results = {}
    
    for name, test_func in tests:
        if verbose:
            print(f"\n--- {name} Tests ---")
        
        try:
            results[name] = test_func(func, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  ✗ ERROR: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    try:
        from kernels.stage0_attention import attention_forward
        run_all_tests(attention_forward)
    except RuntimeError as e:
        print(f"\nCannot run tests: {e}")
