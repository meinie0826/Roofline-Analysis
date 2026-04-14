#!/usr/bin/env python
"""
Check if the installed flash-attn-4 supports ablation interfaces.

Run this first to determine if you need to clone FA4 source.
"""
import sys

def check_ablation_support():
    print("=" * 60)
    print("FA4 Ablation Interface Check")
    print("=" * 60)
    
    # Check 1: Can import flash_attn?
    try:
        import flash_attn
        print(f"✓ flash_attn installed at: {flash_attn.__file__}")
        print(f"  Version: {getattr(flash_attn, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"✗ flash_attn not installed: {e}")
        print("\n  Install with: pip install flash-attn-4 --no-build-isolation")
        return False
    
    # Check 2: Can import cute interface?
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        print(f"✓ _flash_attn_fwd available")
    except ImportError as e:
        print(f"✗ _flash_attn_fwd not available: {e}")
        print("  Ablation requires internal FA4 interface")
        return False
    
    # Check 3: Does _flash_attn_fwd accept ablation kwargs?
    import inspect
    sig = inspect.signature(_flash_attn_fwd)
    params = list(sig.parameters.keys())
    
    ablation_params = [
        "_ablation_q_stage",
        "_ablation_no_lpt", 
        "_ablation_rescale_threshold",
        "_ablation_ex2_emu_freq",
    ]
    
    supported = []
    unsupported = []
    for p in ablation_params:
        if p in params:
            supported.append(p)
        else:
            unsupported.append(p)
    
    print(f"\n  Ablation parameters support:")
    for p in supported:
        print(f"    ✓ {p}")
    for p in unsupported:
        print(f"    ✗ {p}")
    
    # Check 4: Device capability
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm = major * 10 + minor
            print(f"\n  GPU: {torch.cuda.get_device_name()}")
            print(f"  Compute capability: SM{sm}")
            if sm >= 100:
                print(f"  ✓ SM100+ (Blackwell) - FA4 kernels supported")
            else:
                print(f"  ⚠ SM{sm} < SM100 - FA4 requires B200/B100")
        else:
            print("\n  ⚠ No CUDA device detected")
    except Exception as e:
        print(f"\n  ⚠ Could not check GPU: {e}")
    
    print("\n" + "=" * 60)
    if len(unsupported) == 0:
        print("RESULT: Full ablation support ✓")
        print("  You can run: bash fa4/run_experiment_with_pip.sh")
        return True
    else:
        print("RESULT: Ablation NOT supported")
        print("\n  To enable ablation, you need to:")
        print("  1. Clone FA4 source:")
        print("     git clone https://github.com/Dao-AILab/flash-attention.git")
        print("     cd flash-attention")
        print("  2. Copy benchmark script:")
        print("     cp /path/to/Roofline-Analysis/fa4/benchmark_ablation_sm100.py benchmarks/")
        print("  3. Install from source:")
        print("     pip install -e . --no-build-isolation")
        print("  4. Run:")
        print("     bash /path/to/Roofline-Analysis/fa4/run_experiment.sh .")
        return False


if __name__ == "__main__":
    success = check_ablation_support()
    sys.exit(0 if success else 1)
