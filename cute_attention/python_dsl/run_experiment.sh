#!/usr/bin/env bash
# =============================================================================
# FlashAttention-4 Benchmark Runner
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
git pull

echo "==============================================="
echo " FlashAttention-4 Stage Benchmark"
echo "==============================================="
echo ""

# Check Python and CUDA
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "ERROR: CUDA required"
    exit 1
}

# Run benchmark for different stages
for stage in 0 2 6 12; do
    echo ""
    echo "=== Stage $stage ==="
    python3 -c "
import torch
import sys
sys.path.insert(0, 'python_dsl')
from fa4_cute_stages import flash_attention, print_optimization_breakdown

if $stage == 0:
    print_optimization_breakdown()
else:
    B, S, H, D = 1, 4096, 16, 128
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    
    print(f'Testing Stage $stage: B={B}, S={S}, H={H}, D={D}')
    try:
        out, lse = flash_attention(q, k, v, stage=$stage)
        print(f'Output shape: {out.shape}')
        print(f'Stage $stage: Implementation available')
    except Exception as e:
        print(f'Stage $stage: {e}')
"
done

echo ""
echo "==============================================="
echo " Benchmark Complete"
echo "==============================================="
