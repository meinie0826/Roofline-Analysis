#!/bin/bash
# Run on B200 GPU server
# Requirements: pip install torch nvidia-cutlass-dsl==4.2.0

cd /sgl-workspace/Roofline-Analysis
git pull

echo "==============================================="
echo " FlashAttention-4 Stage Analysis"
echo "==============================================="
python3 cute_attention/python_dsl/flash_attention_stages.py --stages

echo ""
echo "==============================================="
echo " Running Benchmarks"
echo "==============================================="

# Test different stages
for stage in 0 1 8; do
    echo ""
    echo "--- Stage $stage ---"
    python3 -c "
import torch
import sys
sys.path.insert(0, 'python_dsl')
from flash_attention_stages import flash_attention

torch.manual_seed(42)
B, S, H, D = 1, 4096, 16, 128
q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')

out, metrics = flash_attention(q, k, v, stage=$stage)
print(f'  {metrics[\"config\"]}')
print(f'  Time: {metrics[\"time_ms\"]:.3f} ms')
print(f'  TFLOPs: {metrics[\"tflops\"]:.1f}')
print(f'  TC Util: {metrics[\"tc_util_pct\"]:.1f}%')
"
done
