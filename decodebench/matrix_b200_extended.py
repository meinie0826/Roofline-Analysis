"""Extended B200 decode attention benchmark sweep.

Use this when the lightweight default matrix is not enough. It expands coverage
for latency-first serving, batch/context trade-offs, GQA ratio, head dimension,
FP8 long-context, and MLA parameter sweeps.
"""

from matrix_b200 import BACKENDS, DEFAULTS, workload


WORKLOADS = [
    # Latency-first single-request serving.
    workload("GQA", "bf16", batch_size=1, context_len=4096, page_size=64),
    workload("GQA", "bf16", batch_size=1, context_len=32768, page_size=64),
    workload("GQA", "bf16", batch_size=1, context_len=131072, page_size=64),
    workload("GQA", "fp8", batch_size=1, context_len=32768, page_size=64),
    workload("GQA", "fp8", batch_size=1, context_len=131072, page_size=64),

    # Batch sweep at fixed context.
    workload("GQA", "bf16", batch_size=1, context_len=8192, page_size=64),
    workload("GQA", "bf16", batch_size=8, context_len=8192, page_size=64),
    workload("GQA", "bf16", batch_size=32, context_len=8192, page_size=64),
    workload("GQA", "bf16", batch_size=128, context_len=8192, page_size=64),
    workload("GQA", "fp8", batch_size=8, context_len=8192, page_size=64),
    workload("GQA", "fp8", batch_size=32, context_len=8192, page_size=64),
    workload("GQA", "fp8", batch_size=128, context_len=8192, page_size=64),

    # GQA ratio sweep: group size 2/4/8/16/32.
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=32, num_kv_heads=16),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=32, num_kv_heads=8),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=32, num_kv_heads=4),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=32, num_kv_heads=2),
    workload("MQA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=32, num_kv_heads=1),

    # Head-dimension sensitivity.
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, head_dim=64),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64, head_dim=192),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=64, head_dim=64),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=64, head_dim=128),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=64, head_dim=192),

    # Page-size sweep on a long-context BF16 shape.
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=16),
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=32),
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=64),
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=128),

    # MLA B200-capable TRTLLM-GEN sweep.
    workload("MLA", "bf16", batch_size=1, context_len=32768, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=8, context_len=32768, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=32, context_len=32768, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "fp16", batch_size=64, context_len=4096, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=256, qk_nope_head_dim=64, kv_lora_rank=256),
]

CONFIG = {
    "metadata": {
        "name": "decode-attn-b200-extended-sweep",
        "gpu": "B200",
        "goal": "Extended decode attention sweep for latency, batch, GQA ratio, head dim, page size, FP8, and MLA coverage.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
