"""B200 decode attention benchmark search space.

The default matrix is capability-driven instead of hand-picking one row per
backend. It intentionally contains both kernel-only backends and vLLM's own
attention benchmark paths; use the ``layer`` field in result JSON to separate
kernel-only from framework-level rows during analysis.
"""

DEFAULTS = {
    "decode_steps": 1,
    "warmup_steps": 20,
    "repeat": 100,
}

BACKENDS = [
    {
        "name": "flashinfer_paged_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "FlashInfer BatchDecodeWithPagedKVCacheWrapper / XQA",
        "status": "implemented",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16", "fp8"},
        "supported_page_sizes": {16, 32, 64, 128},
    },
    {
        "name": "flashinfer_trtllm_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
        "status": "implemented",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16", "fp8"},
        "supported_page_sizes": {64, 128},
    },
    {
        "name": "flashattn_kvcache",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "FlashAttention FA4 flash_attn.cute.interface.flash_attn_varlen_func with page_table",
        "status": "implemented_if_fa4_is_installed",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {16, 32, 64, 128},
    },
    {
        "name": "vllm_paged_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "vllm._custom_ops.paged_attention_v2",
        "status": "implemented_if_vllm_is_installed",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {16, 32},
    },
    {
        "name": "flashmla_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flash_mla.flash_mla_with_kvcache",
        "status": "implemented_if_flashmla_is_installed",
        "supported_attention": {"MLA"},
        "supported_kv_dtypes": {"bf16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "vllm_flash",
        "layer": "framework_benchmark",
        "enabled": True,
        "kernel_path": "vLLM attention_benchmarks backend=flash",
        "status": "enabled_framework_level_not_kernel_only",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {16, 32, 64, 128},
    },
    {
        "name": "vllm_flashinfer",
        "layer": "framework_benchmark",
        "enabled": True,
        "kernel_path": "vLLM attention_benchmarks backend=flashinfer",
        "status": "enabled_framework_level_not_kernel_only",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16", "fp8"},
        "supported_page_sizes": {16, 32, 64, 128},
    },
]


def ctx_label(context_len: int) -> str:
    return f"{context_len // 1024}k" if context_len % 1024 == 0 else str(context_len)


def workload(
    attention: str,
    kv_dtype: str,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_q_heads: int = 32,
    num_kv_heads: int | None = None,
    head_dim: int = 128,
    head_dim_v: int | None = None,
) -> dict:
    if attention == "MLA":
        if num_kv_heads is None:
            num_kv_heads = 1
        item = {
            "id": f"mla_{kv_dtype}_b{batch_size}_ctx{ctx_label(context_len)}_p{page_size}",
            "attention": attention,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "head_dim_v": 512 if head_dim_v is None else head_dim_v,
            "kv_dtype": kv_dtype,
            "page_size": page_size,
            "batch_size": batch_size,
            "context_len": context_len,
        }
        return item

    if num_kv_heads is None:
        num_kv_heads = {"MHA": num_q_heads, "GQA": 8, "MQA": 1}[attention]
    return {
        "id": f"{attention.lower()}_{kv_dtype}_b{batch_size}_ctx{ctx_label(context_len)}_p{page_size}",
        "attention": attention,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype": kv_dtype,
        "page_size": page_size,
        "batch_size": batch_size,
        "context_len": context_len,
    }


WORKLOADS = [
    # Cross-backend fair set: BF16/FP16 and page_size=32 is supported by vLLM paged.
    workload("MHA", "bf16", batch_size=16, context_len=4096, page_size=32),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=32),
    workload("MQA", "bf16", batch_size=128, context_len=4096, page_size=32),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=32),

    # Serving-default page sizes used by FlashInfer/TRT-LLM/FA4-style paged KV.
    workload("MHA", "bf16", batch_size=16, context_len=4096, page_size=64),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64),
    workload("MQA", "bf16", batch_size=128, context_len=4096, page_size=64),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=64),

    # FP8 SOTA decode paths.
    workload("GQA", "fp8", batch_size=64, context_len=4096, page_size=64),
    workload("GQA", "fp8", batch_size=64, context_len=4096, page_size=128),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=64),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=128),
    workload("MQA", "fp8", batch_size=128, context_len=8192, page_size=64),

    # Page-size sensitivity at a fixed GQA BF16 serving shape.
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=16),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=128),

    # Batch/context trade-off at roughly realistic online decode scales.
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=32),
    workload("GQA", "bf16", batch_size=1, context_len=131072, page_size=64),
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=64),
    workload("GQA", "bf16", batch_size=32, context_len=8192, page_size=64),
    workload("GQA", "bf16", batch_size=128, context_len=2048, page_size=64),

    # MLA-only DeepSeek-style decode shapes.
    workload("MLA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=32, context_len=32768, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
]

CONFIG = {
    "metadata": {
        "name": "decode-attn-b200-expanded-sota",
        "gpu": "B200",
        "goal": "Compare SOTA decode attention paths across fair common shapes, serving page sizes, FP8 long-context, batch/context trade-offs, and MLA.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
