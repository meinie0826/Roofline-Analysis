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
        "supported_attention": {"MHA", "GQA"},
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
        "supported_page_sizes": {64},
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
        "enabled": False,
        "kernel_path": "flash_mla.flash_mla_with_kvcache",
        "status": "disabled_on_b200_dense_decode_mla_sm90a_only",
        "supported_attention": {"MLA"},
        "supported_kv_dtypes": {"bf16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "flashattn_mla_decode",
        "layer": "kernel",
        "enabled": False,
        "kernel_path": "hopper.flash_attn_interface.flash_attn_with_kvcache(qv, page_table)",
        "status": "disabled_on_b200_hopper_mla_path_sm90_only",
        "supported_attention": {"MLA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "flashinfer_trtllm_mla_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla",
        "status": "implemented_if_flashinfer_trtllm_mla_is_available_bf16_only_in_tested_b200_package",
        "supported_attention": {"MLA"},
        "supported_kv_dtypes": {"bf16"},
        "supported_page_sizes": {64, 128},
    },
    {
        "name": "vllm_flash",
        "layer": "framework_benchmark",
        "enabled": False,
        "kernel_path": "vLLM attention_benchmarks backend=flash",
        "status": "disabled_by_default_requires_model_access_framework_level",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {16, 32, 64, 128},
    },
    {
        "name": "vllm_flashinfer",
        "layer": "framework_benchmark",
        "enabled": False,
        "kernel_path": "vLLM attention_benchmarks backend=flashinfer",
        "status": "disabled_by_default_requires_model_access_framework_level",
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
    qk_nope_head_dim: int = 128,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
) -> dict:
    if attention == "MLA":
        if num_kv_heads is None:
            num_kv_heads = 1
        if kv_lora_rank is None:
            kv_lora_rank = 512 if head_dim_v is None else head_dim_v
        if qk_rope_head_dim is None:
            qk_rope_head_dim = head_dim
        mla_suffix = ""
        if (
            num_q_heads != 128
            or qk_nope_head_dim != 128
            or kv_lora_rank != 512
            or qk_rope_head_dim != 64
            or (head_dim_v is not None and head_dim_v != 512)
        ):
            mla_suffix = f"_qh{num_q_heads}_nope{qk_nope_head_dim}_r{kv_lora_rank}_rope{qk_rope_head_dim}_v{512 if head_dim_v is None else head_dim_v}"
        item = {
            "id": f"mla_{kv_dtype}_b{batch_size}_ctx{ctx_label(context_len)}_p{page_size}{mla_suffix}",
            "attention": attention,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "head_dim_v": 512 if head_dim_v is None else head_dim_v,
            "qk_nope_head_dim": qk_nope_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_dtype": kv_dtype,
            "page_size": page_size,
            "batch_size": batch_size,
            "context_len": context_len,
        }
        return item

    if num_kv_heads is None:
        num_kv_heads = {"MHA": num_q_heads, "GQA": 8, "MQA": 1}[attention]
    default_num_kv_heads = {"MHA": num_q_heads, "GQA": 8, "MQA": 1}[attention]
    shape_suffix = ""
    if num_q_heads != 32 or num_kv_heads != default_num_kv_heads or head_dim != 128:
        shape_suffix = f"_qh{num_q_heads}_kvh{num_kv_heads}_hd{head_dim}"
    return {
        "id": f"{attention.lower()}_{kv_dtype}_b{batch_size}_ctx{ctx_label(context_len)}_p{page_size}{shape_suffix}",
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
