"""B200 decode attention kernel benchmark search space."""

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
        "supported_workload_ids": {
            "mha_bf16_b16_ctx4k_p64",
            "gqa_bf16_b64_ctx4k_p64",
            "gqa_fp8_b64_ctx4k_p64",
            "gqa_fp8_b64_ctx4k_p128",
            "gqa_fp8_b32_ctx32k_p64",
            "gqa_fp8_b32_ctx32k_p128",
        },
    },
    {
        "name": "flashinfer_trtllm_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
        "status": "implemented",
        "supported_workload_ids": {
            "mha_bf16_b16_ctx4k_p64",
            "gqa_bf16_b64_ctx4k_p64",
            "gqa_fp8_b64_ctx4k_p64",
            "gqa_fp8_b32_ctx32k_p64",
            "mqa_fp8_b128_ctx8k_p64",
        },
    },
    {
        "name": "flashattn_kvcache",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flash_attn.flash_attn_with_kvcache",
        "status": "implemented_if_flash_attention_is_installed",
        "supported_workload_ids": {
            "mha_bf16_b16_ctx4k_p64",
            "gqa_bf16_b64_ctx4k_p64",
        },
    },
    {
        "name": "vllm_paged_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "vllm._custom_ops.paged_attention_v2",
        "status": "implemented_if_vllm_is_installed",
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {16, 32},
        "supported_workload_ids": {
            "mha_bf16_b16_ctx4k_p32",
            "gqa_bf16_b64_ctx4k_p32",
        },
    },
    {
        "name": "vllm_flash",
        "layer": "kernel_or_framework",
        "enabled": False,
        "kernel_path": "vLLM attention_benchmarks backend=flash",
        "status": "disabled_because_not_kernel_only",
        "supported_kv_dtypes": {"bf16", "fp16"},
    },
    {
        "name": "vllm_flashinfer",
        "layer": "kernel_or_framework",
        "enabled": False,
        "kernel_path": "vLLM attention_benchmarks backend=flashinfer",
        "status": "disabled_because_not_kernel_only",
    },
]


def workload(
    attention: str,
    kv_dtype: str,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_q_heads: int = 32,
    num_kv_heads: int | None = None,
    head_dim: int = 128,
) -> dict:
    if num_kv_heads is None:
        num_kv_heads = {"MHA": num_q_heads, "GQA": 8, "MQA": 1}[attention]
    return {
        "id": f"{attention.lower()}_{kv_dtype}_b{batch_size}_ctx{context_len // 1024}k_p{page_size}",
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
    # Baseline: standard MHA, BF16 KV.
    workload("MHA", "bf16", batch_size=16, context_len=4096, page_size=64),
    workload("MHA", "bf16", batch_size=16, context_len=4096, page_size=32),

    # GQA dtype comparison at fixed Blackwell-friendly page size.
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=32),
    workload("GQA", "fp8", batch_size=64, context_len=4096, page_size=64),

    # Page-size sensitivity for paged KV decode.
    workload("GQA", "fp8", batch_size=64, context_len=4096, page_size=128),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=64),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=128),

    # MQA specialization stress case.
    workload("MQA", "fp8", batch_size=128, context_len=8192, page_size=64),
]

CONFIG = {
    "metadata": {
        "name": "decode-attn-kernel-b200-mvp",
        "gpu": "B200",
        "goal": "Compare SOTA open-source decode attention kernel paths on Blackwell/B200 and explain winner/loser causes.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
