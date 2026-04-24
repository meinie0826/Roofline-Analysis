"""B200 MLA-only decode benchmark search space.

FlashMLA dense decode and Tri Dao's Hopper qv/page_table MLA path are kept in
this file but disabled for B200 because the currently tested kernels are SM90a
or Hopper-oriented. The B200 MLA default is FlashInfer's TRTLLM-GEN MLA API.
"""

DEFAULTS = {
    "decode_steps": 1,
    "warmup_steps": 20,
    "repeat": 100,
}

BACKENDS = [
    {
        "name": "flashinfer_trtllm_mla_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla",
        "status": "implemented_if_flashinfer_trtllm_mla_is_available",
        "supported_attention": {"MLA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {64, 128},
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
]


def ctx_label(context_len: int) -> str:
    return f"{context_len // 1024}k" if context_len % 1024 == 0 else str(context_len)


def workload(
    kv_dtype: str,
    batch_size: int,
    context_len: int,
    page_size: int = 64,
    num_q_heads: int = 128,
    num_kv_heads: int = 1,
    head_dim: int = 64,
    head_dim_v: int = 512,
    qk_nope_head_dim: int = 128,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
) -> dict:
    return {
        "id": f"mla_{kv_dtype}_b{batch_size}_ctx{ctx_label(context_len)}_p{page_size}",
        "attention": "MLA",
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "head_dim_v": head_dim_v,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "kv_dtype": kv_dtype,
        "page_size": page_size,
        "batch_size": batch_size,
        "context_len": context_len,
    }


WORKLOADS = [
    workload("bf16", batch_size=64, context_len=4096, page_size=64),
    workload("bf16", batch_size=32, context_len=32768, page_size=64),
    workload("bf16", batch_size=16, context_len=65536, page_size=64),
    workload("fp16", batch_size=64, context_len=4096, page_size=64),
]

CONFIG = {
    "metadata": {
        "name": "decode-attn-kernel-b200-mla",
        "gpu": "B200",
        "goal": "Benchmark B200-capable MLA decode kernels, with SM90-only MLA paths documented but disabled.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
