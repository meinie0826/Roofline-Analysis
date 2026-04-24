"""B200 MLA-only decode benchmark search space."""

DEFAULTS = {
    "decode_steps": 1,
    "warmup_steps": 20,
    "repeat": 100,
}

BACKENDS = [
    {
        "name": "flashmla_decode",
        "layer": "kernel",
        "enabled": True,
        "kernel_path": "flash_mla.flash_mla_with_kvcache",
        "status": "implemented_if_flashmla_is_installed",
    },
]


def workload(
    kv_dtype: str,
    batch_size: int,
    context_len: int,
    page_size: int = 64,
    num_q_heads: int = 128,
    num_kv_heads: int = 1,
    head_dim: int = 64,
    head_dim_v: int = 512,
) -> dict:
    return {
        "id": f"mla_{kv_dtype}_b{batch_size}_ctx{context_len // 1024}k_p{page_size}",
        "attention": "MLA",
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "head_dim_v": head_dim_v,
        "kv_dtype": kv_dtype,
        "page_size": page_size,
        "batch_size": batch_size,
        "context_len": context_len,
    }


WORKLOADS = [
    workload("bf16", batch_size=64, context_len=4096),
    workload("bf16", batch_size=32, context_len=32768),
]

CONFIG = {
    "metadata": {
        "name": "decode-attn-kernel-b200-mla",
        "gpu": "B200",
        "goal": "Benchmark MLA-only decode kernels on Blackwell/B200.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
