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

    # GQA dtype comparison at fixed Blackwell-friendly page size.
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64),
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
