"""B200 framework/reference decode attention benchmark matrix.

This matrix is intentionally separate from the kernel-only SOTA matrix. It is
for dense PyTorch SDPA/cuDNN references and framework/native runners whose
measurement boundaries are not identical to the direct kernel wrappers.

External backends are enabled by setting command-template environment variables:

- ``TRTLLM_NATIVE_BENCH_CMD`` for TensorRT-LLM's native benchmark harness.
- ``SGLANG_BENCH_CMD`` for an SGLang serving/paged decode benchmark harness.

Templates receive these placeholders: ``{batch_size}``, ``{context_len}``,
``{num_q_heads}``, ``{num_kv_heads}``, ``{head_dim}``, ``{head_dim_v}``,
``{qk_nope_head_dim}``, ``{kv_lora_rank}``, ``{qk_rope_head_dim}``,
``{kv_dtype}``, ``{page_size}``, ``{warmup_steps}``, ``{repeat}``, and ``{output}``.
The command must write DecodeBench-compatible JSON to ``{output}``.
"""

from __future__ import annotations

import os

from matrix_b200 import DEFAULTS, workload


BACKENDS = [
    {
        "name": "torch_sdpa_cudnn",
        "layer": "framework_reference",
        "enabled": True,
        "kernel_path": "torch.nn.functional.scaled_dot_product_attention backend=cudnn",
        "status": "implemented_if_torch_cudnn_sdpa_is_available",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "torch_sdpa_flash",
        "layer": "framework_reference",
        "enabled": True,
        "kernel_path": "torch.nn.functional.scaled_dot_product_attention backend=flash",
        "status": "implemented_if_torch_flash_sdpa_is_available",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "torch_sdpa_auto",
        "layer": "framework_reference",
        "enabled": True,
        "kernel_path": "torch.nn.functional.scaled_dot_product_attention backend=auto",
        "status": "implemented_if_torch_cuda_is_available",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16"},
        "supported_page_sizes": {64},
    },
    {
        "name": "tensorrt_llm_native",
        "layer": "native_framework_benchmark",
        "enabled": bool(os.environ.get("TRTLLM_NATIVE_BENCH_CMD")),
        "kernel_path": "TensorRT-LLM native benchmark command",
        "status": "enabled_when_TRTLLM_NATIVE_BENCH_CMD_is_set",
        "command_template_env": "TRTLLM_NATIVE_BENCH_CMD",
        "supported_attention": {"MHA", "GQA", "MQA", "MLA"},
        "supported_kv_dtypes": {"bf16", "fp16", "fp8"},
        "supported_page_sizes": {64, 128},
    },
    {
        "name": "sglang_serving",
        "layer": "serving_framework_benchmark",
        "enabled": bool(os.environ.get("SGLANG_BENCH_CMD")),
        "kernel_path": "SGLang serving benchmark command",
        "status": "enabled_when_SGLANG_BENCH_CMD_is_set",
        "command_template_env": "SGLANG_BENCH_CMD",
        "supported_attention": {"MHA", "GQA", "MQA"},
        "supported_kv_dtypes": {"bf16", "fp16", "fp8"},
        "supported_page_sizes": {64, 128},
    },
]


WORKLOADS = [
    # Keep dense SDPA references moderate; large ctx128k shapes allocate huge dense KV tensors.
    workload("MHA", "bf16", batch_size=16, context_len=4096, page_size=64),
    workload("GQA", "bf16", batch_size=64, context_len=4096, page_size=64),
    workload("MQA", "bf16", batch_size=128, context_len=4096, page_size=64),
    workload("GQA", "fp16", batch_size=64, context_len=4096, page_size=64),
    workload("GQA", "bf16", batch_size=32, context_len=8192, page_size=64),
    workload("GQA", "bf16", batch_size=8, context_len=32768, page_size=64),
    # Native/framework runners can cover FP8 and MLA when their command templates support it.
    workload("GQA", "fp8", batch_size=64, context_len=4096, page_size=64),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=64),
    workload("GQA", "fp8", batch_size=32, context_len=32768, page_size=128),
    workload("MLA", "bf16", batch_size=64, context_len=4096, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
    workload("MLA", "bf16", batch_size=32, context_len=32768, page_size=64, num_q_heads=128, head_dim=64, head_dim_v=512),
]


CONFIG = {
    "metadata": {
        "name": "decode-attn-b200-framework-reference",
        "gpu": "B200",
        "goal": "Compare PyTorch SDPA/cuDNN references plus TensorRT-LLM native and SGLang external framework runners.",
    },
    "defaults": DEFAULTS,
    "backends": BACKENDS,
    "workloads": WORKLOADS,
}
