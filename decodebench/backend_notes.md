# DecodeBench backend notes

## Enabled in B200 kernel matrices

- `flashinfer_paged_decode`: general FlashInfer paged decode; broad BF16/FP16/FP8 coverage, but current FlashInfer 0.6.8 rejects MQA group size 32 in this path.
- `flashinfer_trtllm_decode`: FlashInfer wrapper around TRTLLM-GEN/XQA-style decode; strong B200 path, currently restricted to `page_size=64` because `page_size=128` misses TRTLLM-GEN kernels in the tested package.
- `flashattn_kvcache`: Tri Dao FlashAttention, preferring FA4 CuTE paged-varlen (`flash_attn.cute.interface.flash_attn_varlen_func`) with `page_table`.
- `vllm_paged_decode`: direct vLLM paged attention op; useful baseline, limited to `page_size={16,32}` and BF16/FP16.
- `flashinfer_trtllm_mla_decode`: FlashInfer TRTLLM-GEN MLA API for B200-capable MLA decode; current tested FlashInfer package is treated as BF16-only because FP16 MLA dispatch misses the TRTLLM-GEN kernel.

## Implemented but disabled by default on B200

- `flashmla_decode`: DeepSeek FlashMLA dense decode; installed successfully, but dense decode reports SM90a-only on B200.
- `flashattn_mla_decode`: Tri Dao Hopper `flash_attn_with_kvcache(qv=..., page_table=...)` MLA path; kept for Hopper/SM90 comparison, not a B200 default.
- `vllm_flash` / `vllm_flashinfer`: vLLM framework benchmark paths; useful for framework-level studies, but disabled by default because they may require gated model access and are not pure kernel timings.

## Potential future additions

## Framework/reference matrix

Use `decodebench/matrix_b200_framework.py` for non-kernel-only comparisons:

- `torch_sdpa_cudnn`: PyTorch SDPA forced to cuDNN attention when available.
- `torch_sdpa_flash`: PyTorch SDPA forced to FlashAttention backend when available.
- `torch_sdpa_auto`: PyTorch SDPA default dispatcher reference.
- `tensorrt_llm_native`: built-in TensorRT-LLM native `TrtllmAttentionWrapper` benchmark. It forces `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=0`, so it bypasses the FlashInfer TRTLLM-GEN wrapper dispatch path.
- `sglang_serving`: external SGLang command adapter for framework/serving measurements.

Example SDPA run:

```bash
python decodebench/run_matrix.py --config decodebench/matrix_b200_framework.py --execute --resume
```

SGLang is enabled only when its command template env var is set. The template must write DecodeBench-compatible JSON to `{output}` with at least `compare_latency_us` or `kernel_latency_p50_us`.

```bash
export SGLANG_BENCH_CMD='python /path/to/sglang_decode.py --batch-size {batch_size} --context-len {context_len} --num-q-heads {num_q_heads} --num-kv-heads {num_kv_heads} --head-dim {head_dim} --kv-dtype {kv_dtype} --page-size {page_size} --warmup {warmup_steps} --repeat {repeat} --output {output}'
python decodebench/run_matrix.py --config decodebench/matrix_b200_framework.py --execute --resume
```
