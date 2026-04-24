# DecodeBench backend notes

## Enabled in B200 kernel matrices

- `flashinfer_paged_decode`: general FlashInfer paged decode; broad BF16/FP16/FP8 coverage, but current FlashInfer 0.6.8 rejects MQA group size 32 in this path.
- `flashinfer_trtllm_decode`: FlashInfer wrapper around TRTLLM-GEN/XQA-style decode; strong B200 path, currently restricted to `page_size=64` because `page_size=128` misses TRTLLM-GEN kernels in the tested package.
- `flashattn_kvcache`: Tri Dao FlashAttention, preferring FA4 CuTE paged-varlen (`flash_attn.cute.interface.flash_attn_varlen_func`) with `page_table`.
- `vllm_paged_decode`: direct vLLM paged attention op; useful baseline, limited to `page_size={16,32}` and BF16/FP16.
- `flashinfer_trtllm_mla_decode`: FlashInfer TRTLLM-GEN MLA API for B200-capable MLA decode.

## Implemented but disabled by default on B200

- `flashmla_decode`: DeepSeek FlashMLA dense decode; installed successfully, but dense decode reports SM90a-only on B200.
- `flashattn_mla_decode`: Tri Dao Hopper `flash_attn_with_kvcache(qv=..., page_table=...)` MLA path; kept for Hopper/SM90 comparison, not a B200 default.
- `vllm_flash` / `vllm_flashinfer`: vLLM framework benchmark paths; useful for framework-level studies, but disabled by default because they may require gated model access and are not pure kernel timings.

## Potential future additions

- TensorRT-LLM native benchmark path, to avoid FlashInfer wrapper limitations and expose TRTLLM-GEN kernel selection more directly.
- cuDNN/PyTorch SDPA as framework/reference baselines, not SOTA paged-KV kernel baselines.
- SGLang end-to-end serving measurements, kept separate from kernel-only matrices.
