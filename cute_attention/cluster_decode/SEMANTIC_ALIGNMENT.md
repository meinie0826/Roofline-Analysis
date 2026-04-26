# Megakernel Semantic Alignment

This note tracks whether `cluster_decode` is computing the same decode-layer
semantics as ClusterFusion, SGLang, and vLLM.

## Current verdict

The CuTeDSL megakernel now matches its local PyTorch reference, the
CTA-cluster ownership pattern from ClusterFusion, and the key decode-layer
contract that current-token K/V participate in the same attention step.

The previous semantic gap was attention over the current token:

- `megakernel_reference_forward` now treats `k_cache/v_cache` as previous
  tokens and appends current `k_new/v_new` to the attention domain.
- `cluster_megakernel_forward` mirrors that behavior: it writes `k_out/v_out`,
  scans `k_cache[0:seq_len]`, then folds current K/V into the online-softmax
  state on CTA rank 0 before cluster reductions.
- ClusterFusion's kernels write the current token K/V and then explicitly add
  the current token to the online-softmax attention state.
- SGLang and vLLM pass current-token `q/k/v` into their attention layer; that
  layer stores current K/V in the KV cache and performs attention using the
  decode metadata/cache that includes the current token.

So the core single-token dense-cache MHA attention semantics are aligned. The
remaining gaps are framework interface/generalization gaps, not the basic
decode math for this scoped path.

## Stage-by-stage comparison

| Stage | `cluster_decode` today | ClusterFusion upstream | SGLang / vLLM |
|---|---|---|---|
| RMSNorm | Yes, input RMSNorm before attention projection | Yes | Yes, decoder layer input RMSNorm before self-attn |
| QKV projection | Yes, packed `[W_q; W_k; W_v]` | Yes | Yes, packed `QKVParallelLinear` |
| RoPE | GPT-J/interleaved only | GPT-J in `kernel.cuh`; Neox in `kernel_batch_sglang.cuh` | Neox by default for Llama, GPT-J only when configured |
| KV output/update | Returns `k_new/v_new`; caller may store later | Kernel emits or stores current K/V | Attention path stores current K/V into paged cache |
| Attention domain | Previous dense cache plus current token | Previous cache plus current token | Paged cache/current decode metadata including current token |
| Attention heads | MHA only (`num_heads == num_kv_heads`) | Llama-2 style MHA in the single-token kernel; SGLang batch kernel is paged | MHA/GQA/MQA supported |
| Output projection | Yes, per-head contribution summed | Yes | Yes |

## Evidence from code

- ClusterFusion Python wrapper passes only previous cache to the fused kernel:
  `self.cache_k[:bsz, :start_pos]`, then stores returned `xk/xv` at
  `start_pos`.
- ClusterFusion CUDA still processes current K/V inside attention after cache
  reads (`Process KV of current token`).
- SGLang `LlamaAttention.forward` computes packed qkv, applies RoPE, then calls
  `RadixAttention(q, k, v, forward_batch)`.
- vLLM `LlamaAttention.forward` computes packed qkv, applies RoPE, then calls
  `Attention(q, k, v)`; `Attention.forward` updates the KV cache before calling
  the backend when the backend does not fuse the update itself.
- `cluster_decode/megakernel_reference.py` now appends current K/V to the
  attention tensors used by the PyTorch reference.

## Remaining work for broader framework equivalence

1. Add Neox RoPE mode before claiming equivalence with default SGLang/vLLM
   Llama paths. GPT-J mode remains useful for the current ClusterFusion
   non-SGLang test path.
2. Add optional GQA/MQA head mapping (`num_kv_heads < num_heads`).
3. Add a paged-KV wrapper or adapter before using framework caches directly.
4. Add an external reference/benchmark harness that imports SGLang/vLLM only
   when their dependencies are available and the requested config is inside
   this kernel's supported subset.

## Scope that remains intentionally narrower than SGLang/vLLM

Even after adding current-token attention, this megakernel is still a focused
single-token dense-cache MHA kernel:

- no paged KV-cache interface yet,
- no batching,
- no GQA/MQA head mapping,
- no tensor-parallel all-reduce,
- no sliding-window or attention sinks,
- no quantized KV cache.

That is still meaningful for a Llama-2-7B-like dense MHA decode microkernel,
but it should not be described as fully equivalent to general SGLang/vLLM Llama
attention until those interfaces are implemented or explicitly ruled out.
