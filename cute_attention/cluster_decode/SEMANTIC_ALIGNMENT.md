# Megakernel Semantic Alignment

This note tracks whether `cluster_decode` is computing the same decode-layer
semantics as ClusterFusion, SGLang, and vLLM.

## Current verdict

The CuTeDSL megakernel currently matches its local PyTorch reference and the
CTA-cluster ownership pattern from ClusterFusion, but it is not yet an
end-to-end semantic match for the upstream decode layer.

The main gap is attention over the current token:

- `megakernel_reference_forward` computes `k_new` and `v_new`, but attention is
  only over the input `k_cache` and `v_cache`.
- `cluster_megakernel_forward` mirrors that behavior: it writes `k_out/v_out`,
  then Stage 3 loops over `k_cache[0:seq_len]` only.
- ClusterFusion's kernels write the current token K/V and then explicitly add
  the current token to the online-softmax attention state.
- SGLang and vLLM pass current-token `q/k/v` into their attention layer; that
  layer stores current K/V in the KV cache and performs attention using the
  decode metadata/cache that includes the current token.

So, with the same caller contract as upstream frameworks, our current output is
missing the current token's own K/V contribution.

## Stage-by-stage comparison

| Stage | `cluster_decode` today | ClusterFusion upstream | SGLang / vLLM |
|---|---|---|---|
| RMSNorm | Yes, input RMSNorm before attention projection | Yes | Yes, decoder layer input RMSNorm before self-attn |
| QKV projection | Yes, packed `[W_q; W_k; W_v]` | Yes | Yes, packed `QKVParallelLinear` |
| RoPE | GPT-J/interleaved only | GPT-J in `kernel.cuh`; Neox in `kernel_batch_sglang.cuh` | Neox by default for Llama, GPT-J only when configured |
| KV output/update | Returns `k_new/v_new`; caller may store later | Kernel emits or stores current K/V | Attention path stores current K/V into paged cache |
| Attention domain | Existing dense cache only | Existing cache plus current token | Paged cache/current decode metadata including current token |
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
- `cluster_decode/megakernel_reference.py` currently says the opposite in code:
  "seq_len already contains the cached tokens; we do NOT append here".

## What must change for semantic equivalence

1. Define the caller contract to match ClusterFusion/SGLang/vLLM decode:
   `k_cache/v_cache` contain previous tokens only; current K/V are produced by
   this kernel and must also participate in this token's attention.
2. Update the PyTorch reference first:
   concatenate or otherwise include `k_new/v_new` in Stage 3 attention.
3. Update the CuTeDSL megakernel:
   after scanning the cache slice, fold `(Q_rot dot K_new, V_new)` into the
   online-softmax state once per head, matching ClusterFusion's current-token
   epilogue.
4. Add a regression test that compares old-cache-only versus append-current
   behavior and proves the kernel matches the append-current reference.
5. Add Neox RoPE mode before claiming equivalence with default SGLang/vLLM
   Llama paths. GPT-J mode remains useful for the current ClusterFusion
   non-SGLang test path.

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
