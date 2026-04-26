# SGLang External Reference and Benchmarks

This note records what it would take to use SGLang directly as a correctness
reference and benchmark baseline.

## Import status in this checkout

Probed from the repository root:

```bash
PYTHONPATH=3rd/sglang/python python3 -c "import sglang"
```

This currently fails in this local Python environment with:

```text
ModuleNotFoundError: No module named 'numpy'
```

## Can they be used as references?

Yes, but only behind an optional dependency gate and only for a narrow supported
subset at first.

The full framework attention layers are not plain functions. They depend on
runtime metadata, KV-cache managers, paged cache layouts, backend selection, and
often compiled custom ops. A robust external reference should therefore:

1. Try importing the framework and skip cleanly if dependencies are missing.
2. Instantiate only a minimal Llama attention path or use framework utility
   functions where possible.
3. Restrict configs to the subset our megakernel implements.
4. Compare both correctness and latency only after warmup, because framework
   backends may compile or initialize caches on first use.

## Supported subset for initial comparison

The current megakernel supports:

- single-token decode (`q_len = 1`),
- one batch item,
- dense previous KV cache,
- MHA only (`num_heads == num_kv_heads`),
- fp16/bf16-like model tensors with fp32 accumulation,
- GPT-J/interleaved RoPE,
- no quantized KV cache,
- no sliding-window attention,
- no attention sinks,
- no tensor-parallel all-reduce inside the op.

For ClusterFusion's non-SGLang Llama path, GPT-J/interleaved RoPE is a good
match. For default SGLang Llama paths, Neox-style RoPE is the important
remaining mismatch.

## Branches to explicitly reject for now

An external correctness/benchmark harness should refuse or skip:

- `num_key_value_heads != num_attention_heads` (GQA/MQA),
- paged KV cache as the input contract unless an adapter converts it to dense,
- `rope_is_neox_style=True` until the CuTeDSL kernel has a Neox path,
- partial rotary dimensions,
- sliding-window or hybrid attention layers,
- attention sinks,
- quantized KV cache (`fp8`, `nvfp4`, etc.),
- tensor parallel world size greater than 1,
- non-causal or encoder-only attention,
- multimodal/mrope variants.

## Recommended harness shape

The repository now has an optional reference module. Normal tests can import it
without requiring either framework; framework-specific tests skip when imports
are unavailable:

```text
cluster_decode/external_reference.py
  probe_sglang_import()
  validate_supported_external_config(...)
  sglang_megakernel_reference_forward(...)
  external_reference_status(...)
```

`sglang_megakernel_reference_forward(...)` uses SGLang's GPT-J/interleaved RoPE
implementation and instantiates `sglang.srt.layers.radix_attention.RadixAttention`
with a minimal dense `ForwardBatch` adapter. It intentionally keeps RMSNorm,
packed QKV, and W_o in this harness so the weights/layout match the megakernel.

The next step is to add full framework runners behind this gate:

```text
  run_sglang_llama_attention_reference(...) # skipped if unavailable/unsupported
  benchmark_external_vs_megakernel(...)
```

Core `pytest cluster_decode/tests/` should stay independent of SGLang.
The optional reference checks live in:

```bash
python3 -m pytest cluster_decode/tests/test_external_reference.py -v
```

That keeps the fast correctness suite stable while still allowing a stronger
SGLang-level check on machines with the full dependencies and compatible GPU
backend installed.
