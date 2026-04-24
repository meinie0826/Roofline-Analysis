"""DecodeBench project notes.

Goal:
    Benchmark open-source SOTA decode attention kernels and explain why each
    workload's winner wins and why the losers are slower.

Current machine:
    B200 / Blackwell.

Current implemented backend:
    FlashInfer BatchDecodeWithPagedKVCacheWrapper.

File split:
    flashinfer_kernel.py contains tensor setup and kernel invocation.
    flashinfer_benchmark.py contains CLI, timing, and JSON output.

Third-party source layout:
    Put vendored benchmark dependencies under 3rd/.
    For vLLM attention benchmarks, the default expected path is:
    3rd/vllm/benchmarks/attention_benchmarks

Do not report model TPS/TPOT here. This is kernel-only measurement.
Use kernel latency and approximate KV-read bandwidth instead.
"""
