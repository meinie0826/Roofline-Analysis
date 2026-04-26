"""Optional SGLang/vLLM integration probes for reference and benchmark work.

The core megakernel tests deliberately do not import either framework.  Their
model paths depend on runtime metadata, paged KV cache managers, backend
selection, and optional compiled kernels.  This module provides lightweight
gates so external-reference tests can fail closed with a useful reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Iterable

from .common import MegakernelConfig


@dataclass(frozen=True)
class FrameworkStatus:
    """Import status for an optional external framework."""

    name: str
    available: bool
    version: str | None = None
    error: str | None = None


def probe_framework_import(name: str) -> FrameworkStatus:
    """Try importing `sglang` or `vllm` without making it a hard dependency."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - depends on local env
        return FrameworkStatus(name=name, available=False, error=f"{type(exc).__name__}: {exc}")

    return FrameworkStatus(
        name=name,
        available=True,
        version=getattr(module, "__version__", None),
    )


def probe_framework_imports(
    names: Iterable[str] = ("sglang", "vllm"),
) -> dict[str, FrameworkStatus]:
    """Return optional import status for all requested frameworks."""
    return {name: probe_framework_import(name) for name in names}


def validate_supported_external_config(
    config: MegakernelConfig,
    *,
    batch_size: int = 1,
    q_len: int = 1,
    num_kv_heads: int | None = None,
    rope_style: str = "gptj",
    paged_kv: bool = False,
    sliding_window: bool = False,
    attention_sinks: bool = False,
    quantized_kv: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    """Reject SGLang/vLLM branches that the current megakernel does not model.

    This is intended for optional external reference/benchmark tests.  It keeps
    comparisons honest by failing before a framework path silently exercises a
    feature outside the dense single-token MHA kernel we currently implement.
    """
    config.validate()
    kv_heads = config.num_heads if num_kv_heads is None else num_kv_heads

    unsupported: list[str] = []
    if batch_size != 1:
        unsupported.append("batch_size != 1")
    if q_len != 1:
        unsupported.append("q_len != 1")
    if kv_heads != config.num_heads:
        unsupported.append("GQA/MQA num_kv_heads != num_heads")
    if rope_style.lower() != "gptj":
        unsupported.append("non-GPT-J RoPE")
    if paged_kv:
        unsupported.append("paged KV cache")
    if sliding_window:
        unsupported.append("sliding-window attention")
    if attention_sinks:
        unsupported.append("attention sinks")
    if quantized_kv:
        unsupported.append("quantized KV cache")
    if tensor_parallel_size != 1:
        unsupported.append("tensor parallel size != 1")

    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"Unsupported external-reference config: {joined}.")


def external_reference_status(config: MegakernelConfig | None = None) -> str:
    """Human-readable status for CLI/debug use."""
    config = config or MegakernelConfig()
    try:
        validate_supported_external_config(config)
        config_status = "config supported"
    except ValueError as exc:
        config_status = str(exc)

    imports = probe_framework_imports()
    parts = [config_status]
    for status in imports.values():
        if status.available:
            suffix = f" {status.version}" if status.version else ""
            parts.append(f"{status.name}: available{suffix}")
        else:
            parts.append(f"{status.name}: unavailable ({status.error})")
    return "\n".join(parts)
