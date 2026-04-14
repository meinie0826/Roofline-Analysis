from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .baseline_fa4 import baseline_fa4_forward
from .common import AttentionConfig, available_backends
from .reference import (
    causal_attention_blocked_reference,
    causal_attention_online_reference,
    causal_attention_reference,
)
from .future import stage4_forward, stage5_forward
from .stage0_naive import stage0_forward
from .stage3_blocked import stage3_forward


StageFn = Callable


@dataclass(frozen=True)
class StageDefinition:
    name: str
    description: str
    implementation: StageFn
    backend: str


STAGES: dict[str, StageDefinition] = {
    "reference": StageDefinition(
        name="reference",
        description="PyTorch reference: full score materialization + causal mask + softmax + PV.",
        implementation=causal_attention_reference,
        backend="torch",
    ),
    "stage0": StageDefinition(
        name="stage0",
        description="Naive CuTe stage: CTA-per-row, explicit scores, no fusion.",
        implementation=stage0_forward,
        backend="cute-or-torch-fallback",
    ),
    "stage1": StageDefinition(
        name="stage1",
        description="PyTorch online-softmax reference for validating running max/sum math.",
        implementation=causal_attention_online_reference,
        backend="torch",
    ),
    "stage2": StageDefinition(
        name="stage2",
        description="PyTorch blocked reference for validating KV blocking before CuTe tiling.",
        implementation=causal_attention_blocked_reference,
        backend="torch",
    ),
    "baseline_fa4": StageDefinition(
        name="baseline_fa4",
        description="Baseline only: flash-attention CuTe implementation for comparison, not our final path.",
        implementation=baseline_fa4_forward,
        backend="flash-attention-cute-baseline",
    ),
    "stage3": StageDefinition(
        name="stage3",
        description="Blocked + online softmax CuTe kernel (our own implementation).",
        implementation=stage3_forward,
        backend="own-cute-dsl",
    ),
    "stage4": StageDefinition(
        name="stage4",
        description="Reserved for our own pipelined MMA CuTe kernel.",
        implementation=stage4_forward,
        backend="own-cute-dsl",
    ),
    "stage5": StageDefinition(
        name="stage5",
        description="Final target: our own full causal FlashAttention-style CuTe kernel.",
        implementation=stage5_forward,
        backend="own-cute-dsl",
    ),
}


def get_stage(name: str) -> StageDefinition:
    try:
        return STAGES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STAGES))
        raise KeyError(f"Unknown stage '{name}'. Available stages: {available}") from exc


def run_stage(name: str, q, k, v, config: AttentionConfig | None = None):
    stage = get_stage(name)
    return stage.implementation(q, k, v, config or AttentionConfig())


def describe_stages() -> list[dict[str, str]]:
    backends = available_backends()
    return [
        {
            "name": stage.name,
            "backend": stage.backend,
            "description": stage.description,
            "torch": str(backends["torch"]),
            "cute": str(backends["cute"]),
            "fa4_repo": str(backends["fa4_repo"]),
        }
        for stage in STAGES.values()
    ]
