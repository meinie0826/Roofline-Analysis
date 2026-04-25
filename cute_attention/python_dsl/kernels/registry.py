from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .baseline_fa4 import baseline_fa4_forward
from .baseline_sdpa import baseline_sdpa_forward
from .common import AttentionConfig, available_backends
from .stage1_fa2 import stage1_forward
from .stage4_mma import stage4_forward
from .stage5_pipeline import stage5_forward
from .stage6_q16 import stage6_forward
from .stage7_score16 import stage7_forward
from .stage8_noscore import stage8_forward
from .stage9_threadgroup import stage9_forward
from .stage10_hybrid import stage10_forward
from .stage11_mma import stage11_forward
from .stage12_pipeline2 import stage12_forward
from .stage13_multistage import stage13_forward
from .stage14_warpspec import stage14_forward
from .stage15_sm90style import stage15_forward
from .stage16_multistage import stage16_forward
from .stage17_multistage import stage17_forward
from .stage18_sm90_features import stage18_forward
from .stage19_warpgroup import stage19_forward
from .stage20_warpspec import stage20_forward
from .stage21_state_machine import stage21_forward
from .stage22_tma import stage22_forward
from .stage0_naive import stage0_forward
from .stage2_colblocked import stage2_forward
from .stage3_blocked import stage3_forward


StageFn = Callable


@dataclass(frozen=True)
class StageDefinition:
    name: str
    description: str
    implementation: StageFn
    backend: str


STAGES: dict[str, StageDefinition] = {
    "stage0": StageDefinition(
        name="stage0",
        description="Naive CuTe stage: CTA-per-row, explicit scores, no fusion.",
        implementation=stage0_forward,
        backend="cute-or-torch-fallback",
    ),
    "stage1": StageDefinition(
        name="stage1",
        description="Our own CuTe FA2-style kernel: blocked causal attention with online softmax.",
        implementation=stage1_forward,
        backend="own-cute-dsl",
    ),
    "baseline_fa4": StageDefinition(
        name="baseline_fa4",
        description="Baseline only: flash-attention CuTe implementation for comparison, not our final path.",
        implementation=baseline_fa4_forward,
        backend="flash-attention-cute-baseline",
    ),
    "baseline_sdpa": StageDefinition(
        name="baseline_sdpa",
        description="PyTorch SDPA baseline (causal, qkv only).",
        implementation=baseline_sdpa_forward,
        backend="torch-sdpa-baseline",
    ),
    "stage2": StageDefinition(
        name="stage2",
        description="Column-blocked CuTe kernel: blocked along head_dim with per-tile dot-product reduction.",
        implementation=stage2_forward,
        backend="own-cute-dsl",
    ),
    "stage3": StageDefinition(
        name="stage3",
        description="Blocked + online softmax CuTe kernel (our own implementation).",
        implementation=stage3_forward,
        backend="own-cute-dsl",
    ),
    "stage4": StageDefinition(
        name="stage4",
        description="Our own CuTe stage: stage1 math + K-tile shared-memory staging.",
        implementation=stage4_forward,
        backend="own-cute-dsl",
    ),
    "stage5": StageDefinition(
        name="stage5",
        description="Our own CuTe stage: stage4 math + K/V shared-memory staging.",
        implementation=stage5_forward,
        backend="own-cute-dsl",
    ),
    "stage6": StageDefinition(
        name="stage6",
        description="Our own CuTe stage: stage5 math + Q shared-memory fp16 staging.",
        implementation=stage6_forward,
        backend="own-cute-dsl",
    ),
    "stage7": StageDefinition(
        name="stage7",
        description="Our own CuTe stage: stage6 math + score/prob shared-memory fp16 staging.",
        implementation=stage7_forward,
        backend="own-cute-dsl",
    ),
    "stage8": StageDefinition(
        name="stage8",
        description="Our own CuTe stage: stage6 math + remove score/prob tile (two-pass per block).",
        implementation=stage8_forward,
        backend="own-cute-dsl",
    ),
    "stage9": StageDefinition(
        name="stage9",
        description="Our own CuTe stage: stage6 math + per-row thread-group reductions.",
        implementation=stage9_forward,
        backend="own-cute-dsl",
    ),
    "stage10": StageDefinition(
        name="stage10",
        description="Our own CuTe stage: lane0 row-reduction + thread-group acc update.",
        implementation=stage10_forward,
        backend="own-cute-dsl",
    ),
    "stage11": StageDefinition(
        name="stage11",
        description="Our own CuTe stage: Ampere MMA-based QK/PV mainloop.",
        implementation=stage11_forward,
        backend="own-cute-dsl",
    ),
    "stage12": StageDefinition(
        name="stage12",
        description="Our own CuTe stage: stage11 + double-buffered K/V cp.async pipeline.",
        implementation=stage12_forward,
        backend="own-cute-dsl",
    ),
    "stage13": StageDefinition(
        name="stage13",
        description="Our own CuTe stage: multistage/autotune entrypoint for stage11-stage12 MMA pipeline family.",
        implementation=stage13_forward,
        backend="own-cute-dsl",
    ),
    "stage14": StageDefinition(
        name="stage14",
        description="Our own CuTe stage: Ampere-style MMA + producer/consumer warp specialization.",
        implementation=stage14_forward,
        backend="own-cute-dsl",
    ),
    "stage15": StageDefinition(
        name="stage15",
        description="Our own CuTe stage: SM90-style 256-thread CTA with 4 producer warps and 4 consumer warps.",
        implementation=stage15_forward,
        backend="own-cute-dsl",
    ),
    "stage16": StageDefinition(
        name="stage16",
        description="Our own CuTe stage: stage15 + double-buffered K/V pipeline (warp-specialized, wait_group 1).",
        implementation=stage16_forward,
        backend="own-cute-dsl",
    ),
    "stage17": StageDefinition(
        name="stage17",
        description="Our own CuTe stage: dedicated stage17 entrypoint for deeper warp-specialized multistage K/V staging.",
        implementation=stage17_forward,
        backend="own-cute-dsl",
    ),
    "stage18": StageDefinition(
        name="stage18",
        description="Our own CuTe stage: SM90-oriented experimental backend for independent warp-specialized multistage evolution.",
        implementation=stage18_forward,
        backend="own-cute-dsl",
    ),
    "stage19": StageDefinition(
        name="stage19",
        description="Our own CuTe stage: warpgroup-layout experimental backend built on the stage18 multistage structure.",
        implementation=stage19_forward,
        backend="own-cute-dsl",
    ),
    "stage20": StageDefinition(
        name="stage20",
        description="Our own CuTe stage: aggressive warpspec backend with a circular-buffer steady-state mainloop.",
        implementation=stage20_forward,
        backend="own-cute-dsl",
    ),
    "stage21": StageDefinition(
        name="stage21",
        description="Our own CuTe stage: explicit producer/consumer state-machine backend built from the stage18 mainline.",
        implementation=stage21_forward,
        backend="own-cute-dsl",
    ),
    "stage22": StageDefinition(
        name="stage22",
        description="Stage ported from CUTLASS CuTeDSL Blackwell FMHA example, wrapped behind the local stage22 interface.",
        implementation=stage22_forward,
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
