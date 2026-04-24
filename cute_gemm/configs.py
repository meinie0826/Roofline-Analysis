from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GemmCandidate:
    name: str
    variant: str
    ab_stages: int
    use_tma_store: bool = False
    tma_warps: int = 1
    mma_warps: int = 1
    epilogue_warps: int = 4
    threads_per_cta: int = 192

    def to_dict(self) -> dict:
        return asdict(self)


AB_STAGE_SWEEP = (
    GemmCandidate("ab1", "2cta_tma_nopipeline", 1),
    GemmCandidate("ab2", "2cta_tma_2stage", 2),
    GemmCandidate("ab3", "2cta_tma_3stage", 3),
    GemmCandidate("ab4", "2cta_tma_pipeline", 4),
    GemmCandidate("ab6", "2cta_tma_6stage", 6),
)

TMA_STORE_SWEEP = (
    GemmCandidate("ab4", "2cta_tma_pipeline", 4),
    GemmCandidate("ab4_tma_store", "2cta_tma_pipeline_tma_store", 4, True),
)

DEFAULT_AUTOTUNE_CANDIDATES = AB_STAGE_SWEEP + (
    GemmCandidate("ab4_tma_store", "2cta_tma_pipeline_tma_store", 4, True),
)

CANDIDATE_GROUPS = {
    "ab-stage": AB_STAGE_SWEEP,
    "tma-store": TMA_STORE_SWEEP,
    "default": DEFAULT_AUTOTUNE_CANDIDATES,
}
