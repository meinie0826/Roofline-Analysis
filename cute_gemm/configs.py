from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GemmCandidate:
    name: str
    variant: str
    ab_stages: int
    use_tma_store: bool = False
    tile_shape: tuple[int, int, int] = (256, 256, 64)
    cluster_shape: tuple[int, int, int] = (2, 1, 1)
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

TILE_SHAPE_SWEEP = (
    GemmCandidate("tile256x256x64", "2cta_tma_pipeline_tma_store", 4, True),
    GemmCandidate(
        "tile256x256x128",
        "2cta_tma_pipeline_tma_store_tile256x256x128",
        4,
        True,
        tile_shape=(256, 256, 128),
    ),
)

WARP_SPEC_SWEEP = (
    GemmCandidate("ws_1tma_1mma_4epi", "2cta_tma_pipeline_tma_store", 4, True),
    GemmCandidate(
        "ws_1tma_1mma_3epi",
        "2cta_tma_pipeline_tma_store_ws3epi",
        4,
        True,
        epilogue_warps=3,
        threads_per_cta=160,
    ),
    GemmCandidate(
        "ws_1tma_1mma_5epi",
        "2cta_tma_pipeline_tma_store_ws5epi",
        4,
        True,
        epilogue_warps=5,
        threads_per_cta=224,
    ),
)

DEFAULT_AUTOTUNE_CANDIDATES = AB_STAGE_SWEEP + (
    GemmCandidate("ab4_tma_store", "2cta_tma_pipeline_tma_store", 4, True),
    GemmCandidate(
        "tile256x256x128",
        "2cta_tma_pipeline_tma_store_tile256x256x128",
        4,
        True,
        tile_shape=(256, 256, 128),
    ),
    GemmCandidate(
        "ws_1tma_1mma_3epi",
        "2cta_tma_pipeline_tma_store_ws3epi",
        4,
        True,
        epilogue_warps=3,
        threads_per_cta=160,
    ),
    GemmCandidate(
        "ws_1tma_1mma_5epi",
        "2cta_tma_pipeline_tma_store_ws5epi",
        4,
        True,
        epilogue_warps=5,
        threads_per_cta=224,
    ),
)

CANDIDATE_GROUPS = {
    "ab-stage": AB_STAGE_SWEEP,
    "tma-store": TMA_STORE_SWEEP,
    "tile-shape": TILE_SHAPE_SWEEP,
    "warp-spec": WARP_SPEC_SWEEP,
    "default": DEFAULT_AUTOTUNE_CANDIDATES,
}
