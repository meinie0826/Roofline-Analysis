from dataclasses import asdict, dataclass
from typing import Any


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
    factory_config: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_factory_candidate(
    name: str,
    *,
    tile_shape: tuple[int, int, int] = (256, 256, 64),
    ab_stages: int = 4,
    epilogue_warps: int = 4,
    use_tma_store: bool = True,
) -> GemmCandidate:
    return GemmCandidate(
        name=name,
        variant="configurable",
        ab_stages=ab_stages,
        use_tma_store=use_tma_store,
        tile_shape=tile_shape,
        epilogue_warps=epilogue_warps,
        threads_per_cta=32 * (epilogue_warps + 2),
        factory_config={
            "name": name,
            "tile_m": tile_shape[0],
            "tile_n": tile_shape[1],
            "tile_k": tile_shape[2],
            "ab_stages": ab_stages,
            "epi_warps": epilogue_warps,
            "use_tma_store": use_tma_store,
        },
    )


def make_joint_candidates() -> tuple[GemmCandidate, ...]:
    candidates = []
    for tile_shape in ((256, 256, 64), (256, 256, 128)):
        for ab_stages in (2, 3, 4, 6):
            for epilogue_warps in (3, 4, 5):
                for use_tma_store in (False, True):
                    store_suffix = "store" if use_tma_store else "rmem_store"
                    name = (
                        f"cfg_tile{tile_shape[0]}x{tile_shape[1]}x{tile_shape[2]}"
                        f"_ab{ab_stages}_ws{epilogue_warps}epi_{store_suffix}"
                    )
                    candidates.append(
                        make_factory_candidate(
                            name,
                            tile_shape=tile_shape,
                            ab_stages=ab_stages,
                            epilogue_warps=epilogue_warps,
                            use_tma_store=use_tma_store,
                        )
                    )
    return tuple(candidates)


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

JOINT_FACTORY_SWEEP = make_joint_candidates()

CANDIDATE_GROUPS = {
    "ab-stage": AB_STAGE_SWEEP,
    "tma-store": TMA_STORE_SWEEP,
    "tile-shape": TILE_SHAPE_SWEEP,
    "warp-spec": WARP_SPEC_SWEEP,
    "joint": JOINT_FACTORY_SWEEP,
    "default": DEFAULT_AUTOTUNE_CANDIDATES,
}
