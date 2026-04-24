from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType


@dataclass(frozen=True)
class GemmConfig:
    name: str
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 64
    ab_stages: int = 4
    epi_warps: int = 4
    use_tma_store: bool = True

    @property
    def tile_shape(self) -> tuple[int, int, int]:
        return (self.tile_m, self.tile_n, self.tile_k)

    @property
    def threads_per_cta(self) -> int:
        return 32 * (self.epi_warps + 2)

    @property
    def epilogue_warp_ids(self) -> tuple[int, ...]:
        return tuple(range(self.epi_warps))

    @property
    def mma_warp_id(self) -> int:
        return self.epi_warps

    @property
    def tma_warp_id(self) -> int:
        return self.epi_warps + 1


def _template_path(use_tma_store: bool) -> Path:
    filename = (
        "mma_gemm_2cta_tma_pipeline_tma_store_cutedsl.py"
        if use_tma_store
        else "mma_gemm_2cta_tma_pipeline_cutedsl.py"
    )
    return Path(__file__).with_name(filename)


def _specialize_source(config: GemmConfig) -> str:
    source = _template_path(config.use_tma_store).read_text(encoding="utf-8")

    replacements = {
        "threads_per_cta = 192": f"threads_per_cta = {config.threads_per_cta}",
        "mma_tiler_mnk = (256, 256, 64)": f"mma_tiler_mnk = {config.tile_shape!r}",
        "ab_stages = 4": f"ab_stages = {config.ab_stages}",
        (
            "    epilogue_warp_ids = (0, 1, 2, 3)\n"
            "    mma_warp_id = 4\n"
            "    tma_warp_id = 5"
        ): (
            f"    epilogue_warp_ids = {config.epilogue_warp_ids!r}\n"
            f"    mma_warp_id = {config.mma_warp_id}\n"
            f"    tma_warp_id = {config.tma_warp_id}"
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise ValueError(f"template no longer contains expected snippet: {old!r}")
        source = source.replace(old, new)

    source = source.replace(
        '"variant": "2cta_tma_pipeline_tma_store"',
        f'"variant": "{config.name}"',
    )
    source = source.replace(
        '"variant": "2cta_tma_pipeline"',
        f'"variant": "{config.name}"',
    )
    return source


def _module_cache_path(config: GemmConfig) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z_]", "_", config.name)
    digest = hashlib.sha1(repr(config).encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(tempfile.gettempdir()) / "cute_gemm_configurable"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{safe_name}_{digest}.py"


def _write_module_source(config: GemmConfig) -> Path:
    path = _module_cache_path(config)
    source = _specialize_source(config)
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        path.write_text(source, encoding="utf-8")
    return path


@lru_cache(maxsize=None)
def make_module(config: GemmConfig) -> ModuleType:
    path = _write_module_source(config)
    module_name = f"cute_gemm_config_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.CONFIG = config
    return module
