"""Stage18: SM90-oriented experimental entrypoint.

This stage intentionally reuses the current warp-specialized multistage
backend from stage17, but gives us an independent stage slot for subsequent
SM90-specific iteration without destabilizing stage17.

Current scope:
  - fixed 256-thread producer/consumer schedule
  - multistage K/V staging
  - dedicated autotune entrypoint

Future scope:
  - swap in more SM90-native features such as stronger producer/consumer
    scheduling, warpgroup-oriented organization, or newer pipeline structure.
"""

from __future__ import annotations

from dataclasses import replace

from .common import AttentionConfig
from .stage17_multistage import autotune_stage17_config, stage17_forward


def autotune_stage18_config(q, k, v, config: AttentionConfig | None = None) -> AttentionConfig:
    config = config or AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256, num_stages_kv=(config.num_stages_kv or 3))
    return autotune_stage17_config(q, k, v, tuned)


def stage18_forward(q, k, v, config: AttentionConfig | None = None):
    config = config or AttentionConfig(block_m=64, block_n=64, num_threads=256, num_stages_kv=3)
    tuned = replace(config, num_threads=256, num_stages_kv=(config.num_stages_kv or 3))
    if tuned.autotune:
        tuned = autotune_stage18_config(q, k, v, tuned)
    return stage17_forward(q, k, v, replace(tuned, autotune=False, num_threads=256))
