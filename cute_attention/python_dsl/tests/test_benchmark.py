#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from benchmark import _make_config_for_stage, get_stage_metadata, parse_stage_list
from kernels import AttentionConfig


def test_parse_stage_list_all_includes_new_ablation_stages():
    stages = parse_stage_list("all")
    assert stages == [
        "stage0",
        "stage1",
        "stage3",
        "stage4",
        "stage5",
        "stage6",
        "stage7",
        "stage8",
        "stage9",
        "stage10",
        "stage11",
        "stage12",
        "stage13",
        "stage14",
        "stage15",
        "stage16",
        "stage17",
        "stage18",
        "stage19",
        "baseline_fa4",
        "baseline_sdpa",
    ]


def test_stage_metadata_marks_autotune_and_multistage_coverage():
    rows = {row["stage"]: row for row in get_stage_metadata()}
    assert rows["stage12"]["autotune"] == "True"
    assert rows["stage13"]["multistage"] == "True"
    assert rows["stage17"]["autotune"] == "True"
    assert rows["stage17"]["tuning_axes"] == "block_m,block_n,num_stages_kv"
    assert "warp-specialized" in rows["stage17"]["notes"]
    assert rows["stage18"]["autotune"] == "True"
    assert rows["stage18"]["tuning_axes"] == "block_m,block_n,num_stages_kv"
    assert "SM90-oriented" in rows["stage18"]["notes"]
    assert rows["stage19"]["autotune"] == "True"
    assert rows["stage19"]["tuning_axes"] == "block_m,block_n,num_stages_kv"
    assert "warpgroup-layout" in rows["stage19"]["notes"]


def test_stage17_benchmark_uses_safe_warpspec_seed_config():
    config = _make_config_for_stage("stage17", AttentionConfig(block_m=64, block_n=128, num_threads=128))
    assert config.block_m == 64
    assert config.block_n == 64
    assert config.num_threads == 256
    assert config.num_stages_kv == 3


def test_stage18_benchmark_uses_safe_sm90_seed_config():
    config = _make_config_for_stage("stage18", AttentionConfig(block_m=64, block_n=128, num_threads=128))
    assert config.block_m == 64
    assert config.block_n == 64
    assert config.num_threads == 256
    assert config.num_stages_kv == 3


def test_stage19_benchmark_uses_safe_warpgroup_seed_config():
    config = _make_config_for_stage("stage19", AttentionConfig(block_m=64, block_n=128, num_threads=128))
    assert config.block_m == 64
    assert config.block_n == 64
    assert config.num_threads == 256
    assert config.num_stages_kv == 3
