#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from benchmark import get_stage_metadata, parse_stage_list


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
        "baseline_fa4",
        "baseline_sdpa",
    ]


def test_stage_metadata_marks_autotune_and_multistage_coverage():
    rows = {row["stage"]: row for row in get_stage_metadata()}
    assert rows["stage12"]["autotune"] == "True"
    assert rows["stage13"]["multistage"] == "True"
    assert rows["stage17"]["autotune"] == "True"
    assert "num_threads" in rows["stage17"]["tuning_axes"]
    assert "independent" in rows["stage17"]["notes"]
