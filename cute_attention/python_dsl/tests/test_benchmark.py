#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from benchmark import parse_stage_list


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
        "baseline_fa4",
        "baseline_sdpa",
    ]
