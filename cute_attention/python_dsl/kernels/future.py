from __future__ import annotations


def _not_implemented(stage_name: str, detail: str):
    raise NotImplementedError(f"{stage_name} is reserved for our own CuTe DSL implementation. {detail}")


def stage3_forward(q, k, v, config=None):
    return _not_implemented(
        "stage3",
        "Next step: move blocked causal attention from the PyTorch reference path into a shared-memory CuTe kernel.",
    )


def stage4_forward(q, k, v, config=None):
    return _not_implemented(
        "stage4",
        "Next step: add pipelined GMEM->SMEM staging, warp specialization, and MMA-driven mainloop in our own kernel.",
    )


def stage5_forward(q, k, v, config=None):
    return _not_implemented(
        "stage5",
        "Final target: our own full causal FlashAttention-style CuTe kernel, without calling flash-attention's implementation.",
    )
