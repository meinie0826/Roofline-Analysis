from typing import Tuple

import torch


def make_inputs(mnk: Tuple[int, int, int], seed: int = 0):
    m, n, k = mnk
    torch.manual_seed(seed)
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int32).to(torch.float16)
    b = torch.randint(-2, 3, (n, k), device="cuda", dtype=torch.int32).to(torch.float16)
    return a, b


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("mk,nk->mn", a.float(), b.float())


def check_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    torch.testing.assert_close(got.cpu(), ref.cpu(), atol=atol, rtol=rtol)
