from typing import Any, Tuple

import torch


def _to_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_name = getattr(dtype, "__name__", None) or str(dtype)
    if "Float16" in dtype_name or "float16" in dtype_name:
        return torch.float16
    if "Float32" in dtype_name or "float32" in dtype_name:
        return torch.float32
    if "BFloat16" in dtype_name or "bfloat16" in dtype_name:
        return torch.bfloat16
    raise TypeError(f"unsupported dtype for torch reference: {dtype!r}")


def make_inputs(mnk: Tuple[int, int, int], seed: int = 0):
    m, n, k = mnk
    torch.manual_seed(seed)
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int32).to(torch.float16)
    b = torch.randint(-2, 3, (n, k), device="cuda", dtype=torch.int32).to(torch.float16)
    return a, b


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch_reference_gemm_with_dtype(a, b, torch.float32)


def torch_reference_gemm_with_dtype(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out_dtype = _to_torch_dtype(out_dtype)
    out = torch.einsum("mk,nk->mn", a.float(), b.float())
    return out.to(out_dtype)


def torch_perf_gemm_with_dtype(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out_dtype = _to_torch_dtype(out_dtype)
    if out_dtype == torch.float16:
        return torch.mm(a, b.t())

    if out_dtype == torch.float32:
        try:
            return torch.mm(a, b.t(), out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            return torch.mm(a.float(), b.float().t())

    return torch.mm(a.float(), b.float().t()).to(out_dtype)


def make_torch_cublas_runner(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
):
    out_dtype = _to_torch_dtype(out_dtype)
    b_t = b.t()

    if out_dtype == torch.float16:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=out_dtype)

        def run() -> torch.Tensor:
            return torch.mm(a, b_t, out=out)

        return run

    if out_dtype == torch.float32:
        try:
            torch.mm(a, b_t, out_dtype=torch.float32)

            def run() -> torch.Tensor:
                return torch.mm(a, b_t, out_dtype=torch.float32)

            return run
        except (TypeError, RuntimeError):
            a_f32 = a.float()
            b_t_f32 = b_t.float()
            out = torch.empty(
                (a.shape[0], b.shape[0]), device=a.device, dtype=out_dtype
            )

            def run() -> torch.Tensor:
                return torch.mm(a_f32, b_t_f32, out=out)

            return run

    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=out_dtype)

    def run() -> torch.Tensor:
        out.copy_(torch.mm(a.float(), b_t.float()).to(out_dtype))
        return out

    return run


def torch_gemm_with_dtype(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch_reference_gemm_with_dtype(a, b, out_dtype)


def check_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    torch.testing.assert_close(got.cpu(), ref.cpu(), atol=atol, rtol=rtol)
