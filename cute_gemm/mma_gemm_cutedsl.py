from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack


io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32

threads_per_cta = 128
mma_tiler_mnk = (128, 256, 64)


@cute.struct
class SharedStorage:
    mma_mbar_ptr: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(storage.mma_mbar_ptr.ptr, 1)
    cute.arch.mbarrier_init_fence()

    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        acc_dtype,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)

    sA_stage = sA[(None, None, None, 0)]
    sB_stage = sB[(None, None, None, 0)]

    mma_phase = cutlass.Int32(0)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    num_k_tiles = cute.size(gA, mode=[2])
    for k_tile_idx in cutlass.range(num_k_tiles):
        gA_tile = tCgA[(None, None, None, k_tile_idx)]
        gB_tile = tCgB[(None, None, None, k_tile_idx)]

        for i in cutlass.range(tidx, cute.size(sA_stage), threads_per_cta):
            sA_stage[i] = gA_tile[i]
        for i in cutlass.range(tidx, cute.size(sB_stage), threads_per_cta):
            sB_stage[i] = gB_tile[i]

        pipeline.sync()

        if warp_idx == 0:
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            tcgen05.commit(storage.mma_mbar_ptr.ptr)

        cute.arch.mbarrier_wait(storage.mma_mbar_ptr.ptr, mma_phase)
        mma_phase = mma_phase ^ 1
        pipeline.sync()

    tmem.relinquish_alloc_permit()

    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        cute.autovec_copy(tCrAcc, tDgC[None, None, i])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        io_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        tcgen05.CtaGroup.ONE,
        mma_tiler_mnk[:2],
    )

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        1,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        1,
    )

    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a,
        b,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )


def validate_mnk(mnk: Tuple[int, int, int]) -> None:
    m, n, k = mnk
    if (
        m % mma_tiler_mnk[0] != 0
        or n % mma_tiler_mnk[1] != 0
        or k % mma_tiler_mnk[2] != 0
    ):
        raise ValueError(
            f"mnk must be divisible by tile {mma_tiler_mnk}, got {(m, n, k)}"
        )


def _to_cute_tensor(x, compact_divisor: int):
    return (
        from_dlpack(x, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=compact_divisor)
    )


def run_dense_gemm(a, b):
    import torch

    m, k = a.shape
    n = b.shape[0]
    validate_mnk((m, n, k))
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)

    a_tensor = _to_cute_tensor(a, k)
    b_tensor = _to_cute_tensor(b, k)
    c_tensor = _to_cute_tensor(c, n)

    host_function(a_tensor, b_tensor, c_tensor, no_cache=True)
    return c


def _parse_mnk(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("expected M,N,K")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    import argparse
    from cuda.bindings import driver as cu_driver
    from ref import check_close, make_inputs, torch_gemm

    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=_parse_mnk, default=(128, 256, 64))
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    cu_driver.cuInit(0)
    a, b = make_inputs(args.mnk)
    got = run_dense_gemm(a, b)
    ref = torch_gemm(a, b)
    check_close(got, ref, atol=args.atol)
    print("PASS", {"mnk": args.mnk, "dtype": "fp16->fp32"})
