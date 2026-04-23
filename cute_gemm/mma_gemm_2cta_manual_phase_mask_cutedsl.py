import argparse
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
output_dtype = cutlass.Float16

threads_per_cta = 128
cluster_shape_mnk = (2, 1, 1)
mma_tiler_mnk = (256, 256, 64)


@cute.struct
class SharedStorage:
    mma_sync_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    mA_mk: cute.Tensor,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy,
        None,
    )
    mma_coord_mnk = mma_coord_vmnk[1:]
    is_leader_cta = cta_rank_in_cluster % 2 == 0

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
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
    )
    tmem.allocate(512)

    prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    cons_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        cute.size(cta_layout_vmnk, mode=[0]) * threads_per_cta,
    )
    mbar_base_ptr = storage.mma_sync_mbar_ptr.data_ptr().align(min_align=8)
    mma_full_barrier = pipeline.MbarrierArray(
        barrier_storage=mbar_base_ptr,
        num_stages=1,
        agent=(pipeline.PipelineOp.TCGen05Mma, prod_group),
    )
    mma_empty_barrier = pipeline.MbarrierArray(
        barrier_storage=mbar_base_ptr + 1,
        num_stages=1,
        agent=(pipeline.PipelineOp.AsyncThread, cons_group),
    )

    # Match PipelineUmmaAsync.create(): full barrier uses the local CTA image mask,
    # empty barrier releases back to the leading CTA in the 2CTA pair.
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    mma_mcast_mask = cute.make_layout_image_mask(
        cta_layout_vmnk,
        cta_in_cluster_coord_vmnk,
        mode=0,
    )
    empty_dst_rank = cta_rank_in_cluster // 2 * 2

    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()

    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
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
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, output_dtype)

    sA_stage = sA[(None, None, None, 0)]
    sB_stage = sB[(None, None, None, 0)]

    mma_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        1,
    )
    mma_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        1,
    )
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
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if is_leader_cta and warp_idx == 0:
            prod_state = mma_producer_state.clone()
            mma_empty_barrier.wait(prod_state.index, prod_state.phase)
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
            mma_full_barrier.arrive_tcgen05mma(
                prod_state.index,
                mma_mcast_mask,
                tcgen05.CtaGroup.TWO,
            )
        mma_producer_state.advance()

        cons_state = mma_consumer_state.clone()
        mma_full_barrier.wait(cons_state.index, cons_state.phase)
        mma_empty_barrier.arrive(cons_state.index, empty_dst_rank)
        mma_consumer_state.advance()

    if is_leader_cta:
        # Match PipelineUmmaAsync.producer_tail() for a single-stage pipeline.
        mma_empty_barrier.wait(
            mma_producer_state.index,
            mma_producer_state.phase,
        )

    tmem.relinquish_alloc_permit()

    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(output_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        io_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        acc_dtype,
        tcgen05.CtaGroup.TWO,
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
    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    grid_shape = cute.round_up(
        cute.ceil_div(
            (*c.layout.shape, 1),
            (mma_tiler_mnk[0] // cluster_shape_mnk[0], mma_tiler_mnk[1], 1),
        ),
        cluster_shape_mnk,
    )
    kernel(
        tiled_mma,
        a,
        b,
        c,
        a_smem_layout,
        b_smem_layout,
        cta_layout_vmnk,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
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


def prepare_cute_gemm(a, b):
    import torch

    m, k = a.shape
    n = b.shape[0]
    validate_mnk((m, n, k))
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    a_tensor = _to_cute_tensor(a, k)
    b_tensor = _to_cute_tensor(b, k)
    c_tensor = _to_cute_tensor(c, n)
    return c, a_tensor, b_tensor, c_tensor


def run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor):
    host_function(a_tensor, b_tensor, c_tensor)
    return c


def run_dense_gemm(a, b):
    c, a_tensor, b_tensor, c_tensor = prepare_cute_gemm(a, b)
    return run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor)


def _parse_mnk(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    import torch
    from cuda.bindings import driver as cu_driver
    from ref import check_close, make_inputs, torch_gemm_with_dtype

    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=_parse_mnk, default=(256, 256, 64))
    parser.add_argument("--atol", type=float, default=1e-1)
    args = parser.parse_args()

    cu_driver.cuInit(0)
    a, b = make_inputs(args.mnk)
    got = run_dense_gemm(a, b)
    ref = torch_gemm_with_dtype(a, b, torch.float16)
    check_close(got, ref, atol=args.atol, rtol=1e-5)
    print("PASS", {"mnk": args.mnk, "variant": "2cta_manual_phase_mask"})
