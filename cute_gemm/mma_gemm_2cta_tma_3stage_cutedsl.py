import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
output_dtype = cutlass.Float16

threads_per_cta = 192
threads_in_epilogue = 128
cluster_shape_mnk = (2, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
ab_stages = 3
acc_stages = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
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
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy,
        None,
    )
    mma_coord_mnk = mma_coord_vmnk[1:]
    is_leader_cta = mma_coord_vmnk[0] == 0

    epilogue_warp_ids = (0, 1, 2, 3)
    mma_warp_id = 4
    tma_warp_id = 5

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

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

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
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])

    num_mcast_participants = (
        cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=num_mcast_participants,
        ),
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
        ),
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()

    num_k_tiles = cute.size(gA, mode=[2])
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    if warp_idx == tma_warp_id:
        for k_tile_idx in cutlass.range(num_k_tiles):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile_idx)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_a,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile_idx)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_b,
            )
        ab_producer.tail()

    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        if is_leader_cta:
            acc_empty = acc_producer.acquire_and_advance()
            for k_tile_idx in cutlass.range(num_k_tiles):
                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_full.release()
            acc_empty.commit()

    elif warp_idx < mma_warp_id:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        acc_consumer.wait_and_advance()

        subtile_cnt = 4
        epi_tiler = (
            (
                cute.size(tCtAcc, mode=[0, 0]),
                cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt,
            ),
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

        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(output_dtype))
            cute.autovec_copy(tCrC, tDgC[None, None, i])

        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    tma_load_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    tma_atom_a, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        a,
        a_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    tma_atom_b, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op,
        b,
        b_smem_layout_slice,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    grid_shape = cute.round_up(
        cute.ceil_div(
            (*c.layout.shape, 1),
            (mma_tiler_mnk[0] // cluster_shape_mnk[0], mma_tiler_mnk[1], 1),
        ),
        cluster_shape_mnk,
    )
    kernel(
        tiled_mma,
        tma_atom_a,
        a_tma_tensor,
        tma_atom_b,
        b_tma_tensor,
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
    from cuda.bindings import driver as cu_driver
    from ref import check_close, make_inputs, torch_gemm_with_dtype

    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=_parse_mnk, default=(256, 256, 64))
    parser.add_argument("--atol", type=float, default=1e-1)
    args = parser.parse_args()

    cu_driver.cuInit(0)
    a, b = make_inputs(args.mnk)
    got = run_dense_gemm(a, b)
    ref = torch_gemm_with_dtype(a, b, output_dtype)
    check_close(got, ref, atol=args.atol, rtol=1e-5)
    print("PASS", {"mnk": args.mnk, "variant": "2cta_tma_3stage"})
