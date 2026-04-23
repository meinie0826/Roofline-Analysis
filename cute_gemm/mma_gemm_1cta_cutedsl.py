from typing import Tuple
import itertools

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


def get_cta_v_map_ab(gmem_tensor, mma_tiler_mnk, tiled_mma, input_operand):
    ident = cute.make_identity_layout(gmem_tensor.shape)
    mode = 0 if input_operand == "A" else 1
    mma_tiler_mk = (mma_tiler_mnk[mode], *mma_tiler_mnk[2:])
    g_tile = cute.composition(ident, mma_tiler_mk)
    if input_operand == "A":
        cta_v_map = tiled_mma._thrfrg_A(g_tile)
    elif input_operand == "B":
        cta_v_map = tiled_mma._thrfrg_B(g_tile)
    else:
        raise ValueError(f"Unsupported operand {input_operand}")
    cta_v_map = cute.get(cta_v_map, mode=[1])
    cta_v_map = cute.dice(cta_v_map, (1, (1,) * cute.rank(g_tile)))
    return cta_v_map


def dump_swizzle_mapping(name: str, composed_layout: cute.ComposedLayout, limit: int = 16):
    outer = composed_layout.outer
    total = cute.size(outer)
    rows = []
    first_diff = None
    for i in range(total):
        coord = outer.get_hier_coord(i)
        raw = int(outer(coord))
        swz = int(composed_layout(coord))
        if len(rows) < limit:
            rows.append((i, coord, raw, swz))
        if first_diff is None and raw != swz:
            first_diff = i

    print(f"=== {name} swizzle mapping ===")
    print(f"{'i':>3} | {'coord':>18} | {'raw':>6} | {'swz':>6}")
    for i, coord, raw, swz in rows:
        print(f"{i:3d} | {str(coord):>18} | {raw:6d} | {swz:6d}")
    if first_diff is None:
        print("first raw!=swz index: none in full layout")
    else:
        print(f"first raw!=swz index: {first_diff}")
        print(f"=== {name} first swizzle differences ===")
        print(f"{'i':>3} | {'coord':>18} | {'raw':>6} | {'swz':>6}")
        shown = 0
        for i in range(first_diff, total):
            coord = outer.get_hier_coord(i)
            raw = int(outer(coord))
            swz = int(composed_layout(coord))
            if raw == swz:
                continue
            print(f"{i:3d} | {str(coord):>18} | {raw:6d} | {swz:6d}")
            shown += 1
            if shown >= limit:
                break
    print()


def bank_id_from_elem_offset(
    offset: int, element_bits: int, bank_width_bytes: int = 4, num_banks: int = 32
):
    elem_bytes = max(1, element_bits // 8)
    byte_addr = offset * elem_bytes
    return (byte_addr // bank_width_bytes) % num_banks


def iter_static_coords(shape):
    if isinstance(shape, int):
        for i in range(shape):
            yield i
        return
    if not isinstance(shape, tuple):
        raise TypeError(f"Unsupported shape leaf {type(shape)}: {shape}")
    child_iters = [list(iter_static_coords(s)) for s in shape]
    for prod in itertools.product(*child_iters):
        yield tuple(prod)


def dump_mma_fragment_access(
    name: str,
    tiled_mma: cute.TiledMma,
    smem_layout: cute.ComposedLayout,
    element_type,
    operand: str,
    element_bits: int,
    limit: int = 32,
):
    raw_ptr = cute.make_ptr(
        element_type,
        0,
        cute.AddressSpace.smem,
        assumed_align=128,
    )
    swz_ptr = cute.recast_ptr(
        raw_ptr,
        swizzle_=smem_layout.inner,
        dtype=element_type,
    )
    raw_tensor = cute.make_tensor(raw_ptr, smem_layout.outer)
    swz_tensor = cute.make_tensor(swz_ptr, smem_layout.outer)
    if operand == "A":
        frag_raw = tiled_mma.make_fragment_A(raw_tensor)
        frag_swz = tiled_mma.make_fragment_A(swz_tensor)
    elif operand == "B":
        frag_raw = tiled_mma.make_fragment_B(raw_tensor)
        frag_swz = tiled_mma.make_fragment_B(swz_tensor)
    else:
        raise ValueError(f"Unsupported operand {operand}")

    print(f"=== {name} fragment access ===")
    print(f"{name}.fragment.layout = {cute.pretty_str(frag_raw.layout)}")
    num_k_blocks = cute.size(frag_raw, mode=[2])
    for k_block in range(num_k_blocks):
        frag_raw_k = frag_raw[(None, None, k_block, 0)]
        frag_swz_k = frag_swz[(None, None, k_block, 0)]
        total = cute.size(frag_raw_k)
        print(f"--- {name} k_block={k_block} ---")
        print(
            f"{'step':>4} | {'frag_coord':>18} | {'raw':>6} | {'swz':>6} | {'raw_bank':>8} | {'swz_bank':>8}"
        )
        for i in range(min(limit, total)):
            frag_coord = frag_raw_k.layout.get_hier_coord(i)
            raw = int(frag_raw_k.layout(frag_coord))
            swz = int(frag_swz_k.layout(frag_coord))
            raw_bank = bank_id_from_elem_offset(raw, element_bits)
            swz_bank = bank_id_from_elem_offset(swz, element_bits)
            print(
                f"{i:4d} | {str(frag_coord):>18} | {raw:6d} | {swz:6d} | {raw_bank:8d} | {swz_bank:8d}"
            )
        print()


def dump_layout_mapping(name: str, layout, limit: int = 32):
    total = cute.size(layout)
    print(f"=== {name} ===")
    print(f"{name}.layout = {cute.pretty_str(layout)}")
    print(f"{'i':>3} | {'coord':>18} | {'value':>18}")
    for i, coord in enumerate(iter_static_coords(layout.shape)):
        if i >= min(limit, total):
            break
        value = layout(coord)
        print(f"{i:3d} | {str(coord):>18} | {str(value):>18}")
    print()


def dump_thread_value_partition(
    name: str,
    tiled_mma: cute.TiledMma,
    gmem_tensor: cute.Tensor,
    mma_tiler_mnk,
    operand: str,
    limit_threads: int = 8,
    limit_values: int = 16,
):
    mode = 0 if operand == "A" else 1
    mma_tiler_mk = (mma_tiler_mnk[mode], *mma_tiler_mnk[2:])
    ident = cute.make_identity_layout(gmem_tensor.shape)
    g_tile = cute.composition(ident, mma_tiler_mk)

    print(f"=== {name} thread/value partition ===")
    print(f"{name}.tiled_mma.size = {tiled_mma.size}")
    print(f"{name}.thr_layout_vmnk = {cute.pretty_str(tiled_mma.thr_layout_vmnk)}")
    if operand == "A":
        thrfrg = tiled_mma._thrfrg_A(g_tile)
    elif operand == "B":
        thrfrg = tiled_mma._thrfrg_B(g_tile)
    else:
        raise ValueError(f"Unsupported operand {operand}")

    print(
        f"{name}.note = layout-only dump; ThrMma.partition_{operand} requires Tensor input"
    )
    print(f"{name}.thrfrg.layout = {cute.pretty_str(thrfrg)}")
    print(f"{'i':>3} | {'coord':>18} | {'value':>18}")
    for i, coord in enumerate(iter_static_coords(thrfrg.shape)):
        if i >= min(limit_values, cute.size(thrfrg)):
            break
        value = thrfrg(coord)
        print(f"{i:3d} | {str(coord):>18} | {str(value):>18}")
    print()

    projected = cute.get(thrfrg, mode=[1])
    projected = cute.dice(projected, (1, (1,) * cute.rank(g_tile)))
    print(f"{name}.projected_without_thread.layout = {cute.pretty_str(projected)}")
    print(f"{'i':>3} | {'coord':>18} | {'value':>18}")
    for i, coord in enumerate(iter_static_coords(projected.shape)):
        if i >= min(limit_values, cute.size(projected)):
            break
        value = projected(coord)
        print(f"{i:3d} | {str(coord):>18} | {str(value):>18}")
    print()


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
    debug_swizzle: cutlass.Constexpr[bool] = False,
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
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(storage.mma_mbar_ptr, 1)
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

    if debug_swizzle and bidx == 0 and bidy == 0 and tidx == 0:
        cute.printf("=== swizzle debug begin ===")
        cute.printf("sA.layout       = {}", sA.layout)
        cute.printf("sA_stage.layout = {}", sA_stage.layout)
        cute.printf("sA.iter.swz     = {}", sA.iterator)
        cute.printf(
            "sA.iter.raw     = {}",
            cute.recast_ptr(sA.iterator, swizzle_=None, dtype=io_dtype),
        )
        cute.printf("sB.layout       = {}", sB.layout)
        cute.printf("sB_stage.layout = {}", sB_stage.layout)
        cute.printf("sB.iter.swz     = {}", sB.iterator)
        cute.printf(
            "sB.iter.raw     = {}",
            cute.recast_ptr(sB.iterator, swizzle_=None, dtype=io_dtype),
        )
        cute.printf("=== swizzle debug end ===")

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
            # UMMA completion arrive must be issued by exactly one thread.
            with cute.arch.elect_one():
                tcgen05.commit(storage.mma_mbar_ptr)

        cute.arch.mbarrier_wait(storage.mma_mbar_ptr, mma_phase)
        mma_phase = mma_phase ^ 1
        pipeline.sync()

    tmem.relinquish_alloc_permit()

    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        cute.autovec_copy(tCrAcc, tDgC[None, None, i])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    debug_swizzle: cutlass.Constexpr[bool] = False,
):
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

    if debug_swizzle:
        print("=== host swizzle debug begin ===")
        print("a_smem_layout       =", cute.pretty_str(a_smem_layout))
        print("a_smem_layout.outer =", cute.pretty_str(a_smem_layout.outer))
        print("a_smem_layout.inner =", cute.pretty_str(a_smem_layout.inner))
        print("b_smem_layout       =", cute.pretty_str(b_smem_layout))
        print("b_smem_layout.outer =", cute.pretty_str(b_smem_layout.outer))
        print("b_smem_layout.inner =", cute.pretty_str(b_smem_layout.inner))
        print("=== host swizzle debug end ===")
        dump_swizzle_mapping("A", a_smem_layout, limit=16)
        dump_swizzle_mapping("B", b_smem_layout, limit=16)
        dump_mma_fragment_access("A", tiled_mma, a_smem_layout, io_dtype, "A", io_dtype.width)
        dump_mma_fragment_access("B", tiled_mma, b_smem_layout, io_dtype, "B", io_dtype.width)
        print("=== tiled mma summary ===")
        print(tiled_mma)
        dump_layout_mapping("tiled_mma.tv_layout_A", tiled_mma.tv_layout_A, limit=32)
        dump_layout_mapping("tiled_mma.tv_layout_B", tiled_mma.tv_layout_B, limit=32)
        dump_layout_mapping("tiled_mma.tv_layout_A_tiled", tiled_mma.tv_layout_A_tiled, limit=32)
        dump_layout_mapping("tiled_mma.tv_layout_B_tiled", tiled_mma.tv_layout_B_tiled, limit=32)
        cta_v_map_a = get_cta_v_map_ab(a, mma_tiler_mnk, tiled_mma, "A")
        cta_v_map_b = get_cta_v_map_ab(b, mma_tiler_mnk, tiled_mma, "B")
        dump_layout_mapping("cta_v_map_A", cta_v_map_a, limit=32)
        dump_layout_mapping("cta_v_map_B", cta_v_map_b, limit=32)
        dump_thread_value_partition("A", tiled_mma, a, mma_tiler_mnk, "A")
        dump_thread_value_partition("B", tiled_mma, b, mma_tiler_mnk, "B")

    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a,
        b,
        c,
        a_smem_layout,
        b_smem_layout,
        debug_swizzle,
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


def prepare_cute_gemm(a, b):
    import torch

    m, k = a.shape
    n = b.shape[0]
    validate_mnk((m, n, k))
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)

    a_tensor = _to_cute_tensor(a, k)
    b_tensor = _to_cute_tensor(b, k)
    c_tensor = _to_cute_tensor(c, n)
    return c, a_tensor, b_tensor, c_tensor


def run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor, debug_swizzle: bool = False):
    host_function(a_tensor, b_tensor, c_tensor, debug_swizzle)
    return c


def run_dense_gemm(a, b, debug_swizzle: bool = False):
    c, a_tensor, b_tensor, c_tensor = prepare_cute_gemm(a, b)
    return run_dense_gemm_prepared(c, a_tensor, b_tensor, c_tensor, debug_swizzle)


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
    parser.add_argument("--debug-swizzle", action="store_true")
    args = parser.parse_args()

    cu_driver.cuInit(0)
    a, b = make_inputs(args.mnk)
    got = run_dense_gemm(a, b, debug_swizzle=args.debug_swizzle)
    ref = torch_gemm(a, b)
    check_close(got, ref, atol=args.atol)
    print("PASS", {"mnk": args.mnk, "dtype": "fp16->fp32"})
