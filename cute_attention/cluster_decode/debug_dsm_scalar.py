"""Minimal DSM scalar communication probes for CuTeDSL cluster kernels.

Run on a Blackwell/CuTeDSL machine:

    PYTHONPATH=cute_attention python3 -m cluster_decode.debug_dsm_scalar --variant sync
    PYTHONPATH=cute_attention python3 -m cluster_decode.debug_dsm_scalar --variant ptx-store
    PYTHONPATH=cute_attention python3 -m cluster_decode.debug_dsm_scalar --variant cute-atomic

The `ptx-store` variant follows the inline-PTX pattern used by CUTLASS'
`blackwell/reduce.py`: map a shared-memory pointer to each peer CTA and use
`st.async.shared::cluster...mbarrier::complete_tx` to send one f32 value.

The `cute-atomic` variant intentionally exercises the higher-level
`cute.arch.mapa() + cute.arch.atomic_add()` path that has triggered an NVVM ICE
in the megakernel on sm_100a. Keep it isolated until that path is known-good.
"""

from __future__ import annotations

import argparse

from .common import HAS_CUTE, cutlass, cute, from_dlpack, require_torch


if HAS_CUTE:
    from cutlass import Float32, Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def _mapa_shared_cluster(
        smem_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Int32:
        smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_shared_remote_f32(
        val: Float32,
        smem_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        remote_smem_i32 = _mapa_shared_cluster(
            smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        remote_mbar_i32 = _mapa_shared_cluster(
            mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
        ).ir_value()
        llvm.inline_asm(
            None,
            [remote_smem_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_i32],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
            "r,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def _make_sync_probe_host(cluster_size: int, num_threads: int):
        cluster_shape = (cluster_size, 1, 1)

        @cute.kernel
        def _sync_probe(out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            if tidx == 0:
                out[bidx] = cta_rank.to(out.element_type)

        @cute.jit
        def _sync_probe_host(out: cute.Tensor):
            _sync_probe(out).launch(
                grid=(cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return _sync_probe_host

    def _make_ptx_store_probe_host(cluster_size: int, num_threads: int):
        cluster_shape = (cluster_size, 1, 1)

        @cute.kernel
        def _ptx_store_probe(x: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

            smem = cutlass.utils.SmemAllocator()
            vals_ptr = smem.allocate_array(cutlass.Float32, num_elems=cluster_size)
            vals = cute.make_tensor(vals_ptr, cute.make_layout((cluster_size,)))
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

            if tidx == 0:
                for i in range(cluster_size):
                    vals[i] = cutlass.Float32(0.0)
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            local_val = x[cta_rank].to(cutlass.Float32)

            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, cluster_size * 4)

            if tidx < cluster_size:
                dst_ptr = vals_ptr + cta_rank
                _store_shared_remote_f32(local_val, dst_ptr, mbar_ptr, tidx)

            cute.arch.mbarrier_wait(mbar_ptr, phase=0)

            if tidx == 0:
                total = cutlass.Float32(0.0)
                for i in range(cluster_size):
                    total = total + vals[i]
                out[cta_rank] = total.to(out.element_type)

        @cute.jit
        def _ptx_store_probe_host(x: cute.Tensor, out: cute.Tensor):
            _ptx_store_probe(x, out).launch(
                grid=(cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return _ptx_store_probe_host

    def _make_cute_atomic_probe_host(cluster_size: int, num_threads: int):
        cluster_shape = (cluster_size, 1, 1)

        @cute.kernel
        def _cute_atomic_probe(x: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

            smem = cutlass.utils.SmemAllocator()
            scalar_ptr = smem.allocate_array(cutlass.Float32, num_elems=1)
            local_val = x[cta_rank].to(cutlass.Float32)

            if tidx == 0:
                scalar_ptr[0] = local_val
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for i in range(1, cluster_size):
                dst_cta = (cta_rank + i) % cluster_size
                remote_ptr = cute.arch.mapa(scalar_ptr, dst_cta)
                if tidx == 0:
                    cute.arch.atomic_add(remote_ptr, local_val, scope="cluster")
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            if tidx == 0:
                out[cta_rank] = scalar_ptr[0].to(out.element_type)

        @cute.jit
        def _cute_atomic_probe_host(x: cute.Tensor, out: cute.Tensor):
            _cute_atomic_probe(x, out).launch(
                grid=(cluster_size, 1, 1),
                block=(num_threads, 1, 1),
                cluster=cluster_shape,
            )

        return _cute_atomic_probe_host


def run_probe(variant: str, cluster_size: int, num_threads: int) -> None:
    require_torch()
    if not HAS_CUTE:
        raise RuntimeError("CuTe DSL is required for DSM scalar probes.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DSM scalar probes.")

    x = torch.arange(1, cluster_size + 1, device="cuda", dtype=torch.float32)
    out = torch.empty((cluster_size,), device="cuda", dtype=torch.float32)
    x_cute = from_dlpack(x, assumed_align=16).mark_layout_dynamic()
    out_cute = from_dlpack(out, assumed_align=16).mark_layout_dynamic()

    if variant == "sync":
        host = _make_sync_probe_host(cluster_size, num_threads)
        compiled = cute.compile(host, out_cute)
        compiled(out_cute)
        expected = torch.arange(cluster_size, device="cuda", dtype=torch.float32)
    elif variant == "ptx-store":
        host = _make_ptx_store_probe_host(cluster_size, num_threads)
        compiled = cute.compile(host, x_cute, out_cute)
        compiled(x_cute, out_cute)
        expected = torch.full((cluster_size,), float(cluster_size * (cluster_size + 1) // 2), device="cuda")
    elif variant == "cute-atomic":
        host = _make_cute_atomic_probe_host(cluster_size, num_threads)
        compiled = cute.compile(host, x_cute, out_cute)
        compiled(x_cute, out_cute)
        expected = torch.full((cluster_size,), float(cluster_size * (cluster_size + 1) // 2), device="cuda")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    torch.cuda.synchronize()
    print(f"variant={variant}, cluster_size={cluster_size}, num_threads={num_threads}")
    print("out     =", out.cpu().tolist())
    print("expected=", expected.cpu().tolist())
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal CuTeDSL DSM scalar probes.")
    parser.add_argument("--variant", choices=["sync", "ptx-store", "cute-atomic"], default="ptx-store")
    parser.add_argument("--cluster-size", type=int, choices=[2, 4], default=2)
    parser.add_argument("--num-threads", type=int, default=128)
    args = parser.parse_args()
    run_probe(args.variant, args.cluster_size, args.num_threads)


if __name__ == "__main__":
    main()
