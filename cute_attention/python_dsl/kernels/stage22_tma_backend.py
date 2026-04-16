from __future__ import annotations

from .common import cutlass, cute


def make_stage22_host(seq_len: int, head_dim: int, num_threads: int):
    """Build the simplest standalone causal-attention kernel for stage22.

    This is intentionally a clean rewrite: one CTA computes one output row for one
    batch-head slice, using explicit score storage and reductions. The goal is a
    small, readable baseline we can evolve from, not a TMA / multistage kernel.
    """

    @cute.kernel
    def stage22_causal_attention_kernel(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_idx, bh_idx, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        scores_ptr = smem.allocate_array(cutlass.Float32, num_elems=seq_len)
        reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
        scores = cute.make_tensor(scores_ptr, cute.make_layout((seq_len,)))
        reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

        local_max = -cutlass.Float32.inf
        for kv_idx in range(seq_len):
            if kv_idx % num_threads == tidx:
                score = -cutlass.Float32.inf
                if kv_idx <= query_idx:
                    score = 0.0
                    for d_idx in range(head_dim):
                        score += q[bh_idx, query_idx, d_idx] * k[bh_idx, kv_idx, d_idx]
                    score *= softmax_scale
                    local_max = score if local_max < score else local_max
                scores[kv_idx] = score

        reduce[tidx] = local_max
        cute.arch.barrier()

        stride = num_threads // 2
        while stride > 0:
            if tidx < stride:
                rhs = reduce[tidx + stride]
                lhs = reduce[tidx]
                reduce[tidx] = rhs if lhs < rhs else lhs
            cute.arch.barrier()
            stride //= 2

        row_max = reduce[0]
        local_sum = 0.0
        for kv_idx in range(seq_len):
            if kv_idx % num_threads == tidx:
                prob = 0.0
                if kv_idx <= query_idx:
                    prob = cute.math.exp(scores[kv_idx] - row_max)
                scores[kv_idx] = prob
                local_sum += prob

        reduce[tidx] = local_sum
        cute.arch.barrier()

        stride = num_threads // 2
        while stride > 0:
            if tidx < stride:
                reduce[tidx] = reduce[tidx] + reduce[tidx + stride]
            cute.arch.barrier()
            stride //= 2

        row_sum = reduce[0]
        inv_sum = 1.0 / row_sum

        for d_idx in range(head_dim):
            if d_idx % num_threads == tidx:
                acc = 0.0
                for kv_idx in range(seq_len):
                    if kv_idx <= query_idx:
                        acc += scores[kv_idx] * inv_sum * v[bh_idx, kv_idx, d_idx]
                o[bh_idx, query_idx, d_idx] = acc.to(o.element_type)

    @cute.jit
    def stage22_forward_host(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        batch_heads = cute.size(q.shape, mode=[0])
        stage22_causal_attention_kernel(
            q,
            k,
            v,
            o,
            softmax_scale,
        ).launch(
            grid=(seq_len, batch_heads, 1),
            block=(num_threads, 1, 1),
        )

    return stage22_forward_host
