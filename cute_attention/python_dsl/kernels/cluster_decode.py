from __future__ import annotations

from .common import AttentionConfig, HAS_CUTE, cutlass, cute, from_dlpack, require_torch, torch


_CLUSTER_DECODE_COMPILED_CACHE = {}


if HAS_CUTE:
    def _make_cluster_decode_host(seq_len: int, head_dim: int, num_threads: int):
        @cute.kernel
        def decode_attention_kernel(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bh_idx, _, _ = cute.arch.block_idx()

            smem = cutlass.utils.SmemAllocator()
            reduce_ptr = smem.allocate_array(cutlass.Float32, num_elems=num_threads)
            reduce = cute.make_tensor(reduce_ptr, cute.make_layout((num_threads,)))

            local_max = -cutlass.Float32.inf
            for kv_idx in range(seq_len):
                if kv_idx % num_threads == tidx:
                    score = 0.0
                    for d_idx in range(head_dim):
                        score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                    score *= softmax_scale
                    local_max = score if local_max < score else local_max

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
                    score = 0.0
                    for d_idx in range(head_dim):
                        score += q[bh_idx, 0, d_idx] * k[bh_idx, kv_idx, d_idx]
                    local_sum += cute.math.exp(score * softmax_scale - row_max)

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
                        score = 0.0
                        for qk_d_idx in range(head_dim):
                            score += q[bh_idx, 0, qk_d_idx] * k[bh_idx, kv_idx, qk_d_idx]
                        prob = cute.math.exp(score * softmax_scale - row_max) * inv_sum
                        acc += prob * v[bh_idx, kv_idx, d_idx]
                    o[bh_idx, 0, d_idx] = acc.to(o.element_type)

        @cute.jit
        def cluster_decode_forward_host(
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            o: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            batch_heads = cute.size(q.shape, mode=[0])
            decode_attention_kernel(
                q,
                k,
                v,
                o,
                softmax_scale,
            ).launch(
                grid=(batch_heads, 1, 1),
                block=(num_threads, 1, 1),
            )

        return cluster_decode_forward_host


def _validate_decode_qkv(q, k, v, config: AttentionConfig) -> None:
    require_torch()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v to have shape (batch, heads, seqlen, headdim).")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, v must have the same batch size.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("cluster_decode v0 only supports MHA with matching q/k/v heads.")
    if q.shape[2] != 1:
        raise ValueError("cluster_decode v0 only supports decode q_len=1.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("cluster_decode v0 requires q, k, v to have the same head_dim.")
    if q.shape[-1] != 128:
        raise ValueError("cluster_decode v0 is scoped to head_dim=128.")
    if config.cluster_size not in (2, 4):
        raise ValueError("cluster_decode v0 supports cluster_size 2 or 4.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("cluster_decode targets CUDA tensors only.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, v must be contiguous.")


def _decode_attention_reference(q, k, v, config: AttentionConfig):
    """Decode attention semantics: the single query attends all provided KV tokens."""
    scale = config.resolve_scale(q.shape[-1])
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_f).to(dtype=q.dtype)


def cluster_decode_forward(q, k, v, config: AttentionConfig | None = None):
    """Experimental ClusterFusion-style decode attention entrypoint.

    v0 intentionally keeps the callable surface small: MHA, q_len=1,
    non-paged KV, head_dim=128. It starts as a single-CTA CuTeDSL decode
    kernel; the next step is splitting KV across CTAs in a cluster while
    preserving this API and test coverage.
    """
    require_torch()
    config = config or AttentionConfig()
    _validate_decode_qkv(q, k, v, config)

    if not HAS_CUTE:
        raise RuntimeError("cluster_decode requires cutlass.cute for the planned kernel path.")

    batch, heads, _, head_dim = q.shape
    seq_len = k.shape[2]
    scale = config.resolve_scale(head_dim)

    q_flat = q.reshape(batch * heads, 1, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o_flat = q_flat.new_empty(q_flat.shape)

    q_cute = from_dlpack(q_flat, assumed_align=16).mark_layout_dynamic()
    k_cute = from_dlpack(k_flat, assumed_align=16).mark_layout_dynamic()
    v_cute = from_dlpack(v_flat, assumed_align=16).mark_layout_dynamic()
    o_cute = from_dlpack(o_flat, assumed_align=16).mark_layout_dynamic()

    cache_key = (tuple(q_flat.shape), tuple(k_flat.shape), str(q_flat.dtype), config.num_threads)
    compiled = _cluster_decode_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, config.num_threads)
    compiled(q_cute, k_cute, v_cute, o_cute, scale)

    return o_flat.reshape(batch, heads, 1, head_dim)


def _cluster_decode_compile(cache_key, q_cute, k_cute, v_cute, o_cute, scale, num_threads):
    compiled = _CLUSTER_DECODE_COMPILED_CACHE.get(cache_key)
    if compiled is None:
        seq_len = cache_key[1][1]
        head_dim = cache_key[0][2]
        host = _make_cluster_decode_host(seq_len=seq_len, head_dim=head_dim, num_threads=num_threads)
        compiled = cute.compile(
            host,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            scale,
        )
        _CLUSTER_DECODE_COMPILED_CACHE[cache_key] = compiled
    return compiled
