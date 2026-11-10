"""FP8 linear with dynamic per-token activation scales for sm_12x without CUTLASS FP8 (MBX_FP8_TOKEN_SCALE, default on).

Online FP8 (`--quantization fp8_per_tensor`) quantizes activations per token where CUTLASS FP8 exists (H100) and per
TENSOR otherwise, because torch._scaled_mm on sm_121 has no rowwise scaling ("Rowwise scaling is not currently
supported on your device"). One activation scale per batch couples the rows: a padded CUDA-graph row with stale
values (NaN, or just large) sets the scale for every request in the batch, which gave NaN logits and skewed
probabilities for OpenJev (2026-11-10), and answers moved with whatever else was batched.

This kernel computes C[m, n] = sa[m] * sb[n] * sum_k Aq[m, k] * Bq[k, n] (+ bias[n]) in one Triton GEMM, with the
scales in the fp32 epilogue: sa per token, sb one value (per-tensor weights) or one per output channel. Speed against
the per-tensor torch._scaled_mm (cuBLAS) on the OpenJev shapes (N x K = 34816x5120, 5120x17408, 16384x5120,
5120x6144): faster for M <= 128, within 0-12% for M = 512..8192 (17% for 5120x17408 at M = 8192).

Configs come from a sweep on the RTX Spark N1X (sm_121, 2026-11-10): pointer loads for M <= 128, TMA tensor
descriptors (128x128x64, 4 warps, 4 stages) above.
"""
import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("MBX_FP8_TOKEN_SCALE", "1") != "0"

_allocator_set = False


def supported() -> bool:
    """CUDA sm_12x and MBX_FP8_TOKEN_SCALE is not 0."""
    if not ENABLED:
        return False
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.is_device_capability_family(120)


# M changes every step; without do_not_specialize each new M % 16 class and M == 1 compiles another variant while
# serving (a 5.5 s first request on a 6.8k-token state).
@triton.jit(do_not_specialize=["M"])
def _fp8_token_mm_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bk, stride_bn, stride_cm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    USE_TD: tl.constexpr, EVEN_K: tl.constexpr, PER_CHANNEL: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % group_m
    pid_n = (pid % group) // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if USE_TD:
        # B is the [N, K] row-major weight (K contiguous); out-of-range tiles read zeros.
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BM, BK])
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[stride_bn, 1], block_shape=[BN, BK])
        for k in range(0, tl.cdiv(K, BK)):
            a = a_desc.load([pid_m * BM, k * BK])
            b = b_desc.load([pid_n * BN, k * BK])
            acc = tl.dot(a, b.T, acc)
    else:
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm.to(tl.int64)[:, None] * stride_am + rk[None, :]
        b_ptrs = b_ptr + rn.to(tl.int64)[None, :] * stride_bn + rk[:, None] * stride_bk
        for k in range(0, tl.cdiv(K, BK)):
            if EVEN_K:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
            else:
                k_mask = rk < K - k * BK
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * stride_bk
    sa = tl.load(sa_ptr + rm, mask=m_mask, other=0.0)
    if PER_CHANNEL:
        sb = tl.load(sb_ptr + rn, mask=n_mask, other=0.0)
        acc = acc * sa[:, None] * sb[None, :]
    else:
        acc = acc * (sa * tl.load(sb_ptr))[:, None]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    c_ptrs = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _config(M: int):
    """(BM, BN, BK, GROUP_M, num_warps, num_stages, tma)."""
    if M <= 16:
        return 16, 64, 256, 1, 4, 4, False
    if M <= 32:
        return 32, 64, 256, 1, 4, 3, False
    if M <= 128:
        return 64, 64, 128, 8, 4, 4, False
    return 128, 128, 64, 8, 4, 4, True


def _set_allocator(device: torch.device) -> None:
    # On-device TMA descriptors need a global scratch allocator (same as triton_scaled_mm's).
    global _allocator_set
    if not _allocator_set:
        from vllm.triton_utils.allocation import set_triton_allocator

        set_triton_allocator(device)
        _allocator_set = True


def fp8_token_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """a: [M, K] fp8 (K contiguous), b: [K, N] fp8 (any strides; the vLLM layout is a transposed [N, K] weight),
    scale_a: M fp32 values, scale_b: 1 or N fp32 values, bias: N values or None. Returns [M, N] in out_dtype."""
    M, K = a.shape
    N = b.shape[1]
    assert b.shape[0] == K and a.stride(1) == 1
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    if M == 0:
        return out
    assert scale_a.numel() == M and scale_b.numel() in (1, N)
    BM, BN, BK, group_m, warps, stages, tma = _config(M)
    stride_bk, stride_bn = b.stride()
    # TMA tiles need a K-contiguous weight and 16-byte aligned rows.
    tma = tma and stride_bk == 1 and K % 16 == 0 and a.stride(0) % 16 == 0 and stride_bn % 16 == 0
    if tma:
        _set_allocator(a.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fp8_token_mm_kernel[grid](
        a, b, scale_a, scale_b, bias if bias is not None else scale_b, out,
        M, N, K,
        a.stride(0), stride_bk, stride_bn, out.stride(0),
        BM=BM, BN=BN, BK=BK, GROUP_M=group_m,
        USE_TD=tma, EVEN_K=K % BK == 0, PER_CHANNEL=scale_b.numel() != 1, HAS_BIAS=bias is not None,
        num_warps=warps, num_stages=stages,
    )
    return out
