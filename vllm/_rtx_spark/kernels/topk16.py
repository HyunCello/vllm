# SPDX-License-Identifier: Apache-2.0
"""Exact top-k indices of 16-bit float rows without memsets or atomics.

torch.topk over a long row (the drafter's HEAD2 scorer: k=2048 of ~248k) runs a radix select whose passes
start with memsets. Inside a CUDA graph on the RTX Spark every memset node costs ~30 us of GPU idle, and the
drafter runs the selection once per draft token. This is a 16-bit radix select in four launches over chunks of
the row (grid: chunks x rows):

  1. every program histograms the high byte of its chunk's order-preserving keys
  2. every program finds the high byte the k-th largest key falls in (summing all histograms itself) and
     histograms the low byte of its keys in that bucket
  3. every program derives the threshold key and how many keys equal to it are needed, and counts its keys
     above / equal to the threshold
  4. every program writes its selected indices at offsets given by the counts of the chunks before it

The selected set is exactly the k largest values (NaN ranks highest, as in torch.topk); among values equal to the
k-th largest, the lowest indices are taken (torch.topk leaves that choice unspecified). Output order is
unspecified, as with sorted=False.
"""

import torch

from vllm.triton_utils import tl, triton

_CHUNK = 4096


@triton.jit
def _keys(row_ptr, offs, mask):
    bits = tl.load(row_ptr + offs, mask=mask, other=0).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    # order-preserving: non-negative floats get the sign bit set, negative ones are inverted
    return tl.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)


@triton.jit
def _split(hist_ptr, num_chunks, need):
    """Sum per-chunk 256-bin histograms; return (bin holding the need-th largest, count above that bin)."""
    bins = tl.arange(0, 256)
    total = tl.zeros([256], dtype=tl.int32)
    for c in range(0, num_chunks):
        total += tl.load(hist_ptr + c * 256 + bins)
    at_or_above = tl.flip(tl.cumsum(tl.flip(total, 0), 0), 0)
    above = at_or_above - total
    hit = (at_or_above >= need) & (above < need)
    b = tl.max(tl.where(hit, bins, -1), 0)
    return b, tl.sum(tl.where(bins == b, above, 0), 0)


@triton.jit
def _hist_high(x_ptr, hist_ptr, n_cols, x_stride, num_chunks, CHUNK: tl.constexpr):
    chunk, row = tl.program_id(0), tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n_cols
    keys = _keys(x_ptr + row * x_stride, offs, mask)
    h = tl.histogram(keys >> 8, 256, mask=mask)
    tl.store(hist_ptr + (row * num_chunks + chunk) * 256 + tl.arange(0, 256), h)


@triton.jit
def _hist_low(x_ptr, hist_hi_ptr, hist_lo_ptr, n_cols, x_stride, num_chunks, k, CHUNK: tl.constexpr):
    chunk, row = tl.program_id(0), tl.program_id(1)
    hi, _ = _split(hist_hi_ptr + row * num_chunks * 256, num_chunks, k)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n_cols
    keys = _keys(x_ptr + row * x_stride, offs, mask)
    h = tl.histogram(keys & 0xFF, 256, mask=mask & ((keys >> 8) == hi))
    tl.store(hist_lo_ptr + (row * num_chunks + chunk) * 256 + tl.arange(0, 256), h)


@triton.jit
def _count(x_ptr, hist_hi_ptr, hist_lo_ptr, cnt_ptr, params_ptr, n_cols, x_stride, num_chunks, k,
           CHUNK: tl.constexpr):
    chunk, row = tl.program_id(0), tl.program_id(1)
    hi, above_hi = _split(hist_hi_ptr + row * num_chunks * 256, num_chunks, k)
    lo, above_lo = _split(hist_lo_ptr + row * num_chunks * 256, num_chunks, k - above_hi)
    threshold = hi * 256 + lo
    need_eq = k - above_hi - above_lo
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n_cols
    keys = _keys(x_ptr + row * x_stride, offs, mask)
    gt = tl.sum((mask & (keys > threshold)).to(tl.int32), 0)
    eq = tl.sum((mask & (keys == threshold)).to(tl.int32), 0)
    tl.store(cnt_ptr + (row * num_chunks + chunk) * 2, gt)
    tl.store(cnt_ptr + (row * num_chunks + chunk) * 2 + 1, eq)
    if chunk == 0:
        tl.store(params_ptr + row * 2, threshold)
        tl.store(params_ptr + row * 2 + 1, need_eq)


@triton.jit
def _write(x_ptr, cnt_ptr, params_ptr, out_ptr, n_cols, x_stride, num_chunks, k, CHUNK: tl.constexpr):
    chunk, row = tl.program_id(0), tl.program_id(1)
    threshold = tl.load(params_ptr + row * 2)
    need_eq = tl.load(params_ptr + row * 2 + 1)
    gt_before = 0
    eq_before = 0
    for c in range(0, chunk):
        gt_before += tl.load(cnt_ptr + (row * num_chunks + c) * 2)
        eq_before += tl.load(cnt_ptr + (row * num_chunks + c) * 2 + 1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n_cols
    keys = _keys(x_ptr + row * x_stride, offs, mask)
    is_gt = mask & (keys > threshold)
    is_eq = mask & (keys == threshold)
    gt_pos = gt_before + tl.cumsum(is_gt.to(tl.int32), 0) - 1
    eq_rank = eq_before + tl.cumsum(is_eq.to(tl.int32), 0) - 1
    out_row = out_ptr + row * k
    tl.store(out_row + gt_pos, offs, mask=is_gt)
    tl.store(out_row + (k - need_eq) + eq_rank, offs, mask=is_eq & (eq_rank < need_eq))


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """int32 indices [rows, k] of the k largest values of each row (unordered), like
    ``x.topk(k, dim=-1, sorted=False).indices``; 16-bit float rows run without memsets."""
    if x.dim() != 2 or x.dtype not in (torch.bfloat16, torch.float16) or x.stride(1) != 1 or not x.is_cuda:
        return x.topk(k, dim=-1, sorted=False).indices.to(torch.int32)
    n, cols = x.shape
    k = min(k, cols)
    out = torch.empty((n, k), dtype=torch.int32, device=x.device)
    if n == 0 or k == 0:
        return out
    chunks = triton.cdiv(cols, _CHUNK)
    hists = torch.empty((2, n, chunks, 256), dtype=torch.int32, device=x.device)
    counts = torch.empty((n, chunks, 2), dtype=torch.int32, device=x.device)
    params = torch.empty((n, 2), dtype=torch.int32, device=x.device)
    grid = (chunks, n)
    _hist_high[grid](x, hists[0], cols, x.stride(0), chunks, CHUNK=_CHUNK)
    _hist_low[grid](x, hists[0], hists[1], cols, x.stride(0), chunks, k, CHUNK=_CHUNK)
    _count[grid](x, hists[0], hists[1], counts, params, cols, x.stride(0), chunks, k, CHUNK=_CHUNK)
    _write[grid](x, counts, params, out, cols, x.stride(0), chunks, k, CHUNK=_CHUNK)
    return out
