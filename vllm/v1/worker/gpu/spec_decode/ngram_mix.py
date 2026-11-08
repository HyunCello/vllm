# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MBX ngram mix: prompt-lookup drafts in place of the MTP chain where the text repeats its context.

When the output copies from its context (returning an edited file, quoting code in the reasoning), the
continuation of the longest earlier match of the last tokens predicts long spans that the MTP drafter,
three to six tokens deep, cannot reach. After the MTP drafter has run, each request whose context has a
match of at least MBX_NGRAM_MIN tokens (longest match up to MBX_NGRAM_MAX, rightmost on ties) gets the
match's continuation as its drafts:

  - only when the MTP's own first draft equals the continuation's first token (the MTP sample and its
    proposal distribution stay at position 0, the continuation follows deterministically), or
  - for every match of at least MBX_NGRAM_STRONG tokens (0 = off), the whole draft deterministic.

Lossless with standard and block verification: the cached proposal distribution of every deterministic
draft position is rewritten to a point mass on the drafted token (logits 0 there, -inf elsewhere), so the
verifier sees exactly the distribution the token was drawn from, and the choice between the two drafters
depends only on the context and the MTP's position-0 sample, never on a later drafted token.

Depth: the host fixes each step's draft depth before the step. Once a request's continuation drafts have
been accepted (at least two, at the normal depth), each step records the deepest continuation it could
use; the host raises the next verify depth to that (MBX_NGRAM_DEPTH caps it, default the configured
num_speculative_tokens), one or two steps late (asynchronous copy, never waits). The MTP drafter still
drafts only the policy's depth: positions past it hold the continuation, or, when this step has no usable
continuation, point masses on filler tokens (exact, merely rejected). Verdicts of such steps are hidden
from the MTP depth policy and the acceptance estimator, which model the MTP drafter only.

  MBX_NGRAM_MIX=1     enable
"""
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import _ngram_scan_kernel

logger = init_logger(__name__)

ENABLED = os.environ.get("MBX_NGRAM_MIX", "0") == "1"
MIN_N = int(os.environ.get("MBX_NGRAM_MIN", "4"))
MAX_N = int(os.environ.get("MBX_NGRAM_MAX", "12"))
STRONG = int(os.environ.get("MBX_NGRAM_STRONG", "0"))
DEPTH_CAP = int(os.environ.get("MBX_NGRAM_DEPTH", "0"))
# continuation drafts accepted in the previous step before the depth is raised (0 = raise at once)
CONFIRM = int(os.environ.get("MBX_NGRAM_CONFIRM", "2"))
_BLOCK_L = 256
_BLOCK_V = 4096
_RING = 4


@triton.jit
def _mix_kernel(
    token_ids_ptr,  # int32 [max_num_reqs, L]
    token_ids_stride,
    idx_mapping_ptr,  # int64 [B]
    total_len_ptr,  # int32 [max_num_reqs]
    num_sampled_ptr,  # int32 [B]
    last_sampled_ptr,  # int64 [max_num_reqs]
    scratch_ptr,  # int64 [B, scratch_stride]
    scratch_stride,
    drafts_ptr,  # int64 [B, drafts_stride] (in/out)
    drafts_stride,
    point_ptr,  # int64 [B, K]: token of a point-mass row, -1 keeps the MTP row (out)
    used_ptr,  # int32 [max_num_reqs]: this step drafts the continuation (in: previous step, out)
    hide_ptr,  # int32 [max_num_reqs]: verdict not about the MTP drafter (out)
    desire_ptr,  # int32 [1] (atomic max)
    L,
    NB,
    mtp_depth,
    depth,
    desire_cap,
    MIN_N: tl.constexpr,
    STRONG: tl.constexpr,
    CONFIRM: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)

    nb = tl.arange(0, NB_PO2).to(tl.int64)
    score = tl.max(tl.load(scratch_ptr + b * scratch_stride + nb, mask=nb < NB, other=0), axis=0)
    has = score > 0
    s1 = tl.maximum(score - 1, 0)
    n = s1 // Lp1
    start = s1 - n * Lp1 + n
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    avail = tl.where(has, tl.maximum(seq_len - start, 0), 0)
    last_tok = tl.load(last_sampled_ptr + req)

    k = tl.arange(0, K_PO2).to(tl.int64)
    in_d = (k < depth) & (k < K)
    cont = tl.load(token_ids_ptr + req * token_ids_stride + tl.minimum(start + k, L - 1),
                   mask=in_d & (k < avail), other=0).to(tl.int64)
    cont = tl.where(k < avail, cont, last_tok)
    g0 = tl.sum(tl.where(k == 0, cont, 0), axis=0)
    mtp0 = tl.load(drafts_ptr + b * drafts_stride)

    if STRONG > 0:
        strong = n >= STRONG
    else:
        strong = n < 0
    num_sampled = tl.load(num_sampled_ptr + b)
    use = has & (num_sampled > 0) & (n >= MIN_N) & (strong | (g0 == mtp0))
    # Past the MTP's depth every position is deterministic: the continuation, or a filler when unused.
    past = in_d & (k >= mtp_depth)
    tl.store(drafts_ptr + b * drafts_stride + k, cont, mask=(in_d & use) | past)
    point = tl.where((use & in_d & ((k > 0) | strong)) | past, cont, -1)
    tl.store(point_ptr + b * K + k, point, mask=k < K)

    prev_used = tl.load(used_ptr + req)
    tl.store(used_ptr + req, use.to(tl.int32))
    tl.store(hide_ptr + req, (use | (depth > mtp_depth)).to(tl.int32))

    if CONFIRM > 0:
        want = use & (prev_used != 0) & (num_sampled >= CONFIRM + 1)
    else:   # raise on every continuation step after fully accepted drafts (v1)
        want = use & (num_sampled > 0)
    tl.atomic_max(desire_ptr, tl.where(want, tl.minimum(avail, desire_cap), 0).to(tl.int32))


@triton.jit
def _point_logits_kernel(
    point_ptr,  # int64 [B, K]
    idx_mapping_ptr,  # int64 [B]
    logits_ptr,  # [max_num_reqs, K, V]
    stride_r,
    stride_k,
    V,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)
    blk = tl.program_id(2).to(tl.int64)
    tok = tl.load(point_ptr + b * K + k)
    if tok < 0:
        return
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    offs = blk * BLOCK_V + tl.arange(0, BLOCK_V).to(tl.int64)
    val = tl.where(offs == tok, 0.0, float("-inf"))
    ptr = logits_ptr + req * stride_r + k * stride_k + offs
    tl.store(ptr, val.to(logits_ptr.dtype.element_ty), mask=offs < V)


class NgramMix:
    def __init__(self, runner):
        self.runner = runner
        self.device = runner.device
        self.K = runner.num_speculative_steps
        self.cap = min(DEPTH_CAP, self.K) if DEPTH_CAP > 0 else self.K
        self.L = runner.max_model_len
        self.max_blocks = triton.cdiv(self.L, _BLOCK_L)
        n = runner.max_num_reqs
        self.scratch = torch.zeros((n, self.max_blocks), dtype=torch.int64, device=self.device)
        self.point = torch.full((n, self.K), -1, dtype=torch.int64, device=self.device)
        self.used = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.hide = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.desire = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.ring = [{"host": torch.zeros(1, dtype=torch.int32, pin_memory=True),
                      "event": torch.cuda.Event(), "posted": False} for _ in range(_RING)]
        self.next_slot = 0
        self.announced = False
        logger.info("MBX ngram mix: prompt-lookup drafts for matches of %d-%d tokens (%s), depth up to %d "
                    "after %d accepted continuation drafts", MIN_N, MAX_N,
                    f"strong from {STRONG}" if STRONG > 0 else "MTP-agreement only", self.cap, CONFIRM)

    def mask_verdicts(self, idx_mapping, num_sampled, num_rejected):
        """Verdicts as the MTP models should see them: steps with deterministic drafts count as not drafted."""
        hide = (self.hide[idx_mapping.long()] != 0) & (num_sampled > 0)
        return (torch.where(hide, torch.ones_like(num_sampled), num_sampled),
                torch.where(hide, torch.zeros_like(num_rejected), num_rejected))

    def apply(self, input_batch, draft_tokens, num_sampled, depth, mtp_depth, draft_logits):
        num_reqs = input_batch.num_reqs
        if num_reqs == 0 or depth < 1:
            return
        req_states = self.runner.req_states
        token_ids = req_states.all_token_ids.gpu
        idx_mapping = input_batch.idx_mapping
        max_seq = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max()) + self.K + 1
        nb = max(1, min(self.max_blocks, triton.cdiv(max_seq, _BLOCK_L)))
        _ngram_scan_kernel[(num_reqs, nb)](
            token_ids, token_ids.stride(0), idx_mapping, req_states.total_len.gpu, num_sampled,
            self.scratch, self.scratch.stride(0), self.L,
            MIN_N, MAX_N, triton.next_power_of_2(MAX_N), _BLOCK_L, num_warps=4, num_stages=2)
        self.desire.zero_()
        _mix_kernel[(num_reqs,)](
            token_ids, token_ids.stride(0), idx_mapping, req_states.total_len.gpu, num_sampled,
            req_states.last_sampled_tokens.view(-1), self.scratch, self.scratch.stride(0),
            draft_tokens, draft_tokens.stride(0), self.point, self.used, self.hide, self.desire,
            self.L, nb, min(mtp_depth, depth), depth, self.cap,
            MIN_N=MIN_N, STRONG=STRONG, CONFIRM=CONFIRM, K=self.K, K_PO2=triton.next_power_of_2(self.K),
            NB_PO2=triton.next_power_of_2(nb), num_warps=1)
        if draft_logits is not None:
            V = draft_logits.shape[-1]
            _point_logits_kernel[(num_reqs, min(depth, self.K), triton.cdiv(V, _BLOCK_V))](
                self.point, idx_mapping, draft_logits, draft_logits.stride(0), draft_logits.stride(1), V,
                K=self.K, BLOCK_V=_BLOCK_V, num_warps=4)
        slot = self.ring[self.next_slot]
        self.next_slot = (self.next_slot + 1) % _RING
        slot["host"].copy_(self.desire, non_blocking=True)
        slot["event"].record()
        slot["posted"] = True

    def latest_desire(self):
        """Newest landed desired depth (never waits); None if no copy has landed."""
        for back in range(1, _RING + 1):
            slot = self.ring[(self.next_slot - back) % _RING]
            if slot["posted"] and slot["event"].query():
                return int(slot["host"][0])
        return None

    def depth(self, chosen, D):
        """Verify depth for this step: the policy's, raised to the continuation's (sets D.drafted for trim).

        The MTP drafter keeps drafting `chosen` tokens; apply() fills the positions past it."""
        if (D is None or not D.on or not getattr(D, "active", False)
                or torch.cuda.is_current_stream_capturing()):
            return chosen
        want = self.latest_desire()
        if not want:
            return chosen
        lowest = min(D.fullset) if getattr(D, "fullset", None) else 1
        d = max(chosen, max(lowest, min(want, self.cap, D.K)))
        if d != chosen:
            if not self.announced:
                logger.info("MBX ngram mix: depth raised %d -> %d for a repeated span", chosen, d)
                self.announced = True
            D.drafted = d
        return d
