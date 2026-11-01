# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

# One Triton launch for the n-gram context instead of ~11 torch ops (each eager
# op costs GPU idle on the RTX Spark); 0 = the torch ops.
_FUSED_NGRAM_CONTEXT = os.environ.get("RTX_SPARK_FUSED_NGRAM_CONTEXT", "1") == "1"


@triton.jit
def _ngram_context_kernel(
    context_ptr,  # [max_num_reqs, CONTEXT_LEN] int32
    idx_mapping_ptr,  # [num_reqs] batch row -> request state row
    num_computed_ptr,  # [max_num_reqs] tokens already in the KV cache
    all_token_ids_ptr,  # [max_num_reqs, max_model_len]
    token_row_stride,
    num_reqs,
    eos_token_id,
    CONTEXT_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    in_context = offsets < CONTEXT_LEN
    tokens = tl.full([BLOCK], eos_token_id, dtype=tl.int32)
    if row < num_reqs:
        req = tl.load(idx_mapping_ptr + row).to(tl.int64)
        end = tl.load(num_computed_ptr + req).to(tl.int64)
        positions = end - CONTEXT_LEN + offsets
        valid = in_context & (positions >= 0)
        tokens = tl.load(
            all_token_ids_ptr + req * token_row_stride + tl.maximum(positions, 0),
            mask=valid,
            other=eos_token_id,
        ).to(tl.int32)
    tl.store(context_ptr + row * CONTEXT_LEN + offsets, tokens, mask=in_context)


# After a step's drafting, predict the next step's PLE inputs and let the table reader fetch their rows early.
_NEXT_STEP_PLE_PREFETCH = os.environ.get("MBX_PLE_NEXT_STEP_PREFETCH", "1") == "1"


@triton.jit
def _next_ple_inputs_kernel(
    input_ids_ptr,  # [num_reqs * (1 + num_drafts)] predicted next-step input ids
    query_start_loc_ptr,  # [max_num_reqs + 1]
    context_ptr,  # [max_num_reqs, CONTEXT_LEN]
    idx_mapping_ptr,
    last_sampled_ptr,  # [max_num_reqs]
    draft_tokens_ptr,  # [max_num_reqs, num_speculative_steps]
    draft_stride,
    num_computed_ptr,  # [max_num_reqs], already advanced past this step's accepted tokens
    all_token_ids_ptr,
    token_row_stride,
    num_reqs,
    num_drafts,
    eos_token_id,
    CONTEXT_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Next step of a decode row: [last sampled token, its drafts], as combine_sampled_and_draft_tokens builds it,
    # with the n-gram context _prepare_ngram_context will read then.
    row = tl.program_id(0)
    width = 1 + num_drafts
    tl.store(query_start_loc_ptr + row, tl.minimum(row, num_reqs) * width)
    if row == 0:
        tl.store(query_start_loc_ptr + tl.num_programs(0), num_reqs * width)
    offsets = tl.arange(0, BLOCK)
    in_context = offsets < CONTEXT_LEN
    tokens = tl.full([BLOCK], eos_token_id, dtype=tl.int32)
    if row < num_reqs:
        req = tl.load(idx_mapping_ptr + row).to(tl.int64)
        end = tl.load(num_computed_ptr + req).to(tl.int64)
        positions = end - CONTEXT_LEN + offsets
        valid = in_context & (positions >= 0)
        tokens = tl.load(all_token_ids_ptr + req * token_row_stride + tl.maximum(positions, 0), mask=valid,
                         other=eos_token_id).to(tl.int32)
        base = row * width
        tl.store(input_ids_ptr + base, tl.load(last_sampled_ptr + req).to(tl.int32))
        d = tl.arange(0, BLOCK_D)
        drafts = tl.load(draft_tokens_ptr + req * draft_stride + d, mask=d < num_drafts, other=0)
        tl.store(input_ids_ptr + base + 1 + d, drafts.to(tl.int32), mask=d < num_drafts)
    tl.store(context_ptr + row * CONTEXT_LEN + offsets, tokens, mask=in_context)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        num_steps = vllm_config.num_speculative_tokens
        self._next_input_ids = torch.zeros(
            self.max_num_reqs * (num_steps + 1), dtype=torch.int32, device=self.device
        )
        self._next_query_start_loc = torch.zeros_like(self.ple_query_start_loc)
        self._next_context = torch.zeros_like(self.ngram_context)

    def prefetch_next_ple(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        draft_tokens: torch.Tensor,
        num_drafts: int,
    ) -> None:
        """After drafting: post the PLE rows of the predicted next step to the table reader (rows only, never
        what a later read returns)."""
        if (
            not self.uses_ngram_embedding
            or not _NEXT_STEP_PLE_PREFETCH
            or input_batch.num_reqs == 0
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        from vllm.models.qwen4_exp.common.ngram_embedding import ple_prefetch_next

        num_reqs = input_batch.num_reqs
        num_tokens = num_reqs * (1 + num_drafts)
        all_token_ids = req_states.all_token_ids.gpu
        _next_ple_inputs_kernel[(self.max_num_reqs,)](
            self._next_input_ids,
            self._next_query_start_loc,
            self._next_context,
            input_batch.idx_mapping,
            req_states.last_sampled_tokens,
            draft_tokens,
            draft_tokens.stride(0),
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            num_reqs,
            num_drafts,
            self.ngram_eos_token_id,
            CONTEXT_LEN=self.ngram_context_len,
            BLOCK=triton.next_power_of_2(self.ngram_context_len),
            BLOCK_D=triton.next_power_of_2(max(draft_tokens.shape[1], 1)),
        )
        ple_prefetch_next(
            self._next_input_ids[:num_tokens],
            self._next_query_start_loc,
            self._next_context,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        if _FUSED_NGRAM_CONTEXT:
            # The fill + gather + mask below as one launch.
            all_token_ids = req_states.all_token_ids.gpu
            _ngram_context_kernel[(context.shape[0],)](
                context,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                num_reqs,
                self.ngram_eos_token_id,
                CONTEXT_LEN=self.ngram_context_len,
                BLOCK=triton.next_power_of_2(self.ngram_context_len),
            )
            return context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
