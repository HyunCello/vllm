# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    tokens_stride: tl.constexpr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + row * tokens_stride + step, token, mask=valid)
        previous = index
    for step in range(num_steps, tokens_stride):
        tl.store(tokens_ptr + row * tokens_stride + step, -1, mask=valid)


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    cached_stride_0,
    cached_stride_1,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = req_state * cached_stride_0 + step * cached_stride_1
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class CandidateSampler:
    """The shared DFlash2/LiLiCorr candidate walk and realized proposal cache."""

    def __init__(
        self, max_num_reqs: int, num_steps: int, top_k: int, device: torch.device
    ):
        self.num_steps = num_steps
        self.top_k = top_k
        self.scores = torch.empty(
            max_num_reqs, num_steps, top_k, dtype=torch.float32, device=device
        )
        self.cached_candidate_ids = torch.zeros(
            self.scores.shape, dtype=torch.int64, device=device
        )

    def sample(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
        active_steps: int | None = None,
    ) -> None:
        steps = self.num_steps if active_steps is None else active_steps
        assert 1 <= steps <= self.num_steps
        candidate_ids = candidate_ids.contiguous()
        sample_pos = sample_pos.contiguous()
        idx_mapping = idx_mapping.contiguous()
        block_k = triton.next_power_of_2(self.top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            sample_pos,
            idx_mapping,
            temperature,
            seeds,
            draft_tokens,
            draft_tokens.stride(0),
            self.scores,
            num_steps=steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=draft_logits is not None,
            USE_FP64=use_fp64,
            num_warps=1,
        )

        if draft_logits is not None:
            self._cache_draft_logits(
                candidate_ids, num_reqs * steps, idx_mapping, draft_logits, active_steps=steps
            )

    def _cache_draft_logits(
        self,
        candidate_ids: torch.Tensor,
        num_sample: int,
        idx_mapping: torch.Tensor,
        draft_logits: torch.Tensor,
        active_steps: int | None = None,
    ) -> None:
        steps = self.num_steps if active_steps is None else active_steps
        block_k = triton.next_power_of_2(self.top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self.cached_candidate_ids,
            self.cached_candidate_ids.stride(0),
            self.cached_candidate_ids.stride(1),
            candidate_ids,
            self.scores,
            idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"
    _candidate_top_k_key = "selector_top_k"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.top_k = int(draft_config[self._candidate_top_k_key])
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )
        from vllm._rtx_spark.recipes.qwen3_8.dflash_head_overlap import create_overlap

        self._selector_projection_overlap = create_overlap(vllm_config, device)

    def _draft_tail_active_steps(self, num_reqs):
        if getattr(self, "_draft_tail_force_full", False):
            return self.num_speculative_steps
        lookup = getattr(self, "_draft_tail_lookup", None)
        if lookup is None:
            config = self.vllm_config.speculative_config
            parallel = self.vllm_config.parallel_config
            supported = (
                self._speculator_name == "DFlash2" and self.num_speculative_steps == 7
                and self.max_num_reqs == 16 and not self.enable_adaptive_verification
                and config is not None and config.uses_dynamic_speculative_decoding()
                and tuple(tuple(entry) for entry in (config.num_speculative_tokens_per_batch_size or ()))
                    == ((1, 7, 7), (8, 16, 4))
                and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
                and parallel.data_parallel_size == parallel.prefill_context_parallel_size == parallel.decode_context_parallel_size == 1
            )
            if supported:
                from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
                lookup = tuple(build_dynamic_sd_schedule_lookup(
                    config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.max_num_reqs,
                    vllm_num_speculative_tokens=self.num_speculative_steps,
                ))
            else:
                lookup = ()
            self._draft_tail_lookup = lookup
        return lookup[num_reqs] if 0 < num_reqs < len(lookup) else self.num_speculative_steps

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync_state=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False, num_speculative_tokens=None):
        # A captured B can use its scheduled prefix only when it is the exact
        # live B and agrees with the runner's actual target budget. Otherwise
        # the unchanged eager query8/full7 path remains the safe fallback.
        requested = num_speculative_tokens
        expected = self._draft_tail_active_steps(input_batch.num_reqs)
        manager = self.query_cudagraph_manager
        exact = False
        if (not dummy_run and not is_profile and isinstance(requested, int)
                and requested == expected and expected in (4, 7)
                and manager is not None and manager._graphs_captured):
            desc = manager.dispatch(input_batch.num_reqs,
                input_batch.num_reqs * self.num_query_per_req,
                uniform_token_count=self.num_query_per_req, num_active_loras=0)
            exact = (desc.cg_mode == CUDAGraphMode.FULL
                and desc.num_reqs == input_batch.num_reqs
                and desc.num_tokens == input_batch.num_reqs * self.num_query_per_req
                and desc.uniform_token_count == self.num_query_per_req)
        self._draft_tail_force_full = not exact
        try:
            return super().propose(input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled, num_rejected,
                last_sampled, next_prefill_tokens, temperature, seeds,
                dp_sync_state=dp_sync_state, dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs, is_profile=is_profile or not exact,
                num_speculative_tokens=num_speculative_tokens)
        finally:
            self._draft_tail_force_full = False

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        active_steps = self._draft_tail_active_steps(num_reqs)
        num_sample = num_reqs * active_steps
        indices = self.sample_indices[:num_reqs * self.num_speculative_steps].view(
            num_reqs, self.num_speculative_steps)[:, :active_steps].reshape(-1)
        hidden_states = last_hidden_states[indices].view(num_reqs, active_steps, -1)
        overlap = self._selector_projection_overlap
        overlap_started = False
        if overlap is not None:
            overlap_started = overlap.start(hidden_states, self.model.model.candidate_selector)
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, active_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        if not overlap_started:
            scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
        else:
            scores = overlap.score(self, candidate_ids, unary_logits)
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos[:num_reqs * self.num_speculative_steps].view(
                num_reqs, self.num_speculative_steps)[:, :active_steps].reshape(-1),
            self.sample_idx_mapping[:num_reqs * self.num_speculative_steps].view(
                num_reqs, self.num_speculative_steps)[:, :active_steps].reshape(-1),
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
            active_steps=active_steps,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = candidate_ids.shape[0]
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        return self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
