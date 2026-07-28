# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Single-view sequential Hybrid PCP: one linear split for all layers."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def get_linear_rank_segments(
    pcp_world_size: int,
    pcp_rank: int,
    num_scheduled_tokens: np.ndarray,
    is_prefilling: np.ndarray,
    query_start_loc_np: np.ndarray,
) -> list[RankSegment]:
    """Build contiguous causal (sequential) segments for one PCP rank.

    Prefill tokens stay on the original request row and are split into
    contiguous rank-local ranges. Decode tokens are replicated on every rank.
    """
    rank_segments = []
    rank_offset = 0
    for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
        query_len = int(num_tokens)
        if query_len == 0:
            continue

        global_batch_start = int(query_start_loc_np[global_batch_req_idx])
        if bool(is_prefilling[global_batch_req_idx]):
            base_len, remainder = divmod(query_len, pcp_world_size)
            segment_len = base_len + int(pcp_rank < remainder)
            segment_offset = base_len * pcp_rank + min(pcp_rank, remainder)
        else:
            segment_len = query_len
            segment_offset = 0

        if segment_len == 0:
            continue
        segment_start = global_batch_start + segment_offset
        rank_segments.append(
            RankSegment(
                global_batch_req_idx=global_batch_req_idx,
                global_batch_slice=slice(
                    segment_start, segment_start + segment_len
                ),
                rank_local_batch_slice=slice(
                    rank_offset, rank_offset + segment_len
                ),
            )
        )
        rank_offset += segment_len
    return rank_segments


class LinearBatchPartitioner:
    """Materialize the sequential causal local batch (single view)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        req_states: RequestState,
        max_num_reqs: int,
        max_num_tokens: int,
    ) -> None:
        self._vllm_config = vllm_config
        self._pcp_world_size = pcp_world_size
        self._pcp_rank = pcp_rank
        self._device = device
        self._req_states = req_states
        self._input_buffers = AscendInputBuffers(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        self._req_idx = torch.arange(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

    def partition(
        self,
        global_batch: AscendInputBatch,
    ) -> tuple[AscendInputBatch, int, list[list[RankSegment]]]:
        """Build this rank's local batch and return all-rank segments."""
        if global_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")

        segments_by_rank = self._build_segments_by_rank(global_batch)
        (
            local_num_scheduled_tokens,
            local_start_pos_np,
            gather_idx_np,
            num_local_tokens,
            num_tokens_padded,
        ) = self._plan_local_tokens(global_batch, segments_by_rank)

        if num_tokens_padded > self._input_buffers.max_num_tokens:
            raise RuntimeError(
                "Linear PCP token count exceeds the input buffer size: "
                f"{num_tokens_padded} > {self._input_buffers.max_num_tokens}."
            )

        query_start_loc_np, query_start_loc, seq_lens, is_padding = (
            self._fill_local_token_buffers(
                global_batch,
                local_num_scheduled_tokens,
                local_start_pos_np,
                gather_idx_np,
                num_local_tokens,
                num_tokens_padded,
            )
        )
        logits_indices, cu_num_logits, cu_num_logits_np = self._build_local_logits(
            global_batch,
            local_num_scheduled_tokens,
            query_start_loc,
            seq_lens,
        )
        local_batch = self._assemble_local_batch(
            global_batch,
            local_num_scheduled_tokens,
            local_start_pos_np,
            num_local_tokens,
            num_tokens_padded,
            query_start_loc_np,
            query_start_loc,
            seq_lens,
            is_padding,
            logits_indices,
            cu_num_logits,
            cu_num_logits_np,
        )
        return local_batch, num_tokens_padded, segments_by_rank

    def _build_segments_by_rank(
        self,
        global_batch: AscendInputBatch,
    ) -> list[list[RankSegment]]:
        """Sequential split for every PCP rank."""
        return [
            get_linear_rank_segments(
                self._pcp_world_size,
                rank,
                global_batch.num_scheduled_tokens,
                global_batch.is_prefilling_np,
                global_batch.query_start_loc_np,
            )
            for rank in range(self._pcp_world_size)
        ]

    def _plan_local_tokens(
        self,
        global_batch: AscendInputBatch,
        segments_by_rank: list[list[RankSegment]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Derive per-req counts, start positions, gather idx, and pad width."""
        per_rank_num_tokens = [
            sum(segment.num_tokens for segment in segments)
            for segments in segments_by_rank
        ]
        local_segments = segments_by_rank[self._pcp_rank]
        num_reqs = global_batch.num_reqs
        local_num_scheduled_tokens = np.zeros(num_reqs, dtype=np.int32)
        local_start_pos_np = global_batch.num_computed_tokens_np.copy()
        gather_idx_np = np.empty(per_rank_num_tokens[self._pcp_rank], dtype=np.int64)
        for segment in local_segments:
            req_idx = segment.global_batch_req_idx
            local_num_scheduled_tokens[req_idx] = segment.num_tokens
            local_start_pos_np[req_idx] = (
                global_batch.num_computed_tokens_np[req_idx]
                + segment.global_batch_slice.start
                - global_batch.query_start_loc_np[req_idx]
            )
            gather_idx_np[segment.rank_local_batch_slice] = np.arange(
                segment.global_batch_slice.start,
                segment.global_batch_slice.stop,
                dtype=np.int64,
            )
        num_local_tokens = int(local_num_scheduled_tokens.sum())
        num_tokens_padded = max(per_rank_num_tokens) if per_rank_num_tokens else 0
        return (
            local_num_scheduled_tokens,
            local_start_pos_np,
            gather_idx_np,
            num_local_tokens,
            num_tokens_padded,
        )

    def _fill_local_token_buffers(
        self,
        global_batch: AscendInputBatch,
        local_num_scheduled_tokens: np.ndarray,
        local_start_pos_np: np.ndarray,
        gather_idx_np: np.ndarray,
        num_local_tokens: int,
        num_tokens_padded: int,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather tokens and write positions / seq_lens / padding flags."""
        num_reqs = global_batch.num_reqs
        input_buffers = self._input_buffers

        gather_idx = async_copy_to_gpu(gather_idx_np, device=self._device)
        torch.index_select(
            global_batch.input_ids,
            0,
            gather_idx,
            out=input_buffers.input_ids[:num_local_tokens],
        )

        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(local_num_scheduled_tokens, out=query_start_loc_np[1:])
        async_copy_to_gpu(
            query_start_loc_np,
            out=input_buffers.query_start_loc[: num_reqs + 1],
        )
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]
        local_start_pos = async_copy_to_gpu(local_start_pos_np, device=self._device)
        prepare_pos_seq_lens(
            self._req_idx[:num_reqs],
            query_start_loc,
            local_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_reqs],
        )
        seq_lens = input_buffers.seq_lens[:num_reqs]

        is_padding = input_buffers.is_padding[:num_tokens_padded]
        is_padding[:num_local_tokens].fill_(False)
        is_padding[num_local_tokens:].fill_(True)
        if num_tokens_padded > num_local_tokens:
            input_buffers.input_ids[num_local_tokens:num_tokens_padded].zero_()
            input_buffers.positions[num_local_tokens:num_tokens_padded].zero_()

        return query_start_loc_np, query_start_loc, seq_lens, is_padding

    def _build_local_logits(
        self,
        global_batch: AscendInputBatch,
        local_num_scheduled_tokens: np.ndarray,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Build logits indices for requests that still own local tokens."""
        num_reqs = global_batch.num_reqs
        local_request_has_tokens = local_num_scheduled_tokens > 0
        cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(local_request_has_tokens, out=cu_num_logits_np[1:])
        total_num_logits = int(cu_num_logits_np[-1])
        cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self._device)
        logits_indices = combine_sampled_and_draft_tokens(
            self._input_buffers.input_ids,
            global_batch.idx_mapping,
            self._req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self._req_states.prefill_len.gpu,
            self._req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            1,
        )
        return logits_indices, cu_num_logits, cu_num_logits_np

    def _assemble_local_batch(
        self,
        global_batch: AscendInputBatch,
        local_num_scheduled_tokens: np.ndarray,
        local_start_pos_np: np.ndarray,
        num_local_tokens: int,
        num_tokens_padded: int,
        query_start_loc_np: np.ndarray,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        is_padding: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
    ) -> AscendInputBatch:
        """Pack buffers into the Ascend local InputBatch."""
        num_reqs = global_batch.num_reqs
        input_buffers = self._input_buffers
        local_request_has_tokens = local_num_scheduled_tokens > 0
        local_num_computed_prefill_tokens_np = np.minimum(
            local_start_pos_np, global_batch.prefill_len_np
        )
        local_is_prefilling_np = local_request_has_tokens & (
            local_num_computed_prefill_tokens_np < global_batch.prefill_len_np
        )
        seq_lens_cpu_upper_bound_np = local_start_pos_np + local_num_scheduled_tokens
        input_buffers.seq_lens_np[:num_reqs] = seq_lens_cpu_upper_bound_np

        return replace(
            global_batch,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=num_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=global_batch.prefill_len_np,
            num_computed_prefill_tokens_np=local_num_computed_prefill_tokens_np,
            is_prefilling_np=local_is_prefilling_np,
            input_ids=input_buffers.input_ids[:num_tokens_padded],
            positions=input_buffers.positions[:num_tokens_padded],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
            seq_lens_np=input_buffers.seq_lens_np[:num_reqs],
            attn_state=build_attn_state(
                self._vllm_config,
                input_buffers.seq_lens_np[:num_reqs],
                num_reqs,
                local_num_scheduled_tokens,
                local_num_scheduled_tokens,
            ),
        )


def build_linear_hidden_restore_idx(
    *,
    pcp_world_size: int,
    device: torch.device,
    global_num_tokens: int,
    linear_num_tokens_padded: int,
    segments_by_rank: list[list[RankSegment]],
    is_prefilling: np.ndarray,
) -> torch.Tensor:
    """Build DualChunk-compatible ``_hidden_restore_idx`` for sequential layout.

    Matches upstream ``restore_hidden_states``:
    ``AG(full local hidden padded by rank)[idx] → global_hidden`` with shape ``T``.
    Decode is replicated on every rank; only rank 0's decode slots are indexed
    (same convention as DualChunk). Prefill slots use the owning rank.
    """
    if len(segments_by_rank) != pcp_world_size:
        raise ValueError("Segments must cover every PCP rank.")

    restore = np.full(global_num_tokens, -1, dtype=np.int64)
    padded = linear_num_tokens_padded
    for rank, segments in enumerate(segments_by_rank):
        rank_base = rank * padded
        for segment in segments:
            # Decode is replicated; keep a single source (rank 0), like DualChunk.
            if not bool(is_prefilling[segment.global_batch_req_idx]) and rank != 0:
                continue
            local_start = segment.rank_local_batch_slice.start
            local_stop = segment.rank_local_batch_slice.stop
            g_start = segment.global_batch_slice.start
            g_stop = segment.global_batch_slice.stop
            if g_stop - g_start != local_stop - local_start:
                raise RuntimeError("Segment global/local lengths differ.")
            if local_stop > padded:
                raise RuntimeError("Segment exceeds the padded local width.")
            slot = restore[g_start:g_stop]
            if np.any(slot >= 0):
                raise ValueError("Duplicate sequential ownership for a global token.")
            restore[g_start:g_stop] = np.arange(
                rank_base + local_start,
                rank_base + local_stop,
                dtype=np.int64,
            )

    if global_num_tokens and np.any(restore < 0):
        raise ValueError("Sequential layout is missing some global tokens.")

    return async_copy_to_gpu(restore, device=device)


def init_linear_pcp(
    *,
    is_hybrid: bool,
    vllm_config: VllmConfig,
    pcp_world_size: int,
    pcp_rank: int,
    device: torch.device,
    req_states: RequestState | None,
    max_num_reqs: int | None,
    max_num_tokens: int | None,
) -> LinearBatchPartitioner | None:
    """Create process-lifetime resources for sequential Hybrid PCP."""
    if not is_hybrid:
        return None
    if max_num_reqs is None or max_num_tokens is None:
        raise ValueError("Linear PCP requires max_num_reqs and max_num_tokens.")
    if req_states is None:
        raise ValueError("Linear PCP requires req_states.")
    return LinearBatchPartitioner(
        vllm_config=vllm_config,
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=device,
        req_states=req_states,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
    )


def partition_sequential_batch(
    manager: AscendPCPManager,
    global_batch: AscendInputBatch,
) -> AscendInputBatch:
    """Partition with one sequential view; reuse upstream Manager fields.

    Writes:
      - ``manager._global_batch``
      - ``manager._hidden_restore_idx`` (DualChunk-compatible full-AG semantics)
    Returns the local ``InputBatch`` as the Runner main path (pad via
    ``num_tokens`` / ``is_padding``). Does not call DualChunk ``partition_batch``.
    """
    local_batch, linear_num_tokens_padded, segments_by_rank = (
        manager.partition_linear_batch(global_batch)
    )
    manager._global_batch = global_batch
    manager._hidden_restore_idx = build_linear_hidden_restore_idx(
        pcp_world_size=manager.pcp_world_size,
        device=manager.device,
        global_num_tokens=global_batch.num_tokens,
        linear_num_tokens_padded=linear_num_tokens_padded,
        segments_by_rank=segments_by_rank,
        is_prefilling=global_batch.is_prefilling_np,
    )
    return local_batch
