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

"""Hybrid PCP additive fields and linear partition helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def get_linear_rank_segments(
    pcp_world_size: int,
    pcp_rank: int,
    num_scheduled_tokens: np.ndarray,
    is_prefilling: np.ndarray,
    query_start_loc_np: np.ndarray,
) -> list[RankSegment]:
    """Build the V1-compatible contiguous causal layout for one PCP rank.

    Prefill tokens remain in their original request row and are split into
    contiguous rank-local ranges. Decode tokens are replicated on every rank,
    matching the V2 FA layout's decode semantics.
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


class HybridLinearBatchPartitioner:
    """Materialize the contiguous causal linear batch for Hybrid PCP."""

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
    ) -> tuple[AscendInputBatch, int]:
        """Build this rank's original-request linear PCP batch."""
        if global_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")

        num_scheduled_tokens = global_batch.num_scheduled_tokens
        is_prefilling = global_batch.is_prefilling_np
        segments_by_rank = [
            get_linear_rank_segments(
                self._pcp_world_size,
                rank,
                num_scheduled_tokens,
                is_prefilling,
                global_batch.query_start_loc_np,
            )
            for rank in range(self._pcp_world_size)
        ]
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
        linear_num_tokens_padded = max(per_rank_num_tokens)
        input_buffers = self._input_buffers
        if linear_num_tokens_padded > input_buffers.max_num_tokens:
            raise RuntimeError(
                "Hybrid PCP linear token count exceeds the input buffer size: "
                f"{linear_num_tokens_padded} > {input_buffers.max_num_tokens}."
            )

        gather_idx = async_copy_to_gpu(gather_idx_np, device=self._device)
        torch.index_select(
            global_batch.input_ids,
            0,
            gather_idx,
            out=input_buffers.input_ids[:num_local_tokens],
        )

        local_query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(
            local_num_scheduled_tokens,
            out=local_query_start_loc_np[1:],
        )
        async_copy_to_gpu(
            local_query_start_loc_np,
            out=input_buffers.query_start_loc[: num_reqs + 1],
        )
        local_query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]
        local_start_pos = async_copy_to_gpu(local_start_pos_np, device=self._device)
        prepare_pos_seq_lens(
            self._req_idx[:num_reqs],
            local_query_start_loc,
            local_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_reqs],
        )
        seq_lens = input_buffers.seq_lens[:num_reqs]

        is_padding = input_buffers.is_padding[:linear_num_tokens_padded]
        is_padding[:num_local_tokens].fill_(False)
        is_padding[num_local_tokens:].fill_(True)
        if linear_num_tokens_padded > num_local_tokens:
            input_buffers.input_ids[num_local_tokens:linear_num_tokens_padded].zero_()
            input_buffers.positions[num_local_tokens:linear_num_tokens_padded].zero_()

        local_request_has_tokens = local_num_scheduled_tokens > 0
        cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(
            local_request_has_tokens,
            out=cu_num_logits_np[1:],
        )
        total_num_logits = int(cu_num_logits_np[-1])
        cu_num_logits = async_copy_to_gpu(
            cu_num_logits_np,
            device=self._device,
        )
        logits_indices = combine_sampled_and_draft_tokens(
            input_buffers.input_ids,
            global_batch.idx_mapping,
            self._req_states.last_sampled_tokens,
            local_query_start_loc,
            seq_lens,
            self._req_states.prefill_len.gpu,
            self._req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            1,
        )

        local_num_computed_prefill_tokens_np = np.minimum(
            local_start_pos_np, global_batch.prefill_len_np
        )
        local_is_prefilling_np = (
            local_request_has_tokens
            & (local_num_computed_prefill_tokens_np < global_batch.prefill_len_np)
        )
        seq_lens_cpu_upper_bound_np = (
            local_start_pos_np + local_num_scheduled_tokens
        )
        input_buffers.seq_lens_np[:num_reqs] = seq_lens_cpu_upper_bound_np

        linear_batch = replace(
            global_batch,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=linear_num_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=local_query_start_loc,
            query_start_loc_np=local_query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=global_batch.prefill_len_np,
            num_computed_prefill_tokens_np=local_num_computed_prefill_tokens_np,
            is_prefilling_np=local_is_prefilling_np,
            input_ids=input_buffers.input_ids[:linear_num_tokens_padded],
            positions=input_buffers.positions[:linear_num_tokens_padded],
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
        return linear_batch, linear_num_tokens_padded


def count_decode_prefix_tokens(
    num_scheduled_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> int:
    """Count leading decode tokens under the decode-first hybrid contract."""
    num_decode_reqs = int(np.count_nonzero(~is_prefilling))
    if np.any(is_prefilling[:num_decode_reqs]) or np.any(
        ~is_prefilling[num_decode_reqs:]
    ):
        raise RuntimeError(
            "Hybrid PCP requires decode requests before prefill requests."
        )
    return int(num_scheduled_tokens[:num_decode_reqs].sum())


def build_fa_prefill_global_maps(
    *,
    padded_gather_idx: np.ndarray,
    pcp_world_size: int,
    num_decode_tokens: int,
    fa_tokens_padded: int,
    fa_segments_by_rank: list[list[RankSegment]],
) -> tuple[np.ndarray, ...]:
    """Build per-rank FA prefill maps from upstream DualChunk gather.

    Each map has shape ``[fa_tokens_padded - num_decode_tokens]`` and stores
    global-prefill indices (``global_idx - D``) for real rows, or ``-1`` for FA
    pad rows. ``padded_gather_idx`` is the authoritative token mapping; pad rows
    are detected from DualChunk rank segments (not from gather zeros).
    """
    if len(fa_segments_by_rank) != pcp_world_size:
        raise ValueError("FA segments must cover every PCP rank.")
    decode = num_decode_tokens
    fa_pad = fa_tokens_padded
    prefill_pad = fa_pad - decode
    if prefill_pad < 0:
        raise ValueError("FA padded token count is smaller than decode prefix.")

    expected = fa_pad * pcp_world_size
    if padded_gather_idx.shape[0] != expected:
        raise RuntimeError(
            "Upstream padded_gather_idx width does not match FA pad layout: "
            f"{padded_gather_idx.shape[0]} != {expected}."
        )

    maps: list[np.ndarray] = []
    for rank, segments in enumerate(fa_segments_by_rank):
        num_rank_tokens = sum(segment.num_tokens for segment in segments)
        if num_rank_tokens > fa_pad:
            raise RuntimeError(
                "DualChunk local token count exceeds fa_tokens_padded."
            )
        window = padded_gather_idx[rank * fa_pad : (rank + 1) * fa_pad]
        mapping = np.full(prefill_pad, -1, dtype=np.int64)
        for local in range(decode, num_rank_tokens):
            global_idx = int(window[local])
            if global_idx < decode:
                raise RuntimeError(
                    "FA prefill row mapped to a decode global token; "
                    "decode-first FA layout is violated."
                )
            mapping[local - decode] = global_idx - decode
        maps.append(mapping)
    return tuple(maps)


def _inverse_unique_mapping(
    mapping: np.ndarray,
    expected_size: int,
    name: str,
) -> np.ndarray:
    """Invert a padded local→global map into global→flat AG positions."""
    valid_positions = np.flatnonzero(mapping >= 0)
    values = mapping[valid_positions]
    if values.size != expected_size:
        raise ValueError(
            f"{name} contains {values.size} real tokens; expected {expected_size}."
        )
    if expected_size and (
        values.min() != 0
        or values.max() != expected_size - 1
        or np.unique(values).size != expected_size
    ):
        raise ValueError(f"{name} is not a permutation of global prefill.")
    inverse = np.empty(expected_size, dtype=np.int64)
    inverse[values] = valid_positions
    return inverse


def build_hybrid_pcp_layout(
    *,
    pcp_world_size: int,
    pcp_rank: int,
    device: torch.device,
    num_decode_tokens: int,
    global_num_tokens: int,
    linear_num_tokens: int,
    linear_num_tokens_padded: int,
    fa_num_tokens_padded: int,
    linear_segments_by_rank: list[list[RankSegment]],
    is_prefilling: np.ndarray,
    fa_prefill_global_maps: tuple[np.ndarray, ...],
) -> HybridPCPLayout:
    """Build the three hybrid bridge indices and linear valid mask.

    Index spaces (decode-first; pure decode → empty idx tensors):
      D = num_decode_tokens
      T = global_num_tokens
      L = linear_num_tokens_padded
      F = fa_num_tokens_padded

      ① hybrid_linear_ag_restore_idx [T-D]: AG(linear_prefill) → global_prefill
      ② hybrid_global_to_fa_idx      [F-D]: global_prefill → this-rank fa_prefill
         (gather: ``fa = global_prefill[idx]``)
      ③ hybrid_fa_to_linear_idx      [L-D]: AG(fa_prefill) → this-rank linear_prefill
    """
    if not 0 <= pcp_rank < pcp_world_size:
        raise ValueError(f"Invalid PCP rank {pcp_rank}.")
    if len(fa_prefill_global_maps) != pcp_world_size:
        raise ValueError("FA prefill maps must cover every PCP rank.")
    if len(linear_segments_by_rank) != pcp_world_size:
        raise ValueError("Linear segments must cover every PCP rank.")

    decode = num_decode_tokens
    global_prefill_tokens = global_num_tokens - decode
    linear_prefill_pad = linear_num_tokens_padded - decode
    fa_prefill_pad = fa_num_tokens_padded - decode
    if (
        global_prefill_tokens < 0
        or linear_prefill_pad < 0
        or fa_prefill_pad < 0
    ):
        raise ValueError("Hybrid PCP decode prefix exceeds a view's token count.")

    # ① linear AG restore: global_prefill → rank*(L-D) + local_linear_prefill_offset
    linear_restore = np.full(global_prefill_tokens, -1, dtype=np.int64)
    for rank, segments in enumerate(linear_segments_by_rank):
        for segment in segments:
            if not bool(is_prefilling[segment.global_batch_req_idx]):
                continue
            for offset, global_idx in enumerate(
                range(segment.global_batch_slice.start, segment.global_batch_slice.stop)
            ):
                local_pos = segment.rank_local_batch_slice.start + offset
                if local_pos < decode:
                    raise RuntimeError(
                        "Linear prefill segment overlaps the decode prefix."
                    )
                local_prefill_offset = local_pos - decode
                if local_prefill_offset >= linear_prefill_pad:
                    raise RuntimeError(
                        "Linear prefill offset exceeds the AG pad width."
                    )
                gp = global_idx - decode
                if gp < 0 or gp >= global_prefill_tokens:
                    raise RuntimeError(
                        "Linear segment maps outside global prefill range."
                    )
                if linear_restore[gp] >= 0:
                    raise ValueError(
                        "Duplicate linear ownership for a global prefill token."
                    )
                linear_restore[gp] = rank * linear_prefill_pad + local_prefill_offset
    if global_prefill_tokens and np.any(linear_restore < 0):
        raise ValueError("Linear layout is missing some global prefill tokens.")

    # FA maps: local fa_prefill slot → global_prefill idx (-1 = pad)
    fa_maps = tuple(
        np.asarray(mapping, dtype=np.int64) for mapping in fa_prefill_global_maps
    )
    for mapping in fa_maps:
        if mapping.shape != (fa_prefill_pad,):
            raise ValueError(
                "FA prefill map width must be fa_num_tokens_padded - num_decode_tokens."
            )

    # ② gather idx for this rank's FA prefill rows
    current_fa = fa_maps[pcp_rank]
    global_to_fa = np.where(current_fa >= 0, current_fa, 0)

    # global_prefill → AG(fa_prefill) flat position
    fa_restore = _inverse_unique_mapping(
        np.concatenate(fa_maps) if fa_maps else np.empty(0, dtype=np.int64),
        global_prefill_tokens,
        "FA gathered layout",
    )

    # ③ this-rank linear prefill rows → AG(fa_prefill)
    fa_to_linear = np.zeros(linear_prefill_pad, dtype=np.int64)
    for segment in linear_segments_by_rank[pcp_rank]:
        if not bool(is_prefilling[segment.global_batch_req_idx]):
            continue
        for offset, global_idx in enumerate(
            range(segment.global_batch_slice.start, segment.global_batch_slice.stop)
        ):
            local_prefill_offset = (
                segment.rank_local_batch_slice.start + offset - decode
            )
            fa_to_linear[local_prefill_offset] = fa_restore[global_idx - decode]

    linear_valid_mask = np.arange(linear_num_tokens_padded) < linear_num_tokens

    def to_device(array: np.ndarray) -> torch.Tensor:
        return async_copy_to_gpu(array, device=device)

    return HybridPCPLayout(
        num_decode_tokens=decode,
        linear_num_tokens=linear_num_tokens,
        linear_num_tokens_padded=linear_num_tokens_padded,
        hybrid_linear_ag_restore_idx=to_device(linear_restore),
        hybrid_global_to_fa_idx=to_device(global_to_fa),
        hybrid_fa_to_linear_idx=to_device(fa_to_linear),
        linear_valid_mask=torch.as_tensor(
            linear_valid_mask, dtype=torch.bool, device=device
        ),
    )


def partition_hybrid_batch(
    manager: AscendPCPManager,
    global_batch: AscendInputBatch,
) -> AscendInputBatch:
    """Orchestrate hybrid dual-view partition; return the linear main batch.

    ``partition_batch`` stays DualChunk-only. This wrapper builds FA via that
    path, materializes the contiguous linear batch, fills ``HybridPCPLayout``,
    and publishes ``manager.hybrid``.
    """
    manager.hybrid = None
    fa_batch = manager.partition_batch(global_batch)
    linear_batch, linear_num_tokens_padded = manager.partition_linear_batch(
        global_batch
    )

    num_decode_tokens = count_decode_prefix_tokens(
        global_batch.num_scheduled_tokens,
        global_batch.is_prefilling_np,
    )
    linear_segments_by_rank = [
        get_linear_rank_segments(
            manager.pcp_world_size,
            rank,
            global_batch.num_scheduled_tokens,
            global_batch.is_prefilling_np,
            global_batch.query_start_loc_np,
        )
        for rank in range(manager.pcp_world_size)
    ]
    fa_prefill_global_maps = manager._get_fa_prefill_global_maps(
        num_decode_tokens=num_decode_tokens,
        fa_tokens_padded=fa_batch.num_tokens_after_padding,
    )
    layout = build_hybrid_pcp_layout(
        pcp_world_size=manager.pcp_world_size,
        pcp_rank=manager.pcp_rank,
        device=manager.device,
        num_decode_tokens=num_decode_tokens,
        global_num_tokens=global_batch.num_tokens,
        linear_num_tokens=linear_batch.num_tokens,
        linear_num_tokens_padded=linear_num_tokens_padded,
        fa_num_tokens_padded=fa_batch.num_tokens_after_padding,
        linear_segments_by_rank=linear_segments_by_rank,
        is_prefilling=global_batch.is_prefilling_np,
        fa_prefill_global_maps=fa_prefill_global_maps,
    )
    step_id = manager._hybrid_step_id
    manager._hybrid_step_id = step_id + 1
    manager.hybrid = HybridPreparedStep(
        step_id=step_id,
        global_batch=global_batch,
        linear_batch=linear_batch,
        fa_batch=fa_batch,
        layout=layout,
    )
    return linear_batch


def init_hybrid_pcp(
    *,
    is_hybrid: bool,
    vllm_config: VllmConfig,
    pcp_world_size: int,
    pcp_rank: int,
    device: torch.device,
    req_states: RequestState | None,
    max_num_reqs: int | None,
    max_num_tokens: int | None,
) -> HybridLinearBatchPartitioner | None:
    """Create hybrid PCP process-lifetime resources.

    Returns ``None`` when ``is_hybrid`` is false. Future hybrid-only
    resources should be initialized here (or returned alongside the
    partitioner via a small bag type).
    """
    if not is_hybrid:
        return None
    if max_num_reqs is None or max_num_tokens is None:
        raise ValueError("Hybrid PCP requires max_num_reqs and max_num_tokens.")
    if req_states is None:
        raise ValueError("Hybrid PCP requires req_states.")
    return HybridLinearBatchPartitioner(
        vllm_config=vllm_config,
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=device,
        req_states=req_states,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
    )


@dataclass
class HybridPCPLayout:
    """Hybrid-only fields. FA/global token counts stay on PCPManager batches."""

    num_decode_tokens: int
    linear_num_tokens: int
    linear_num_tokens_padded: int
    hybrid_linear_ag_restore_idx: torch.Tensor
    hybrid_global_to_fa_idx: torch.Tensor
    hybrid_fa_to_linear_idx: torch.Tensor
    linear_valid_mask: torch.Tensor


@dataclass
class HybridPreparedStep:
    """Nested hybrid bag: existing PCPManager batches + additive layout.

    Held as ``AscendPCPManager.hybrid``. Assign / clear the whole object:
    ``manager.hybrid = HybridPreparedStep(...)`` or ``manager.hybrid = None``.
    """

    step_id: int
    global_batch: InputBatch
    linear_batch: InputBatch
    fa_batch: InputBatch
    layout: HybridPCPLayout
