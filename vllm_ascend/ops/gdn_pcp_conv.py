# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DualChunk Hybrid PCP convolution-state transfer for GDN prefill.

Each PCP rank owns a head and a tail segment of an original request. Both
virtual rows point at the same persistent cache slot, so they cannot use that
slot as their causal-convolution workspace. This module builds one temporary
state row per local segment, restores histories in global chunk order, and
writes only the final request history back to the persistent cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.distributed import get_pcp_group


def _pad_rows(tensor: torch.Tensor, capacity: int, value: int | float = 0) -> torch.Tensor:
    if tensor.shape[0] > capacity:
        raise RuntimeError(f"DualChunk PCP segment count exceeds collective capacity: {tensor.shape[0]} > {capacity}.")
    out = tensor.new_full((capacity, *tensor.shape[1:]), value)
    out[: tensor.shape[0]].copy_(tensor)
    return out


def _extract_local_tails(
    tokens_dim_major: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right-aligned tails ``[segments, dim, width]`` and lengths."""
    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    offsets = torch.arange(
        width,
        dtype=query_start_loc.dtype,
        device=tokens_dim_major.device,
    )
    indices = query_start_loc[1:, None] - width + offsets[None, :]
    safe = (
        tokens_dim_major
        if tokens_dim_major.shape[1] > 0
        else tokens_dim_major.new_zeros((tokens_dim_major.shape[0], 1))
    )
    gathered = safe[:, indices.clamp(0, safe.shape[1] - 1)].permute(1, 0, 2)
    valid = offsets[None, :] >= (width - seq_lens.clamp(min=0, max=width)[:, None])
    return torch.where(valid[:, None, :], gathered, 0), seq_lens


def _append_tail(
    history: torch.Tensor,
    local_tail: torch.Tensor,
    local_count: torch.Tensor,
) -> torch.Tensor:
    width = history.shape[-1]
    count = local_count.clamp(min=0, max=width)
    pos = torch.arange(
        width,
        dtype=count.dtype,
        device=history.device,
    )[None, None, :]
    shifted = torch.gather(
        history,
        2,
        (pos + count[:, None, None]).clamp(max=width - 1).expand_as(history),
    )
    return torch.where(pos >= (width - count[:, None, None]), local_tail, shifted)


def _fold_dual_chunk_histories(
    tails: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold canonical ``[request, chunk]`` histories from left to right."""
    num_chunks = tails.shape[1]
    previous = torch.empty_like(tails)
    history = initial
    for chunk_idx in range(num_chunks):
        previous[:, chunk_idx] = history
        updated = _append_tail(
            history,
            tails[:, chunk_idx],
            counts[:, chunk_idx],
        )
        history = torch.where(valid[:, chunk_idx, None, None], updated, history)
    return previous, history


@dataclass
class DualChunkPcpConvPlan:
    """Temporary segment cache and final persistent-cache write-back."""

    conv_state: torch.Tensor
    cache_indices: torch.Tensor
    initial_state_mode: torch.Tensor
    _persistent_conv_state: torch.Tensor
    _final_state_indices: torch.Tensor
    _final_histories: torch.Tensor
    _state_len: int

    def write_back_final_state(self) -> None:
        valid = self._final_state_indices >= 0
        self._persistent_conv_state[self._final_state_indices[valid], : self._state_len, :] = self._final_histories[
            valid
        ].transpose(-1, -2)


def prepare_dual_chunk_pcp_conv(
    *,
    conv_state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_capacity: int,
    state_len: int,
) -> DualChunkPcpConvPlan:
    """Prepare independent head/tail histories before the prefill conv call."""
    if segment_capacity <= 0 or segment_capacity % 2:
        raise RuntimeError(f"DualChunk PCP segment capacity must be a positive even number, got {segment_capacity}.")
    num_segments = query_start_loc.shape[0] - 1
    if segment_ids.shape[0] != num_segments or state_indices.shape[0] != num_segments:
        raise RuntimeError(
            "DualChunk PCP conv metadata is not aligned with local prefill rows: "
            f"segments={num_segments}, ids={segment_ids.shape[0]}, "
            f"states={state_indices.shape[0]}."
        )

    local_tails, local_lens = _extract_local_tails(
        mixed_qkv.transpose(0, 1),
        query_start_loc,
        state_len,
    )
    padded_tails = _pad_rows(local_tails, segment_capacity)
    padded_meta = torch.full(
        (segment_capacity, 2),
        -1,
        dtype=torch.int64,
        device=mixed_qkv.device,
    )
    padded_meta[:num_segments, 0] = segment_ids.to(torch.int64)
    padded_meta[:num_segments, 1] = local_lens.to(torch.int64)

    pcp_group = get_pcp_group()
    gathered_tails = pcp_group.all_gather(padded_tails.unsqueeze(0).contiguous(), 0)
    gathered_meta = pcp_group.all_gather(padded_meta.unsqueeze(0).contiguous(), 0)
    flat_tails = gathered_tails.flatten(0, 1)
    flat_meta = gathered_meta.flatten(0, 1)

    world_size = pcp_group.world_size
    global_num_reqs = segment_capacity // 2
    num_chunks = 2 * world_size
    total_segments = global_num_reqs * num_chunks
    canonical_tails = local_tails.new_zeros((total_segments, *local_tails.shape[1:]))
    canonical_counts = torch.zeros(
        total_segments,
        dtype=torch.int64,
        device=mixed_qkv.device,
    )
    valid_payload = (flat_meta[:, 0] >= 0) & (flat_meta[:, 0] < total_segments)
    canonical_ids = flat_meta[valid_payload, 0]
    canonical_tails.index_copy_(0, canonical_ids, flat_tails[valid_payload])
    canonical_counts.index_copy_(0, canonical_ids, flat_meta[valid_payload, 1])

    canonical_tails = canonical_tails.view(
        global_num_reqs,
        num_chunks,
        *local_tails.shape[1:],
    )
    canonical_counts = canonical_counts.view(global_num_reqs, num_chunks)
    canonical_valid = canonical_counts > 0
    initial = local_tails.new_zeros((global_num_reqs, local_tails.shape[1], state_len))
    previous, final = _fold_dual_chunk_histories(
        canonical_tails,
        canonical_counts,
        canonical_valid,
        initial,
    )

    local_previous = previous.view(total_segments, *previous.shape[2:]).index_select(0, segment_ids.to(torch.int64))
    temporary_state = conv_state.new_zeros((num_segments, *conv_state.shape[1:]))
    temporary_state[:, :state_len, :] = local_previous.transpose(-1, -2)
    local_cache_indices = torch.arange(
        num_segments,
        dtype=state_indices.dtype,
        device=state_indices.device,
    )
    initial_state_mode = torch.ones(
        num_segments,
        dtype=torch.bool,
        device=state_indices.device,
    )
    # Cache block ids are rank-local. The runtime guard guarantees every rank
    # owns at least one segment for every prefill request, so derive write-back
    # indices only from this rank's rows rather than a remote payload.
    final_state_indices = torch.full(
        (global_num_reqs,),
        -1,
        dtype=state_indices.dtype,
        device=state_indices.device,
    )
    local_req_ids = torch.div(
        segment_ids.to(torch.int64),
        num_chunks,
        rounding_mode="floor",
    )
    local_chunk_ids = torch.remainder(segment_ids.to(torch.int64), num_chunks)
    local_head = local_chunk_ids < world_size
    final_state_indices.index_copy_(
        0,
        local_req_ids[local_head],
        state_indices[local_head],
    )
    return DualChunkPcpConvPlan(
        conv_state=temporary_state,
        cache_indices=local_cache_indices,
        initial_state_mode=initial_state_mode,
        _persistent_conv_state=conv_state,
        _final_state_indices=final_state_indices,
        _final_histories=final,
        _state_len=state_len,
    )
