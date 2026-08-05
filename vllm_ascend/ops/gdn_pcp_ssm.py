# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DualChunk Hybrid PCP state correction for GDN prefill."""

from __future__ import annotations

import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_ascend.ops.triton.fla.chunk_delta_hupdate import (
    chunk_gated_delta_rule_fwd_hupdate,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)

_CHUNK_SIZE = 64


def _pad_rows(tensor: torch.Tensor, capacity: int) -> torch.Tensor:
    if tensor.shape[0] > capacity:
        raise RuntimeError(f"DualChunk PCP segment count exceeds collective capacity: {tensor.shape[0]} > {capacity}.")
    out = tensor.new_zeros((capacity, *tensor.shape[1:]))
    out[: tensor.shape[0]].copy_(tensor)
    return out


def _compute_local_transition(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
) -> torch.Tensor:
    h_update = chunk_gated_delta_rule_fwd_hupdate(
        k=k,
        w=w,
        u=u,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        update_chunk_offsets=prepare_update_chunk_offsets(cu_seqlens, _CHUNK_SIZE),
        num_decodes=0,
    )
    final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    return h_update[:, final_chunk_indices, :, :, :].squeeze(0)


def _scan_dual_chunk_states(
    local_final: torch.Tensor,
    local_transition: torch.Tensor,
    valid: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan canonical ``[request, chunk]`` affine state transformations."""
    previous = torch.empty_like(local_final)
    state = initial
    for chunk_idx in range(local_final.shape[1]):
        previous[:, chunk_idx] = state
        corrected = local_final[:, chunk_idx] + torch.matmul(
            local_transition[:, chunk_idx],
            state,
        )
        state = torch.where(
            valid[:, chunk_idx, None, None, None],
            corrected,
            state,
        )
    return previous, state


def _recompute_local_h(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, v_new, _final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    return h.transpose(1, 2).contiguous(), v_new.transpose(1, 2).contiguous()


def correct_dual_chunk_pcp_ssm_state(
    *,
    local_final_state: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_capacity: int,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correct head/tail segment inputs and return globally final local rows.

    Hybrid PCP currently rejects prefix caching and continued prefills, so each
    segment's provisional run starts from zero. The affine segment summary is
    therefore ``state_out = local_final + transition @ state_in``.
    """
    if segment_capacity <= 0 or segment_capacity % 2:
        raise RuntimeError(f"DualChunk PCP segment capacity must be a positive even number, got {segment_capacity}.")
    num_segments = local_final_state.shape[0]
    if segment_ids.shape[0] != num_segments:
        raise RuntimeError(
            "DualChunk PCP SSM metadata is not aligned with local prefill rows: "
            f"states={num_segments}, ids={segment_ids.shape[0]}."
        )

    local_transition = _compute_local_transition(
        k=k,
        w=w,
        u=u,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    padded_final = _pad_rows(local_final_state, segment_capacity)
    padded_transition = _pad_rows(local_transition, segment_capacity)
    padded_ids = torch.full(
        (segment_capacity,),
        -1,
        dtype=torch.int64,
        device=segment_ids.device,
    )
    padded_ids[:num_segments] = segment_ids.to(torch.int64)

    pcp_group = get_pcp_group()
    gathered_final = pcp_group.all_gather(padded_final.unsqueeze(0), 0).flatten(0, 1)
    gathered_transition = pcp_group.all_gather(padded_transition.unsqueeze(0), 0).flatten(0, 1)
    gathered_ids = pcp_group.all_gather(padded_ids.unsqueeze(0), 0).flatten(0, 1)

    world_size = pcp_group.world_size
    global_num_reqs = segment_capacity // 2
    num_chunks = 2 * world_size
    total_segments = global_num_reqs * num_chunks
    canonical_final = local_final_state.new_zeros((total_segments, *local_final_state.shape[1:]))
    canonical_transition = local_transition.new_zeros((total_segments, *local_transition.shape[1:]))
    canonical_valid = torch.zeros(
        total_segments,
        dtype=torch.bool,
        device=segment_ids.device,
    )
    valid_payload = (gathered_ids >= 0) & (gathered_ids < total_segments)
    canonical_ids = gathered_ids[valid_payload]
    canonical_final.index_copy_(0, canonical_ids, gathered_final[valid_payload])
    canonical_transition.index_copy_(
        0,
        canonical_ids,
        gathered_transition[valid_payload],
    )
    canonical_valid[canonical_ids] = True

    canonical_final = canonical_final.view(
        global_num_reqs,
        num_chunks,
        *local_final_state.shape[1:],
    )
    canonical_transition = canonical_transition.view(
        global_num_reqs,
        num_chunks,
        *local_transition.shape[1:],
    )
    canonical_valid = canonical_valid.view(global_num_reqs, num_chunks)
    initial = local_final_state.new_zeros((global_num_reqs, *local_final_state.shape[1:]))
    previous, final = _scan_dual_chunk_states(
        canonical_final,
        canonical_transition,
        canonical_valid,
        initial,
    )

    flat_previous = previous.view(total_segments, *previous.shape[2:])
    local_initial = flat_previous.index_select(0, segment_ids.to(torch.int64))
    h, v_new = _recompute_local_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=local_initial,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    local_req_ids = torch.div(
        segment_ids.to(torch.int64),
        num_chunks,
        rounding_mode="floor",
    )
    local_final = final.index_select(0, local_req_ids)
    return h, v_new, local_final
