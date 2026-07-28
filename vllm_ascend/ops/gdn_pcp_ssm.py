# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Sequential Hybrid PCP: correct GDN SSM after parallel local chunk.
#
# Algorithm:
#   1. Each rank already ran local chunk_h with its own initial_state (s0)
#   2. Compute per-rank transition matrix for the local segment
#   3. AG (local_final_state, transition)
#   4. Gather + correct prev states: correct_i = final_i + T_i @ (correct_{i-1} - s0)
#   5. Rank > 0: recompute local h with correct_{r-1} as initial
#   6. Decode final_state = last valid rank's kernel end state (AG + pick),
#      NOT the AscendC_final × Triton_Φ formula end (that mismatches step 5)
#
# Call site (thin, inside chunk after local fwd_h, before fwd_o):
#   if get_pcp_group().world_size > 1:
#       h, v_new, final_state = correct_pcp_prefill_ssm_state(...)
#
# Callers must gate on ``get_pcp_group().world_size > 1`` before invoking.

from __future__ import annotations

import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_ascend.ops.triton.fla.chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from vllm_ascend.ops.triton.fla.utils import (
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)

_CHUNK_SIZE = 64


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
    """State transition matrix of this rank's local segment, ``[B, N, H, K, K]``."""
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
    return h_update[:, final_chunk_indices, :, :, :]


def _gather_and_correct_ssm_states(
    *,
    initial_state: torch.Tensor,
    local_final_state: torch.Tensor,
    local_transition: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather local finals/transitions and correct SSM states across ranks.

    Contract: every PCP rank keeps the same ``num_reqs``. Requests with no
    local tokens appear as zero-length rows in ``cu_seqlens`` (see sequential
    partition). Those empty rows are treated as identity so the physical last
    rank matches the last rank that actually owns tokens.

    Returns:
        ``(final_state_for_decode, previous_rank_corrected_state)``
        Rank 0's previous is zeros (unused).
    """
    pcp_group = get_pcp_group()
    world_size = pcp_group.world_size
    my_rank = pcp_group.rank_in_group

    local_valid = (cu_seqlens[1:] - cu_seqlens[:-1]) > 0
    valid_marker = local_valid[:, None, None, None].expand(
        -1,
        local_final_state.shape[1],
        local_final_state.shape[2],
        1,
    )
    final_payload = torch.cat(
        (local_final_state, valid_marker.to(local_final_state.dtype)),
        dim=-1,
    )
    all_final_payload = pcp_group.all_gather(final_payload.unsqueeze(0), 0)
    all_final = all_final_payload[..., :-1]
    valid_by_rank = all_final_payload[:, :, 0, 0, -1] > 0
    all_transition = pcp_group.all_gather(local_transition, 0)

    valid_expanded = valid_by_rank[:, :, None, None, None]
    all_final = torch.where(valid_expanded, all_final, initial_state.unsqueeze(0))
    identity = torch.eye(
        all_transition.shape[-1],
        dtype=all_transition.dtype,
        device=all_transition.device,
    ).view(1, 1, 1, *all_transition.shape[-2:])
    all_transition = torch.where(valid_expanded, all_transition, identity)

    corrected = local_final_state.new_empty(world_size, *local_final_state.shape)
    corrected[0] = all_final[0]
    for rank in range(1, world_size):
        # correct_i = final_i + T_i @ (correct_{i-1} - s0)
        corrected[rank] = all_final[rank] + torch.matmul(
            all_transition[rank],
            corrected[rank - 1] - initial_state,
        )

    final_state = corrected[-1]
    if my_rank == 0:
        prev_state = torch.zeros_like(final_state)
    else:
        prev_state = corrected[my_rank - 1]
    return final_state, prev_state


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute local ``fwd_h`` with a corrected initial state.

    Only needed for rank > 0: their first ``h`` / ``v_new`` used the wrong
    initial (local s0).

    Returns AscendC-layout ``(h, v_new)`` for ``fwd_o``, plus the recomputed
    ``final_state`` from the same kernel (needed for decode write-back).
    """
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
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
    return (
        h.transpose(1, 2).contiguous(),
        v_new.transpose(1, 2).contiguous(),
        final_state,
    )


def _pick_last_valid_end_state(
    local_end_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """All-gather per-rank end states and keep the last valid rank's copy.

    Decode must consume the end state from the same kernel path that produced
    the last prefill tokens. Empty local rows are skipped so a short request
    that only touched early ranks still picks the correct end.
    """
    pcp_group = get_pcp_group()
    local_valid = (cu_seqlens[1:] - cu_seqlens[:-1]) > 0
    valid_marker = local_valid[:, None, None, None].expand(
        -1,
        local_end_state.shape[1],
        local_end_state.shape[2],
        1,
    )
    payload = torch.cat(
        (local_end_state, valid_marker.to(local_end_state.dtype)),
        dim=-1,
    )
    gathered = pcp_group.all_gather(payload.unsqueeze(0).contiguous(), 0)
    all_ends = gathered[..., :-1]
    valid_by_rank = gathered[:, :, 0, 0, -1] > 0

    decode_final = all_ends[0]
    for rank in range(1, gathered.shape[0]):
        decode_final = torch.where(
            valid_by_rank[rank][:, None, None, None],
            all_ends[rank],
            decode_final,
        )
    return decode_final


def correct_pcp_prefill_ssm_state(
    *,
    initial_state: torch.Tensor,
    local_final_state: torch.Tensor,
    local_h: torch.Tensor,
    local_v_new: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correct SSM across PCP ranks after a parallel local ``fwd_h``.

    Args:
        initial_state: ``s0`` used by this rank's local chunk, ``[N, H, K, V]``.
        local_final_state: Uncorrected local final state, ``[N, H, K, V]``.
        local_h: Uncorrected local recurrent states from ``fwd_h``.
        local_v_new: Uncorrected local ``v_new`` from ``fwd_h``.
        k, w, u, g: Same tensors the local chunk used (transition / recompute h).
        cu_seqlens: Prefill cumulative lengths, ``[N + 1]``.
        chunk_indices / chunk_offsets: Optional precomputed chunk meta; if omitted
            the implementation may derive them from ``cu_seqlens``.

    Returns:
        ``(h, v_new, final_state)`` ready for ``fwd_o`` and decode write-back.
        Rank 0 keeps local ``h`` / ``v_new``; rank > 0 recomputes them. All ranks
        share the same decode ``final_state`` from the last valid rank's kernel.
    """
    local_transition = _compute_local_transition(
        k=k,
        w=w,
        u=u,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    # Formula final is only used to derive prev_state for rank>0 recompute.
    # Decode write-back must NOT trust AscendC_final + Triton_Φ composition:
    # last-rank tokens come from Triton recompute, so decode state must too.
    _, prev_state = _gather_and_correct_ssm_states(
        initial_state=initial_state,
        local_final_state=local_final_state,
        local_transition=local_transition,
        cu_seqlens=cu_seqlens,
    )

    local_has_tokens = bool(((cu_seqlens[1:] - cu_seqlens[:-1]) > 0).any().item())
    if get_pcp_group().rank_in_group == 0 or not local_has_tokens:
        h, v_new = local_h, local_v_new
        local_end_state = local_final_state
    else:
        h, v_new, local_end_state = _recompute_local_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=prev_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    final_state = _pick_last_valid_end_state(local_end_state, cu_seqlens)
    return h, v_new, final_state
