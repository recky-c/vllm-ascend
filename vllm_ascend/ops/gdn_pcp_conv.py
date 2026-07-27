# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Sequential Hybrid PCP: transfer conv history across ranks before GDN prefill.
#
# Each rank only holds a contiguous local segment. Rank r>0 must receive the
# chained trailing conv history from earlier ranks and write it into
# conv_state before causal_conv1d. After conv, all ranks store the final
# chained history for decode.
#
# Callers must gate on ``get_pcp_group().world_size > 1`` before invoking
# these helpers.

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.distributed import get_pcp_group


@dataclass
class PcpPrefillConvTransfer:
    """Result of PCP prefill conv transfer; write back after local causal_conv1d."""

    _conv_state: torch.Tensor
    _cache_indices: torch.Tensor
    _state_len: int
    _final_history: torch.Tensor

    def write_back_final_conv_state(self) -> None:
        """Store the chained final conv history into ``conv_state`` for decode."""
        self._conv_state[self._cache_indices, : self._state_len, :] = (
            self._final_history.transpose(-1, -2)
        )


def _extract_local_tails(
    tokens_dim_major: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-aligned local tails; zero where the request is shorter than width.

    ``tokens_dim_major``: ``[dim, num_tokens]``
    Returns history ``[num_seqs, dim, width]``, seq_lens ``[num_seqs]``.
    """
    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    offsets = torch.arange(width, dtype=query_start_loc.dtype, device=tokens_dim_major.device)
    indices = query_start_loc[1:, None] - width + offsets[None, :]
    if tokens_dim_major.shape[1] == 0:
        safe = tokens_dim_major.new_zeros((tokens_dim_major.shape[0], 1))
    else:
        safe = tokens_dim_major
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
    pos = torch.arange(width, dtype=count.dtype, device=history.device)[None, None, :]
    shifted = torch.gather(
        history,
        2,
        (pos + count[:, None, None]).clamp(max=width - 1).expand_as(history),
    )
    return torch.where(pos >= (width - count[:, None, None]), local_tail, shifted)


def _chain_histories(
    local_history: torch.Tensor,
    local_seq_lens: torch.Tensor,
    initial_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AG local tails and fold left → (previous_for_this_rank, final)."""
    pcp_group = get_pcp_group()
    width = local_history.shape[-1]
    marker = local_seq_lens.clamp(min=0, max=width)[:, None, None].expand(
        -1, local_history.shape[1], 1
    )
    payload = torch.cat((local_history, marker.to(local_history.dtype)), dim=-1)
    gathered = pcp_group.all_gather(payload.unsqueeze(0).contiguous(), 0)
    tails = gathered[..., :-1]
    counts = gathered[:, :, 0, -1].to(dtype=torch.int64)

    history = initial_history
    previous = initial_history
    my_rank = pcp_group.rank_in_group
    for rank in range(gathered.shape[0]):
        if rank == my_rank:
            previous = history
        history = _append_tail(history, tails[rank], counts[rank])
    return previous, history


def force_prefill_initial_state_mode(
    initial_state_mode: torch.Tensor,
    *,
    num_prefills: int,
    prefill_num_rows: int,
) -> torch.Tensor:
    """Mark prefill rows so causal_conv1d reads the transferred conv_state."""
    prefill_seq_offset = max(0, prefill_num_rows - num_prefills)
    out = initial_state_mode.clone()
    out[prefill_seq_offset:] = True
    return out


def transfer_pcp_prefill_conv_state(
    conv_state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_prefills: int,
    state_len: int,
) -> PcpPrefillConvTransfer | None:
    """Inject previous-rank conv history into ``conv_state`` before local conv.

    Caller must already be on the PCP path
    (``get_pcp_group().world_size > 1``). Returns ``None`` when there are no
    prefill cache rows; otherwise returns a handle whose
    ``write_back_final_conv_state()`` must run after causal_conv1d.
    """
    num_seqs = query_start_loc.shape[0] - 1
    prefill_offset = max(0, num_seqs - num_prefills)
    cache_indices = state_indices[prefill_offset:]
    if cache_indices.numel() == 0:
        return None

    local_tail, local_lens = _extract_local_tails(
        mixed_qkv.transpose(0, 1),
        query_start_loc[prefill_offset:],
        state_len,
    )
    initial = conv_state[cache_indices, :state_len, :].transpose(-1, -2)
    previous, final = _chain_histories(local_tail, local_lens, initial)

    if get_pcp_group().rank_in_group > 0:
        conv_state[cache_indices, :state_len, :].copy_(previous.transpose(-1, -2))

    return PcpPrefillConvTransfer(
        _conv_state=conv_state,
        _cache_indices=cache_indices,
        _state_len=state_len,
        _final_history=final,
    )
