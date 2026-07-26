# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Protocol

import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    HybridPCPForwardViewProtocol,
)


class PCPCollectiveProtocol(Protocol):
    rank_in_group: int

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int,
    ) -> torch.Tensor: ...


def validate_gdn_linear_inputs(
    mixed_qkv: torch.Tensor,
    num_actual_tokens: int,
    forward_view: HybridPCPForwardViewProtocol,
) -> None:
    if mixed_qkv.shape[0] != forward_view.linear_num_tokens_padded:
        raise RuntimeError(
            "Hybrid PCP GDN must consume the padded linear layout: "
            f"{mixed_qkv.shape[0]} != "
            f"{forward_view.linear_num_tokens_padded}."
        )
    if num_actual_tokens != forward_view.linear_num_tokens:
        raise RuntimeError(
            "Hybrid PCP GDN metadata does not match the linear valid-token "
            f"count: {num_actual_tokens} != "
            f"{forward_view.linear_num_tokens}."
        )


def extract_local_conv_history(
    tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right-aligned local tails without crossing request boundaries."""

    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    offsets = torch.arange(
        width,
        dtype=query_start_loc.dtype,
        device=tokens.device,
    )
    indices = query_start_loc[1:, None] - width + offsets[None, :]
    if tokens.shape[1] == 0:
        safe_tokens = tokens.new_zeros((tokens.shape[0], 1))
    else:
        safe_tokens = tokens
    safe_indices = indices.clamp(0, safe_tokens.shape[1] - 1)
    history = safe_tokens[:, safe_indices].permute(1, 0, 2)
    valid = offsets[None, :] >= (width - seq_lens.clamp(min=0, max=width)[:, None])
    return torch.where(valid[:, None, :], history, 0), seq_lens


def _append_conv_history(
    history: torch.Tensor,
    local_tail: torch.Tensor,
    local_count: torch.Tensor,
) -> torch.Tensor:
    width = history.shape[-1]
    count = local_count.clamp(min=0, max=width)
    positions = torch.arange(
        width,
        dtype=count.dtype,
        device=history.device,
    )[None, None, :]
    shifted_indices = (positions + count[:, None, None]).clamp(max=width - 1)
    shifted_history = torch.gather(
        history,
        2,
        shifted_indices.expand_as(history),
    )
    use_local = positions >= (width - count[:, None, None])
    return torch.where(use_local, local_tail, shifted_history)


def select_pcp_conv_histories(
    local_history: torch.Tensor,
    local_seq_lens: torch.Tensor,
    initial_history: torch.Tensor,
    pcp_group: PCPCollectiveProtocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select previous/final histories from the last effective PCP ranks."""

    marker = local_seq_lens.clamp(
        min=0,
        max=local_history.shape[-1],
    )[:, None, None].expand(
        -1,
        local_history.shape[1],
        1,
    )
    payload = torch.cat(
        (local_history, marker.to(local_history.dtype)),
        dim=-1,
    )
    gathered = pcp_group.all_gather(
        payload.unsqueeze(0).contiguous(),
        0,
    )
    histories = gathered[..., :-1]
    counts_by_rank = gathered[:, :, 0, -1].to(dtype=torch.int64)

    history = initial_history
    previous_history = initial_history
    for rank in range(gathered.shape[0]):
        if rank == pcp_group.rank_in_group:
            previous_history = history
        history = _append_conv_history(
            history,
            histories[rank],
            counts_by_rank[rank],
        )
    return previous_history, history
