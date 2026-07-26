# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterable

import numpy as np
import torch

from vllm_ascend.worker.v2.pcp.contracts import HybridPCPLayout


def build_linear_prefill_lengths(
    num_scheduled_tokens: np.ndarray,
    is_prefilling: np.ndarray,
    *,
    pcp_world_size: int,
    cp_interleave: int,
) -> np.ndarray:
    """Return ``[num_reqs, pcp_size]`` contiguous causal split lengths."""

    lengths = np.zeros(
        (num_scheduled_tokens.shape[0], pcp_world_size),
        dtype=np.int32,
    )
    lengths[~is_prefilling, :] = num_scheduled_tokens[~is_prefilling, None]
    prefill_lens = num_scheduled_tokens[is_prefilling, None]
    if prefill_lens.size == 0:
        return lengths

    rank_offsets = np.arange(pcp_world_size, dtype=np.int32)[None, :] * cp_interleave
    base = prefill_lens // cp_interleave // pcp_world_size * cp_interleave
    remainder = prefill_lens - base * pcp_world_size
    lengths[is_prefilling] = base + np.clip(
        remainder - rank_offsets,
        0,
        cp_interleave,
    )
    return lengths


def count_decode_prefix_tokens(
    num_scheduled_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> int:
    num_decode_reqs = int(np.count_nonzero(~is_prefilling))
    if np.any(is_prefilling[:num_decode_reqs]) or np.any(~is_prefilling[num_decode_reqs:]):
        raise RuntimeError("Hybrid PCP requires decode requests before prefill requests.")
    return int(num_scheduled_tokens[:num_decode_reqs].sum())


def _inverse_unique_mapping(
    mapping: np.ndarray,
    expected_size: int,
    name: str,
) -> np.ndarray:
    valid_positions = np.flatnonzero(mapping >= 0)
    values = mapping[valid_positions]
    if values.size != expected_size:
        raise ValueError(f"{name} contains {values.size} real tokens; expected {expected_size}.")
    if expected_size and (
        values.min() != 0 or values.max() != expected_size - 1 or np.unique(values).size != expected_size
    ):
        raise ValueError(f"{name} is not a permutation of global prefill.")
    inverse = np.empty(expected_size, dtype=np.int64)
    inverse[values] = valid_positions
    return inverse


def build_hybrid_pcp_layout(
    *,
    linear_prefill_by_rank: Iterable[np.ndarray],
    fa_prefill_by_rank: Iterable[np.ndarray],
    pcp_rank: int,
    num_decode_tokens: int,
    global_num_tokens: int,
    linear_num_tokens: int,
    fa_num_tokens: int,
    device: torch.device,
) -> HybridPCPLayout:
    """Build bridge indices from padded local-to-global maps."""

    linear_maps = tuple(np.asarray(mapping, dtype=np.int64) for mapping in linear_prefill_by_rank)
    fa_maps = tuple(np.asarray(mapping, dtype=np.int64) for mapping in fa_prefill_by_rank)
    if len(linear_maps) != len(fa_maps) or not linear_maps:
        raise ValueError("Linear and FA maps must cover the same PCP ranks.")
    if not 0 <= pcp_rank < len(linear_maps):
        raise ValueError(f"Invalid PCP rank {pcp_rank}.")

    global_prefill_tokens = global_num_tokens - num_decode_tokens
    linear_flat = np.concatenate(linear_maps)
    fa_flat = np.concatenate(fa_maps)
    linear_restore = _inverse_unique_mapping(
        linear_flat,
        global_prefill_tokens,
        "linear gathered layout",
    )
    fa_restore = _inverse_unique_mapping(
        fa_flat,
        global_prefill_tokens,
        "FA gathered layout",
    )

    current_fa = fa_maps[pcp_rank]
    global_to_fa = np.where(current_fa >= 0, current_fa, 0)
    current_linear = linear_maps[pcp_rank]
    fa_to_linear = np.zeros(current_linear.shape[0], dtype=np.int64)
    valid_linear = current_linear >= 0
    fa_to_linear[valid_linear] = fa_restore[current_linear[valid_linear]]

    linear_num_tokens_padded = num_decode_tokens + current_linear.shape[0]
    fa_num_tokens_padded = num_decode_tokens + current_fa.shape[0]
    linear_valid_mask = np.zeros(linear_num_tokens_padded, dtype=np.bool_)
    linear_valid_mask[:num_decode_tokens] = True
    linear_valid_mask[num_decode_tokens:][valid_linear] = True

    def to_device(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.int64, device=device)

    return HybridPCPLayout(
        num_decode_tokens=num_decode_tokens,
        global_num_tokens=global_num_tokens,
        linear_num_tokens=linear_num_tokens,
        linear_num_tokens_padded=linear_num_tokens_padded,
        fa_num_tokens=fa_num_tokens,
        fa_num_tokens_padded=fa_num_tokens_padded,
        hybrid_linear_ag_restore_idx=to_device(linear_restore),
        hybrid_global_to_fa_idx=to_device(global_to_fa),
        hybrid_fa_to_linear_idx=to_device(fa_to_linear),
        linear_valid_mask=torch.as_tensor(
            linear_valid_mask,
            dtype=torch.bool,
            device=device,
        ),
    )
