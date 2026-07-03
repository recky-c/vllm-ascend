# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence
from typing import Any

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}


def spec_debug_trace_enabled() -> bool:
    """Return whether verbose speculative-decoding trace logs are enabled."""
    trace = os.getenv("VLLM_ASCEND_SPEC_DEBUG_TRACE", "")
    legacy = os.getenv("VLLM_ASCEND_SPEC_DEBUG", "")
    return trace.lower() in _TRUE_VALUES or legacy.lower() == "trace"


def spec_debug_max_items(default: int = 8) -> int:
    raw_value = os.getenv("VLLM_ASCEND_SPEC_DEBUG_MAX_ITEMS")
    if raw_value is None:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def tensor_debug_preview(value: Any, max_items: int | None = None) -> Any:
    if value is None:
        return None
    if max_items is None:
        max_items = spec_debug_max_items()
    if not torch.is_tensor(value):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value[:max_items])
        return value

    tensor = value.detach()
    if tensor.ndim == 0:
        return tensor.cpu().item()
    if tensor.ndim == 1:
        return tensor[:max_items].cpu().tolist()
    if tensor.ndim == 2:
        return tensor[:max_items, :max_items].cpu().tolist()

    flattened = tensor.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "values": flattened[:max_items].cpu().tolist(),
    }


def split_flat_token_preview(
    flat_token_ids: torch.Tensor,
    counts: Sequence[int],
    max_items: int | None = None,
) -> list[list[int]]:
    if max_items is None:
        max_items = spec_debug_max_items()

    flat_list = flat_token_ids.detach().reshape(-1).cpu().tolist()
    rows: list[list[int]] = []
    offset = 0
    for count in counts[:max_items]:
        next_offset = offset + int(count)
        rows.append(flat_list[offset:next_offset][:max_items])
        offset = next_offset
    return rows
