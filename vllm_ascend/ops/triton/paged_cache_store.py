# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scatter token rows into page-strided cache components without flattening pages."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _store(values, slots, cache, rows, capacity, value_stride: tl.constexpr,
           block_size: tl.constexpr, block_stride: tl.constexpr, row_stride: tl.constexpr,
           width: tl.constexpr, tile: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, tile)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (row < rows) & (slot >= 0) & (slot < capacity)
    data = tl.load(values + row * value_stride + columns, valid & (columns < width), other=0)
    address = cache + slot // block_size * block_stride + slot % block_size * row_stride + columns
    tl.store(address, data, valid & (columns < width))


def store_paged_cache(values: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor) -> None:
    """Cache axes are [block, token, ...]; each token's components are dense."""
    width = values[0].numel() if values.shape[0] else 0
    if width == 0:
        return
    rows = values.shape[0]
    values = values.view(rows, width)
    if cache.ndim < 3 or cache.stride(-1) != 1 or cache.stride(1) != width:
        raise ValueError("Paged cache stores require dense token records")
    _store[(rows,)](values, slots, cache, rows, cache.shape[0] * cache.shape[1],
                   values.stride(0), cache.shape[1], cache.stride(0), cache.stride(1),
                   width, triton.next_power_of_2(width))
