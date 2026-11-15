# SPDX-License-Identifier: Apache-2.0
"""Store native-quantized indexer keys and one scale per token in one kernel.

Keep native quantization and its INT8 tie behavior. The destination scale dtype
controls the final conversion (FP16 or FP32). Inputs must be contiguous and the
key width a power of two. Valid cache slots must be unique; padded negative
slots are skipped. K and scale may have different page strides.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["capacity", "key_block_stride", "scale_block_stride"])
def _store_indexer_key_scale(
    quant_ptr,
    scales_ptr,
    slots_ptr,
    cache_ptr,
    scale_cache_ptr,
    capacity,
    key_block_stride,
    scale_block_stride,
    KEY_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, KEY_WIDTH)
    key = tl.load(quant_ptr + row * KEY_WIDTH + column)
    scale = tl.load(scales_ptr + row)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    page = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    tl.store(cache_ptr + page * key_block_stride + offset * KEY_WIDTH + column, key, valid)
    tl.store(scale_cache_ptr + page * scale_block_stride + offset, scale, valid)


def store_indexer_key_scale(
    quant: torch.Tensor,
    scales: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    block_size: int = 1,
    key_block_stride: int | None = None,
    scale_block_stride: int | None = None,
) -> None:
    """Publish at the existing cache-write point, after the connector wait."""
    rows = slots.numel()
    key_width = quant.shape[-1]
    key_block_stride = block_size * key_width if key_block_stride is None else key_block_stride
    scale_block_stride = block_size if scale_block_stride is None else scale_block_stride
    # Rows only set the launch grid; 12 was a test setting, not a kernel limit.
    _store_indexer_key_scale[(rows,)](
        quant,
        scales,
        slots,
        cache,
        scale_cache,
        cache.numel() // key_width,
        key_block_stride,
        scale_block_stride,
        key_width,
        block_size,
    )
