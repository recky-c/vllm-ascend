# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    allocate_shared_indexer_cache,
)


@pytest.mark.parametrize("c8_first", [False, True])
@pytest.mark.parametrize("replication", [1, 2])
def test_mixed_views_share_one_allocation_and_keep_transfer_geometry(c8_first, replication):
    bf16 = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sfa_dcp_replicated_indexer_size=replication,
    )
    c8 = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        sfa_dcp_replicated_indexer_size=replication,
    )
    specs = {"c8": c8, "bf16": bf16} if c8_first else {"bf16": bf16, "c8": c8}
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_shared_indexer_cache(specs, 3, allocate)
    assert allocations == [3 * bf16.page_size_bytes]
    bf_view = raw["bf16"][0].view(torch.bfloat16).view(3, replication * 128, 128)
    data = raw["c8"][0].view(3, replication * 128, 128)
    scales = raw["c8"][1].view(torch.float16).view(3, replication * 128, 1)
    assert bf_view.data_ptr() == data.data_ptr()
    assert scales.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
    assert scales.data_ptr() - data.data_ptr() == data.numel()
    # These are the per-view byte lengths and strides used by KV transfer.
    for view, block_bytes in ((bf_view, replication * 32768), (data, replication * 16384), (scales, replication * 256)):
        assert view[0].numel() * view.element_size() == block_bytes
        assert view.stride(0) * view.element_size() == block_bytes
    data.fill_(7)
    scales.fill_(0.5)
    assert torch.all(data == 7) and torch.all(scales == 0.5)
    bf_view.fill_(1.25)
    assert torch.all(bf_view == 1.25)
    data.fill_(-3)
    scales.fill_(2)
    assert torch.all(data == -3) and torch.all(scales == 2)


@pytest.mark.parametrize("replication", [1, 2, 4])
@pytest.mark.parametrize("manager_block_size", [128, 256])
@pytest.mark.parametrize("c4", [False, True])
def test_page_indexer_components_alias_and_preserve_neighbor_pages(replication, manager_block_size, c4):
    from vllm_ascend.attention.indexer_cache import view_indexer_pages

    # uint8 exercises the packed C4 byte ABI without requiring FP4 arithmetic.
    key_width = 64 if c4 else 128
    scale_dim = 4 if c4 else 1
    scale_dtype = torch.uint8 if c4 else torch.float16
    spec = AscendSFAIndexerCacheSpec(
        block_size=manager_block_size, num_kv_heads=1, head_size=key_width,
        dtype=torch.uint8 if c4 else torch.int8,
        scale_dim=scale_dim, scale_dtype=scale_dtype, cache_sparse_li_c4=c4,
        cache_sparse_li_c8=not c4, page_interleaved=True,
        sfa_dcp_replicated_indexer_size=replication,
    )
    size = 3 * spec.page_size_bytes
    backing = torch.full((size + 32,), 19, dtype=torch.uint8)
    keys, scales = view_indexer_pages(backing[:size], spec, 3, 128)
    pages = 3 * manager_block_size // 128 * replication
    assert keys.shape == (pages, 128, 1, key_width)
    if c4:
        assert scales.shape == (pages, 128, 1, 2, 2)
    page_bytes = spec.page_size_bytes // (manager_block_size // 128 * replication)
    assert keys.stride(0) * keys.element_size() == page_bytes
    assert scales.stride(0) * scales.element_size() == page_bytes
    assert scales.data_ptr() - keys.data_ptr() == 128 * key_width
    assert keys.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
    keys[1].fill_(7)
    scales[1].fill_(2)
    assert torch.all(backing[:page_bytes] == 19)
    assert torch.all(backing[2 * page_bytes:] == 19)
    assert torch.all(keys[1] == 7) and torch.all(scales[1] == 2)


def test_shared_page_indexer_slot_keeps_explicit_layer_capacity():
    from vllm_ascend.attention.indexer_cache import view_indexer_pages
    from dataclasses import replace

    small = AscendSFAIndexerCacheSpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8,
        scale_dim=1, scale_dtype=torch.float16, cache_sparse_li_c8=True, page_interleaved=True,
    )
    large = replace(small, block_size=256)
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_shared_indexer_cache({"small": small, "large": large}, 3, allocate)
    assert allocations == [3 * large.page_size_bytes]
    small_keys, _ = view_indexer_pages(raw["small"], small, 3, 128)
    large_keys, _ = view_indexer_pages(raw["large"], large, 3, 128)
    assert small_keys.shape[0] == 3 and large_keys.shape[0] == 6
    assert small_keys.data_ptr() == large_keys.data_ptr()
