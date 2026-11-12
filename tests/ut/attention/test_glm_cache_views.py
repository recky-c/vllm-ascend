# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.attention.glm_cache_views import (
    GLMCacheComponents,
    allocate_glm_cache_views,
    component_layout,
    map_glm_cache_components,
    store_paged_components,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


def make_config():
    main = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    return KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(size=1024, layers=["main.0", "main.1"], offset=32, layer_stride=96, block_stride=320),
            KVCacheTensor(size=1024, layers=["indexer.0"], offset=224, layer_stride=40, block_stride=320),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["main.0", "main.1"], kv_cache_spec=main),
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=indexer),
        ],
    )


def allocate(config=None, device="cpu"):
    raw = torch.full((1024,), 33, dtype=torch.int8, device=device)
    allocator = Mock(return_value=raw)
    caches = allocate_glm_cache_views(config or make_config(), [4, 4], allocator)
    allocator.assert_called_once_with(1024)
    return raw, caches


def test_page_major_components_alias_backing_without_touching_neighbors():
    raw, caches = allocate()
    nope, rope = caches["main.0"]
    key, scale = caches["indexer.0"]
    assert isinstance(caches["main.0"], GLMCacheComponents)
    assert nope.shape == (3, 4, 1, 8)
    assert rope.shape == (3, 4, 1, 4)
    assert key.shape == (3, 4, 1, 8)
    assert scale.shape == (3, 4, 1, 1)
    assert nope.stride() == (160, 8, 8, 1)
    assert key.stride() == (320, 8, 8, 1)
    assert scale.stride() == (160, 1, 1, 1)
    for cache in caches.values():
        for component in cache:
            assert component.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert rope.data_ptr() - nope.data_ptr() == 64
    assert scale.data_ptr() - key.data_ptr() == 32
    assert caches["main.1"][0].data_ptr() - nope.data_ptr() == 96
    key[1, 2].fill_(7)
    scale[1, 2].fill_(0.5)
    expected = torch.full_like(raw, 33)
    expected[224 + 320 + 16 : 224 + 320 + 24] = 7
    expected[224 + 320 + 32 + 4 : 224 + 320 + 32 + 6] = torch.tensor([0.5], dtype=torch.float16).view(torch.int8)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("size", 2048, "one shared backing"),
        ("offset", 1000, "exceeds backing"),
        ("offset", 225, "Unaligned"),
        ("block_stride", 16, "geometry mismatch"),
        ("layer_stride", 0, "positive layer stride"),
        ("host_resident", True, "device descriptors"),
    ],
)
def test_invalid_descriptor_rejected_before_allocation(field, value, message):
    config = make_config()
    config.kv_cache_tensors[1] = replace(config.kv_cache_tensors[1], **{field: value})
    allocator = Mock()
    with pytest.raises(ValueError, match=message):
        allocate_glm_cache_views(config, [4, 4], allocator)
    allocator.assert_not_called()


def test_block_split_and_missing_owner_rejected():
    config = make_config()
    allocator = Mock()
    with pytest.raises(ValueError, match="block splitting"):
        allocate_glm_cache_views(config, [2, 4], allocator)
    config.kv_cache_tensors.pop()
    with pytest.raises(ValueError, match="every owner"):
        allocate_glm_cache_views(config, [4, 4], allocator)
    allocator.assert_not_called()


@pytest.mark.parametrize("flag", ["cache_sparse_sfa_c8", "cache_sparse_sfa_turboquant"])
def test_packed_main_preserves_record_width(flag):
    spec = replace(make_config().kv_cache_groups[0].kv_cache_spec, dtype=torch.int8, head_size=386, **{flag: True})
    assert component_layout(spec) == [((1, 386), torch.int8)]


def test_c4_scale_shape_and_replicated_rejection():
    spec = replace(make_config().kv_cache_groups[1].kv_cache_spec, scale_dim=4, cache_sparse_li_c4=True)
    assert component_layout(spec)[1][0] == (1, 2, 2)
    with pytest.raises(ValueError, match="Replicated"):
        component_layout(replace(spec, sfa_dcp_replicated_indexer_size=2))


def test_mapping_preserves_component_identity_and_delegates_other_caches():
    _, caches = allocate()
    groups = [SimpleNamespace(layer_names=list(caches), kernel_block_size=4)]
    legacy = torch.empty(1)
    original = Mock(side_effect=lambda values, groups: values)
    result = map_glm_cache_components({**caches, "other": legacy}, iter(groups), original)
    assert original.call_args.args[0] == {"other": legacy}
    assert result["main.0"] is caches["main.0"]
    groups[0].kernel_block_size = 2
    with pytest.raises(ValueError, match="cannot be split"):
        map_glm_cache_components(caches, groups, original)


def reference_scatter(key, value, key_cache, value_cache, slots):
    # CPU reference validates adaptation only, not native kernel support.
    for updates, cache in ((key, key_cache), (value, value_cache)):
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, token = divmod(slot, cache.shape[1])
                cache[block, token] = updates[row]


def test_byte_store_updates_underlying_storage_and_skips_padding():
    raw, caches = allocate()
    before = raw.clone()
    keys = torch.arange(32, dtype=torch.int8).view(4, 8)
    scales = torch.tensor([[0.5], [1.25], [2.5], [4.0]], dtype=torch.float16)
    slots = torch.tensor([0, 5, -1, 11])
    store_paged_components(caches["indexer.0"], (keys, scales), slots, reference_scatter)
    expected = before.clone()
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, token = divmod(slot, 4)
        start = 224 + block * 320
        expected[start + token * 8 : start + (token + 1) * 8] = keys[row]
        expected[start + 32 + token * 2 : start + 34 + token * 2] = scales[row].view(torch.int8)
    assert torch.equal(raw, expected)


def test_empty_store_does_not_launch_and_dtype_mismatch_rejected():
    _, caches = allocate()
    scatter = Mock()
    store_paged_components(
        caches["indexer.0"],
        (torch.empty(0, 8, dtype=torch.int8), torch.empty(0, 1)),
        torch.empty(0, dtype=torch.int64),
        scatter,
    )
    scatter.assert_not_called()
    with pytest.raises(ValueError, match="dtype/token mismatch"):
        store_paged_components(
            caches["indexer.0"], (torch.zeros(1, 8, dtype=torch.int8), torch.ones(1, 1)), torch.tensor([0]), scatter
        )
