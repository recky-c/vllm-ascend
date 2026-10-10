# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout, MambaSpec, SlidingWindowSpec

from vllm_ascend import envs
from vllm_ascend.patch.platform import patch_kv_cache_utils as planner
from vllm_ascend.worker.v2 import attn_utils


def qwen_dflash_specs():
    page = 1310720
    target = FullAttentionSpec(block_size=640, num_kv_heads=2, head_size=256, dtype=torch.bfloat16)
    draft = FullAttentionSpec(block_size=640, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, non_causal=True)
    sliding = SlidingWindowSpec(
        block_size=640, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, sliding_window=4096
    )
    mamba = MambaSpec(
        block_size=128000,
        shapes=((4,), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page,
        mamba_cache_mode="none",
        num_speculative_blocks=3,
    )
    specs = {}
    for prefix, count, spec in (
        ("target_full", 8, target),
        ("draft_swa", 5, sliding),
        ("draft_full", 1, draft),
        ("target_gdn", 24, mamba),
    ):
        specs.update({f"{prefix}.{index}": spec for index in range(count)})
    assert {spec.page_size_bytes for spec in specs.values()} == {page}
    return specs


def test_group_five_preserves_four_semantic_buckets_and_round_robin(monkeypatch):
    specs = qwen_dflash_specs()
    original = dict(specs)
    monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", "5")
    groups = planner._get_kv_cache_groups_uniform_page_size(specs)
    assert len(groups) == 9
    assert max(len(group.layer_names) for group in groups) == 5
    assert sum(5 - len(group.layer_names) for group in groups) == 7
    assert Counter(name for group in groups for name in group.layer_names) == Counter(specs.keys())
    assert Counter(group.layer_names[0].split(".")[0] for group in groups) == {
        "target_full": 2,
        "draft_swa": 1,
        "draft_full": 1,
        "target_gdn": 5,
    }
    for group in groups:
        kinds = {name.split(".")[0] for name in group.layer_names}
        assert len(kinds) == 1
        assert all(specs[name] == group.kv_cache_spec for name in group.layer_names)
    assert groups[0].layer_names == [f"target_full.{index}" for index in (0, 2, 4, 6)]
    assert groups[1].layer_names == [f"target_full.{index}" for index in (1, 3, 5, 7)]
    assert groups[2].kv_cache_spec.sliding_window == 4096
    assert groups[3].kv_cache_spec.non_causal is True
    assert specs == original


def test_group_zero_uses_original_heuristic_exactly(monkeypatch):
    specs = qwen_dflash_specs()
    monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", "0")
    original = planner._orig_get_kv_cache_groups_uniform_page_size(specs)
    with patch.object(
        planner,
        "_orig_get_kv_cache_groups_uniform_page_size",
        wraps=planner._orig_get_kv_cache_groups_uniform_page_size,
    ) as fallback:
        actual = planner._get_kv_cache_groups_uniform_page_size(specs)
    fallback.assert_called_once_with(specs)
    assert actual == original
    assert len(actual) == 38


@pytest.mark.parametrize("value", ["-1", "5.0", "", "true", "+5", " 5", "５"])
def test_group_size_rejects_invalid_environment(monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", value)
    with pytest.raises(ValueError, match="nonnegative integer"):
        _ = envs.VLLM_ASCEND_KV_CACHE_GROUP_SIZE


def test_group_size_default_and_specialized_planner_are_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", raising=False)
    assert envs.VLLM_ASCEND_KV_CACHE_GROUP_SIZE == 0
    monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", "5")
    specialized = [object()]
    monkeypatch.setattr(planner, "_get_kimi_k3_dspark_mixed_kv_cache_groups", lambda _: specialized)
    assert planner._get_kv_cache_groups_uniform_page_size(qwen_dflash_specs()) is specialized


def test_fixed_grouping_rejects_nonuniform_pages():
    specs = qwen_dflash_specs()
    specs["target_full.0"] = replace(specs["target_full.0"], page_size_padded=2621440)
    with pytest.raises(ValueError, match="uniform physical page sizes"):
        planner._get_fixed_width_kv_cache_groups(specs, 5)
    with pytest.raises(ValueError, match="must be positive"):
        planner._get_fixed_width_kv_cache_groups({}, 0)


def test_group_five_allocator_and_mamba_stride_remain_per_layer(monkeypatch):
    specs = qwen_dflash_specs()
    groups = planner._get_fixed_width_kv_cache_groups(specs, 5)
    page = next(iter(specs.values())).page_size_bytes
    assert kv_cache_utils._pool_bytes_per_block(groups) == 5 * page
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
            cache_dtype="auto",
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        additional_config={},
        kv_transfer_config=None,
        quant_config=None,
    )
    cache = planner._orig_get_kv_cache_config_from_groups(config, groups, 5 * page * 4)
    assert cache.num_blocks == 4
    assert len(cache.kv_cache_tensors) == 9
    for tensor in cache.kv_cache_tensors:
        assert tensor.size == 5 * page * 4
        assert tensor.layer_stride == page * 4
        assert tensor.block_stride == page
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    raw = attn_utils._allocate_kv_cache(cache, shared_layers={}, device=torch.device("cpu"))
    assert len({value.untyped_storage().data_ptr() for value in raw.values()}) == 1
    for group in groups:
        ranges = sorted((raw[name].data_ptr(), raw[name].data_ptr() + raw[name].numel()) for name in group.layer_names)
        assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
    group = next(group for group in groups if isinstance(group.kv_cache_spec, MambaSpec))
    states = [attn_utils._reshape_mamba_kv_cache(raw[name], specs[name]) for name in group.layer_names]
    for conv, state in states:
        assert state.shape == (4, 16, 128, 128)
        assert state.stride(0) == 327680
        assert conv.shape[0] == cache.num_blocks
    states[0][1][1].fill_(7)
    assert torch.count_nonzero(states[0][1][0]) == 0
    assert torch.count_nonzero(states[1][1]) == 0


@pytest.mark.parametrize("width,expected_groups,expected_gdn_groups", [(0, 38, 24), (5, 9, 5)])
def test_fixed_groups_feed_reused_gdn_metadata_without_mixing_physical_states(
    monkeypatch, width, expected_groups, expected_gdn_groups
):
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

    from tests.ut.attention.test_batch_metadata_reuse import make_fia_builder
    from tests.ut.ops.test_gdn_attn_builder import _make_builder
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    monkeypatch.setenv("VLLM_ASCEND_KV_CACHE_GROUP_SIZE", str(width))
    monkeypatch.setenv("VLLM_ASCEND_REUSE_BATCH_METADATA", "1")
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    groups = planner._get_kv_cache_groups_uniform_page_size(qwen_dflash_specs())
    assert len(groups) == expected_groups
    builders, gdn_group_ids = [], []
    for group_id, group in enumerate(groups):
        if isinstance(group.kv_cache_spec, MambaSpec):
            builder = _make_builder(device=torch.device("cpu"), num_heads=32, num_speculative_tokens=3)
            builder.kv_cache_spec = group.kv_cache_spec
            gdn_group_ids.append(group_id)
        else:
            builder = make_fia_builder(torch.device("cpu"))
            builder.kv_cache_spec = group.kv_cache_spec
        builders.append(builder)
    assert len(gdn_group_ids) == expected_gdn_groups
    attn_groups = [
        [
            SimpleNamespace(
                kv_cache_spec=group.kv_cache_spec,
                layer_names=group.layer_names,
                get_metadata_builder=lambda _, builder=builder: builder,
            )
        ]
        for group, builder in zip(groups, builders)
    ]
    prior_query = None
    for step, actual_reqs in enumerate([2, 1, 2]):
        query = torch.tensor([0, 4, actual_reqs * 4], dtype=torch.int32)
        lengths = torch.tensor([32 + step * 4, 48 if actual_reqs == 2 else 0], dtype=torch.int32)
        tables = [
            torch.arange(8, dtype=torch.int32).reshape(2, 4) + group * 100 + step * 10000
            for group in range(len(groups))
        ]
        result = attn_utils.build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=2,
            num_actual_reqs=actual_reqs,
            num_tokens=actual_reqs * 4,
            num_actual_tokens=actual_reqs * 4,
            query_start_loc_gpu=query,
            query_start_loc_cpu=query,
            max_query_len=4,
            seq_lens=lengths,
            seq_lens_cpu_upper_bound=lengths,
            max_seq_len=128,
            block_tables=tables,
            slot_mappings=torch.zeros((len(groups), 8), dtype=torch.int64),
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            attn_state=AscendAttentionState.SpecDecoding,
            model_specific_attn_metadata=MambaHybridAttnMetadata(
                is_prefilling=torch.zeros(2, dtype=torch.bool),
                num_accepted_tokens=torch.tensor([step + 1, 2], dtype=torch.int32),
                num_decode_draft_tokens_cpu=torch.tensor([3, 3 if actual_reqs == 2 else -1], dtype=torch.int32),
            ),
        )
        assert set(result) == set(qwen_dflash_specs())
        states, public_queries = [], []
        for group_id in gdn_group_ids:
            group = groups[group_id]
            metadata = result[group.layer_names[0]]
            assert all(result[layer] is metadata for layer in group.layer_names)
            torch.testing.assert_close(
                metadata.spec_state_indices_tensor, tables[group_id][:actual_reqs], rtol=0, atol=0
            )
            states.append(metadata.spec_state_indices_tensor)
            public_queries.append(metadata.spec_query_start_loc)
        assert len({tensor.data_ptr() for tensor in states}) == expected_gdn_groups
        assert len({id(tensor) for tensor in public_queries}) == 1
        assert public_queries[0] is not prior_query
        prior_query = public_queries[0]
