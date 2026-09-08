# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

import vllm_ascend.core.kv_cache_placement as placement
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.core.kv_cache_placement import (
    KVPPBufferSpec,
    KVPPPhysicalCachePlan,
    build_kvpp_layer_layout,
    build_layer_cache_bundles,
    map_kvpp_layers_to_owners,
)


def config(tp=2, mtp=False):
    return SimpleNamespace(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=SimpleNamespace(method="mtp") if mtp else None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=5, num_nextn_predict_layers=1)),
    )


def name(i):
    return f"model.layers.{i}.self_attn.attn"


def test_owner_order_and_empty_owner_rank():
    names = [name(i) for i in range(5)]
    owners = map_kvpp_layers_to_owners(config(), reversed(names))
    assert list(owners.values()) == [0, 0, 0, 1, 1]
    assert owners == map_kvpp_layers_to_owners(config(), names)
    assert list(map_kvpp_layers_to_owners(config(8), names).values()) == list(range(5))


def test_pp_local_names_and_mtp_exclusion():
    names = [name(3), name(4), name(5)]
    assert map_kvpp_layers_to_owners(config(mtp=True), names) == {name(3): 0, name(4): 1}
    assert map_kvpp_layers_to_owners(config(mtp=True), [name(5)]) == {}


def test_main_precedes_indexer_despite_input_order():
    main = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16)
    indexer = AscendSFAIndexerCacheSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.int8)
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    assert build_layer_cache_bundles({indexer_name: indexer, name(0): main}) == {name(0): (name(0), indexer_name)}


def make_plan(rank=0):
    # Non-aligned sizes exercise the budget's integer search and padding.
    specs = {
        name(i): (KVPPBufferSpec(size, torch.int8, 32), KVPPBufferSpec(3, torch.float16, 32))
        for i, size in enumerate((31, 65, 17, 129, 9, 257))
    }
    logical = {
        key: SimpleNamespace(page_size_bytes=sum(p.size_per_block for p in parts)) for key, parts in specs.items()
    }
    return KVPPPhysicalCachePlan(
        logical, {name(i): i // 3 for i in range(5)}, {key: (key,) for key in specs}, specs, rank
    )


@pytest.mark.parametrize("blocks", [0, 1, 17, 1026])
@pytest.mark.parametrize("rank", [0, 1, 7])
def test_physical_cost_includes_mtp_and_two_max_target_scratch(blocks, rank):
    plan = make_plan(rank)
    sizes = {
        key: build_kvpp_layer_layout(bundle, plan.tensor_specs, blocks)[1] for key, bundle in plan.layer_bundles.items()
    }
    expected = sum(size for key, size in sizes.items() if plan.layer_owner_ranks.get(key, rank) == rank)
    expected += 2 * max(sizes[name(i)] for i in range(5))
    assert plan.get_physical_memory_bytes(blocks) == expected


@pytest.mark.parametrize("budget", [0, 1, 100, 4096, 65535, 1000000])
def test_block_budget_is_maximal_and_planner_uses_logical_bytes(budget):
    plan = make_plan()
    blocks = plan.get_num_blocks(budget)
    assert plan.get_physical_memory_bytes(blocks) <= budget
    assert plan.get_physical_memory_bytes(blocks + 1) > budget
    assert plan.get_planner_memory_bytes(budget) == blocks * sum(
        s.page_size_bytes for s in plan.logical_cache_spec.values()
    )


def test_layout_preserves_offsets_and_has_no_final_padding():
    specs = {"main": (KVPPBufferSpec(33, torch.int8, 32),), "scale": (KVPPBufferSpec(2, torch.float16, 32),)}
    layout, size = build_kvpp_layer_layout(("main", "scale"), specs, 1)
    assert layout == {"main": ((0, 33),), "scale": ((64, 2),)}
    assert size == 66
    assert build_kvpp_layer_layout(("main", "scale"), specs, 0)[1] == 0


@pytest.mark.parametrize("replicated", [1, 8])
@pytest.mark.parametrize("c8", [False, True])
def test_indexer_data_and_scale_bytes(c8, replicated):
    spec = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8 if c8 else torch.bfloat16,
        scale_dim=1 if c8 else 0,
        scale_dtype=torch.float16,
        sfa_dcp_replicated_indexer_size=replicated,
    )
    parts = placement.build_kvpp_buffer_specs(config(), {"indexer": spec})["indexer"]
    assert len(parts) == (2 if c8 else 1)
    assert parts[0].size_per_block == replicated * 128 * 128 * (1 if c8 else 2)
    assert sum(part.size_per_block for part in parts) == spec.page_size_bytes
    if c8:
        assert parts[1].size_per_block == replicated * 128 * 2


def test_group_budget_preserves_complete_logical_spec(monkeypatch):
    cfg = config()
    cfg.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    specs = {
        name(i): MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16) for i in range(5)
    }
    monkeypatch.setattr(placement, "enable_sfa", lambda _: True)
    plan = placement.create_kvpp_cache_allocation_plan(cfg, specs, 0)
    assert plan.logical_cache_spec == specs
    assert len(plan.logical_cache_spec) == 5
    assert not hasattr(plan, "physical_cache_spec")


def test_mixed_c8_real_specs_keep_one_upstream_group():
    cfg = config(mtp=True)
    cfg.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    main = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16
    )
    specs = {name(0): main, "model.layers.0.self_attn.indexer.k_cache": indexer, name(5): main}
    plan = placement.create_kvpp_cache_allocation_plan(cfg, specs, 0)
    assert len(plan.tensor_specs[name(0)]) == 1
    assert len(plan.tensor_specs["model.layers.0.self_attn.indexer.k_cache"]) == 2
    assert name(5) not in plan.layer_owner_ranks
    assert plan.logical_cache_spec == specs


def test_invalid_block_sizes_rejected_before_upstream_grouping():
    specs = {
        name(i): MLAAttentionSpec(block_size=size, num_kv_heads=1, head_size=32, dtype=torch.float16)
        for i, size in enumerate((16, 32))
    }
    with pytest.raises(ValueError, match="common block size"):
        placement.create_kvpp_cache_allocation_plan(config(), specs, 0)


def test_mtp_only_plan_has_no_scratch():
    spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True
    )
    cfg = config(mtp=True)
    cfg.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    plan = placement.create_kvpp_cache_allocation_plan(cfg, {name(5): spec}, 0)
    assert not plan.layer_owner_ranks
    assert plan.get_physical_memory_bytes(17) == 17 * spec.page_size_bytes


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_exact_fit_and_one_byte_short(rank):
    plan = make_plan(rank)
    size = plan.get_physical_memory_bytes(17)
    assert plan.get_num_blocks(size) == 17
    assert plan.get_num_blocks(size - 1) == 16
