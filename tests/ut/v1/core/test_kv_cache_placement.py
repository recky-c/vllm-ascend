from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.v1.core.kv_cache_placement import (
    _get_allocation_groups,
    build_kvpp_physical_cache_plan,
    get_kvpp_layer_owners,
    project_kv_cache_groups_to_worker,
)


def _config(
    *,
    mtp: bool = False,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 2,
    num_hidden_layers: int = 4,
):
    return SimpleNamespace(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=(SimpleNamespace(method="mtp") if mtp else None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=num_hidden_layers,
                num_nextn_predict_layers=1,
            )
        ),
    )


def _spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )


def test_owner_plan_keeps_mtp_cache_replicated():
    target_layers = [f"model.layers.{index}.self_attn.attn" for index in range(4)]
    mtp_layers = [
        "model.layers.4.mtp_block.self_attn.attn",
        "model.layers.4.mtp_block.self_attn.attn.indexer.k_cache",
    ]

    owners = get_kvpp_layer_owners(_config(mtp=True), [*target_layers, *mtp_layers])

    assert [owners[name] for name in target_layers] == [0, 0, 1, 1]
    assert not set(mtp_layers).intersection(owners)


def test_owner_plan_allows_non_last_pp_stage_without_mtp_cache():
    target_layers = [
        f"model.layers.{index}.self_attn.attn" for index in range(2)
    ]

    owners = get_kvpp_layer_owners(
        _config(mtp=True, pipeline_parallel_size=2), target_layers
    )

    assert owners == {target_layers[0]: 0, target_layers[1]: 1}


def test_owner_plan_order_is_stable_for_reordered_cache_names():
    attn_layers = [
        f"model.layers.{index}.self_attn.attn" for index in range(3)
    ]
    indexer_layers = [
        f"model.layers.{index}.self_attn.indexer.k_cache" for index in range(3)
    ]
    names = [
        name
        for layer_names in zip(attn_layers, indexer_layers)
        for name in layer_names
    ]

    forward = get_kvpp_layer_owners(_config(), names)
    reverse = get_kvpp_layer_owners(_config(), reversed(names))

    assert list(forward.items()) == list(reverse.items())


def test_allocation_keeps_two_scratch_caches_per_layout():
    layer_names = [f"model.layers.{index}.self_attn.attn" for index in range(3)]
    spec = _spec()
    worker_spec = dict.fromkeys(layer_names, spec)
    group = KVCacheGroupSpec(layer_names, spec)
    owners = dict.fromkeys(layer_names, 1)

    allocation_groups, scratch_aliases = _get_allocation_groups(
        [group], worker_spec, owners, kvpp_rank=0
    )

    assert allocation_groups[0].layer_names == layer_names[:2]
    assert scratch_aliases == {
        layer_names[0]: [layer_names[0], layer_names[2]],
        layer_names[1]: [layer_names[1]],
    }


# ---------------------------------------------------------------------------
# PP-local placement tests (PP projection -> local owners -> local allocation)
# ---------------------------------------------------------------------------


def _pp_placement_fixture(*, mtp: bool):
    """Target layers 0~7, optional MTP layer 8, PP=2, KVPP=2.

    PP0 worker holds layers 0~3; PP1 worker holds layers 4~7 plus MTP 8.
    """
    config = _config(
        mtp=mtp,
        pipeline_parallel_size=2,
        num_hidden_layers=8,
    )
    target_names = [f"model.layers.{i}.self_attn.attn" for i in range(8)]
    mtp_names = ["model.layers.8.mtp_block.self_attn.attn"] if mtp else []
    spec = _spec()

    global_group = KVCacheGroupSpec(target_names + mtp_names, spec)
    pp0_spec = dict.fromkeys(target_names[:4], spec)
    pp1_spec = dict.fromkeys(target_names[4:] + mtp_names, spec)
    return config, global_group, pp0_spec, pp1_spec


def test_pp0_placement_is_local_and_mtp_free():
    config, _, pp0_spec, _ = _pp_placement_fixture(mtp=True)

    plan = build_kvpp_physical_cache_plan(config, pp0_spec, kvpp_rank=0)
    assert plan is not None
    local_layers = set(plan.physical_spec)

    # PP0 owns target 0~1 persistent, target 2~3 scratch, no MTP.
    assert not any("mtp_block" in name for name in local_layers)
    assert all(name in local_layers for name in list(pp0_spec)[:2])


def test_pp1_placement_replicates_mtp_on_every_kvpp_rank():
    config, _, _, pp1_spec = _pp_placement_fixture(mtp=True)
    mtp_name = "model.layers.8.mtp_block.self_attn.attn"

    for kvpp_rank in (0, 1):
        plan = build_kvpp_physical_cache_plan(
            config, pp1_spec, kvpp_rank
        )
        assert plan is not None
        # MTP must be allocated in full on every KVPP rank.
        assert mtp_name in plan.physical_spec
        # Both KVPP ranks still allocate their own target persistent part.
        target_part = [
            name for name in plan.physical_spec if "mtp_block" not in name
        ]
        assert len(target_part) >= 2


def test_mtp_never_enters_owners_or_scratch_aliases():
    config, global_group, _, pp1_spec = _pp_placement_fixture(mtp=True)
    mtp_name = "model.layers.8.mtp_block.self_attn.attn"
    pp_local = project_kv_cache_groups_to_worker([global_group], pp1_spec)

    owners = get_kvpp_layer_owners(config, pp1_spec)
    _, scratch_aliases = _get_allocation_groups(
        pp_local, pp1_spec, owners, kvpp_rank=0
    )

    assert mtp_name not in owners
    assert mtp_name not in scratch_aliases


def test_projection_keeps_group_index_positions():
    config = _config()
    target_names = [f"model.layers.{i}.self_attn.attn" for i in range(8)]
    mtp_name = "model.layers.8.mtp_block.self_attn.attn"
    spec = _spec()
    global_groups = [
        KVCacheGroupSpec(target_names[:4], spec),
        KVCacheGroupSpec(target_names[4:] + [mtp_name], spec),
    ]
    pp0_spec = dict.fromkeys(target_names[:4], spec)

    projected = project_kv_cache_groups_to_worker(global_groups, pp0_spec)

    # Group count and index positions stay global; PP0's second group is empty.
    assert len(projected) == 2
    assert projected[0].layer_names == target_names[:4]
    assert projected[1].layer_names == []


def test_allocation_rejects_foreign_layers_after_projection():
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    spec = _spec()
    foreign_group = KVCacheGroupSpec(layer_names, spec)
    local_spec = dict.fromkeys(layer_names[:2], spec)
    owners = {layer_names[0]: 0, layer_names[1]: 1}

    with pytest.raises(ValueError, match="outside the current PP stage"):
        _get_allocation_groups(
            [foreign_group], local_spec, owners, kvpp_rank=0
        )


def test_physical_plan_restores_logical_groups_and_scratch_aliases():
    config = _config()
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    spec = _spec()
    worker_spec = dict.fromkeys(layer_names, spec)
    plan = build_kvpp_physical_cache_plan(config, worker_spec, kvpp_rank=0)
    assert plan is not None

    physical_names = list(plan.physical_spec)
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(size=1024, shared_by=[name])
            for name in physical_names
        ],
        kv_cache_groups=[KVCacheGroupSpec(physical_names, spec)],
    )

    plan.apply_to_config(kv_cache_config)

    assert set(kv_cache_config.kv_cache_groups[0].layer_names) == set(
        layer_names
    )
    assert {
        layer_name
        for tensor in kv_cache_config.kv_cache_tensors
        for layer_name in tensor.shared_by
    } == set(layer_names)
