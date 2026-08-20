from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec

from vllm_ascend.v1.core.kv_cache_placement import (
    _get_allocation_groups,
    get_kvpp_layer_owners,
)


def _config(
    *, kvpp_size: int = 2, mtp: bool = False, pipeline_parallel_size: int = 1
):
    return SimpleNamespace(
        additional_config={"kvpp_size": kvpp_size},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        speculative_config=(SimpleNamespace(method="mtp") if mtp else None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=4,
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
