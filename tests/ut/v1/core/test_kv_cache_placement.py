from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec

from vllm_ascend.v1.core.kv_cache_placement import (
    _get_allocation_groups,
    get_kvpp_layer_owners,
)


def _config(*, kvpp_size: int = 2, mtp: bool = False):
    return SimpleNamespace(
        additional_config={"kvpp_size": kvpp_size},
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


def test_owner_plan_partitions_mtp_layers_like_target_layers():
    """MTP KV layers now participate in KVPP ownership, not replication."""
    target_layers = [f"model.layers.{index}.self_attn.attn" for index in range(4)]
    mtp_layers = [
        "model.layers.4.mtp_block.self_attn.attn",
        "model.layers.4.mtp_block.self_attn.attn.indexer.k_cache",
    ]

    owners = get_kvpp_layer_owners(_config(mtp=True), [*target_layers, *mtp_layers])

    assert [owners[name] for name in target_layers] == [0, 0, 1, 1]
    # MTP layers must have an owner under KVPP.
    assert all(name in owners for name in mtp_layers)
    # MTP layers carry layer index 4 and land in the same partition as the
    # other index-4 target layer (the last one), which is owner rank 1.
    assert all(owners[name] == 1 for name in mtp_layers)


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


def test_allocation_gives_each_draft_layer_a_dedicated_scratch():
    """Draft (MTP/Eagle) layers get one scratch per non-owner layer, no rotation."""
    target_layers = [f"model.layers.{index}.self_attn.attn" for index in range(2)]
    draft_layers = ["model.layers.2.mtp_block.self_attn.attn"]
    spec = _spec()
    worker_spec = dict.fromkeys([*target_layers, *draft_layers], spec)
    target_group = KVCacheGroupSpec(target_layers, spec, is_eagle_group=False)
    draft_group = KVCacheGroupSpec(draft_layers, spec, is_eagle_group=True)
    # All layers owned by rank 1; rank 0 is non-owner for every layer.
    owners = {name: 1 for name in [*target_layers, *draft_layers]}

    allocation_groups, scratch_aliases = _get_allocation_groups(
        [target_group, draft_group], worker_spec, owners, kvpp_rank=0
    )

    # Target scratch still rotates two buffers across the two target layers.
    assert scratch_aliases[target_layers[0]] == [target_layers[0], target_layers[1]]
    # Draft scratch is dedicated: the single MTP layer aliases only itself.
    assert scratch_aliases[draft_layers[0]] == [draft_layers[0]]


def test_allocation_owner_keeps_draft_persistent():
    """Owner of a draft layer keeps it persistent (no scratch alias)."""
    draft_layers = ["model.layers.2.mtp_block.self_attn.attn"]
    spec = _spec()
    worker_spec = dict.fromkeys(draft_layers, spec)
    draft_group = KVCacheGroupSpec(draft_layers, spec, is_eagle_group=True)
    owners = {draft_layers[0]: 0}

    allocation_groups, scratch_aliases = _get_allocation_groups(
        [draft_group], worker_spec, owners, kvpp_rank=0
    )

    assert allocation_groups[0].layer_names == draft_layers
    assert scratch_aliases == {}
