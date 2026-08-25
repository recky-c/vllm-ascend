from types import SimpleNamespace

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
)


def _config(*, mtp: bool = False, layers: int = 5):
    return SimpleNamespace(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(method="mtp") if mtp else None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=layers,
                num_nextn_predict_layers=1,
            )
        ),
    )


def _spec():
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )


def _target(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _indexer(index: int) -> str:
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def test_kvpp_01_owner_is_deterministic_and_bundles_share_owner():
    names = [name for i in range(5) for name in (_target(i), _indexer(i))]

    forward = get_kvpp_layer_owners(_config(), names)
    reverse = get_kvpp_layer_owners(_config(), reversed(names))

    assert forward == reverse
    assert all(forward[_target(i)] == forward[_indexer(i)] for i in range(5))


def test_kvpp_02_remainder_partition_is_three_plus_two():
    names = [_target(i) for i in range(5)]

    owners = get_kvpp_layer_owners(_config(), names)

    assert [owners[name] for name in names] == [0, 0, 0, 1, 1]


def test_kvpp_03_mtp_is_not_owned_and_is_physically_replicated():
    targets = [_target(i) for i in range(5)]
    mtp = "model.layers.5.mtp_block.self_attn.attn"
    worker_spec = dict.fromkeys([*targets, mtp], _spec())
    config = _config(mtp=True)

    owners = get_kvpp_layer_owners(config, worker_spec)
    plans = [build_kvpp_physical_cache_plan(config, worker_spec, rank) for rank in range(2)]

    assert mtp not in owners
    assert all(plan is not None and mtp in plan.physical_spec for plan in plans)


def test_kvpp_04_persistent_and_two_scratch_allocations_alias_round_robin():
    names = [_target(i) for i in range(5)]
    spec = _spec()
    worker_spec = dict.fromkeys(names, spec)
    group = KVCacheGroupSpec(names, spec)
    owners = get_kvpp_layer_owners(_config(), names)

    groups, aliases = _get_allocation_groups([group], worker_spec, owners, kvpp_rank=1)

    # Rank 1 owns two persistent layers; three foreign layers share exactly
    # two alternating scratch tensors.
    # Projection preserves logical group order; the physical set contains the
    # two scratch tensors (0/1) plus this rank's persistent tensors (3/4).
    assert groups[0].layer_names == [names[0], names[1], names[3], names[4]]
    assert aliases == {
        names[0]: [names[0], names[2]],
        names[1]: [names[1]],
    }


def test_kvpp_05_apply_physical_plan_restores_logical_cache_view():
    names = [_target(i) for i in range(5)]
    worker_spec = dict.fromkeys(names, _spec())
    plan = build_kvpp_physical_cache_plan(_config(), worker_spec, kvpp_rank=1)
    assert plan is not None
    physical_names = list(plan.physical_spec)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=[name]) for name in physical_names],
        kv_cache_groups=[KVCacheGroupSpec(physical_names, _spec())],
    )

    plan.apply_to_config(config)

    logical_tensor_names = {name for tensor in config.kv_cache_tensors for name in tensor.shared_by}
    assert logical_tensor_names == set(names)
    assert set(config.kv_cache_groups[0].layer_names) == set(names)
