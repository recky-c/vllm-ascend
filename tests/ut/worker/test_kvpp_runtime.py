from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.kvpp import (
    KVPPExecutionPlan,
    KVPPPhase,
    KVPPRuntime,
    KVPPScheduler,
    _active_pages,
    validate_local_mtp_layers,
    validate_v1_mtp_layers,
)


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _indexer(index: int) -> str:
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def _scheduler(layer_count: int = 2) -> KVPPScheduler:
    owners = {_layer(index): 0 for index in range(layer_count)}
    return KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners=owners,
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )


def _begin(scheduler: KVPPScheduler) -> None:
    scheduler.begin_forward(
        torch.tensor([[7, 2, -1], [2, 8, 12]], dtype=torch.int32),
        [5, 9],
    )


def test_kvpp_06_execution_plan_bundles_sparse_main_and_indexer():
    owners = {
        _layer(0): 0,
        _indexer(0): 0,
        _layer(1): 1,
        _indexer(1): 1,
    }

    plan = KVPPExecutionPlan.build(owners, (_layer(0), _layer(1)))

    assert plan.cache_bundles == {
        _layer(0): (_layer(0), _indexer(0)),
        _layer(1): (_layer(1), _indexer(1)),
    }


def test_kvpp_07_active_pages_are_fixed_shape_deduplicated_and_masked():
    table = torch.tensor([[7, 2, -1, 0], [2, 8, 12, 0]], dtype=torch.int32)
    original = table.clone()

    pages = _active_pages(table, [5, 9], block_size=4, num_blocks=10)

    assert pages.page_ids.shape == (table.numel(),)
    assert pages.page_ids.tolist() == [2, 2, 7, 8, 10, 10, 10, 10]
    assert pages.valid_mask.tolist() == [True, False, True, True, False, False, False, False]
    assert pages.count_upper_bound == 5
    assert torch.equal(table, original)


def test_kvpp_08_scheduler_phase_and_next_layer_prefetch_sequence():
    scheduler = _scheduler()
    _begin(scheduler)
    assert scheduler._phase is KVPPPhase.FORWARD_ACTIVE

    scheduler.enter_layer(_layer(0))
    assert scheduler._phase is KVPPPhase.LAYER_ENTERED
    assert scheduler._pending_layer == _layer(0)

    scheduler.wait_for_layer(_layer(0))
    assert scheduler._phase is KVPPPhase.LAYER_WAITED
    assert scheduler._pending_layer == _layer(1)
    scheduler.leave_layer(_layer(0))

    scheduler.enter_layer(_layer(1))
    scheduler.wait_for_layer(_layer(1))
    scheduler.leave_layer(_layer(1))
    scheduler.finish_forward()
    assert scheduler._phase is KVPPPhase.IDLE


def test_kvpp_runtime_disabled_lifecycle_is_noop():
    runtime = KVPPRuntime()
    runtime.begin_forward((torch.zeros(1, 1, dtype=torch.int32),), [1])
    runtime.finish_forward(dummy_skip_attn=True)
    runtime.close()
    assert runtime.scheduler is None


def test_validate_local_mtp_layers_rejects_missing_and_extra_draft_cache():
    layers = (_layer(0), _layer(1))
    validate_local_mtp_layers(layers, {_layer(1)}, is_last_pp_rank=True)

    try:
        validate_local_mtp_layers(layers, set(), is_last_pp_rank=True)
        raise AssertionError("expected last-rank MTP cache miss to fail")
    except RuntimeError as exc:
        assert "last pipeline stage" in str(exc)

    try:
        validate_local_mtp_layers(layers, {_layer(1)}, is_last_pp_rank=False)
        raise AssertionError("expected non-last-rank MTP cache to fail")
    except RuntimeError as exc:
        assert "Non-last pipeline stages" in str(exc)


def test_validate_v1_mtp_layers_rejects_owned_draft_cache():
    layers = (_layer(0), _layer(1), _layer(61))
    owners = {_layer(0): 0, _layer(61): 0}
    spec = SimpleNamespace(method="mtp")
    hf = SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1)

    validate_v1_mtp_layers(
        layers,
        {_layer(0): 0},
        speculative_config=spec,
        hf_config=hf,
        is_last_pp_rank=True,
    )
    try:
        validate_v1_mtp_layers(
            layers,
            owners,
            speculative_config=spec,
            hf_config=hf,
            is_last_pp_rank=True,
        )
        raise AssertionError("expected owned MTP cache to fail")
    except RuntimeError as exc:
        assert "replicated outside KVPP" in str(exc)


def test_kvpp_runtime_v1_disabled_begin_is_noop():
    KVPPRuntime().begin_v1_forward(SimpleNamespace(), 1, [1])
