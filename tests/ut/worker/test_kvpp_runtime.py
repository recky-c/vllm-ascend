from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.kvpp import (
    KVPPExecutionPlan,
    KVPPRuntime,
    KVPPScheduler,
    _active_pages,
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


def test_kvpp_08_prefetch_starts_at_begin_and_advances_on_wait():
    scheduler = _scheduler()
    _begin(scheduler)
    assert scheduler._pending_layer == _layer(0)
    assert scheduler._next_layer_index == 0

    scheduler.wait_for_layer(_layer(0))
    assert scheduler._pending_layer == _layer(1)
    assert scheduler._next_layer_index == 1

    scheduler.wait_for_layer(_layer(1))
    assert scheduler._pending_layer is None
    assert scheduler._next_layer_index == 2
    scheduler.finish_forward()
    assert scheduler._selected_pages is None


def test_kvpp_runtime_disabled_lifecycle_is_noop():
    runtime = KVPPRuntime()
    runtime.begin_forward((torch.zeros(1, 1, dtype=torch.int32),), [1])
    runtime.finish_forward(dummy_skip_attn=True)
    runtime.close()
    assert runtime.scheduler is None
