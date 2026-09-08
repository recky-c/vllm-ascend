# SPDX-License-Identifier: Apache-2.0
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.worker.v2.kvpp as module
from vllm_ascend.core.kv_cache_placement import KVPPBufferSpec, KVPPPhysicalCachePlan
from vllm_ascend.worker.v2.kvpp import KVPPRuntime, KVPPScheduler


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(torch, "npu", MagicMock(), raising=False)
    instance = KVPPScheduler(MagicMock(), ("a", "b", "c"))
    yield instance
    instance.close()


def test_no_history_does_not_transfer(scheduler):
    scheduler.schedule_forward(False)
    for name in scheduler.attention_layer_names:
        scheduler.wait_for_layer(name)
    scheduler.complete_forward()
    scheduler.transport.prefetch.assert_not_called()


def test_one_layer_ahead_and_all_layers_once(scheduler):
    scheduler.schedule_forward(True)
    for name in scheduler.attention_layer_names:
        scheduler.wait_for_layer(name)
    scheduler.complete_forward()
    assert [call.args[0] for call in scheduler.transport.prefetch.call_args_list] == ["a", "b", "c"]


def test_next_forward_cannot_replace_pending_transfer(scheduler):
    scheduler.schedule_forward(True)
    with pytest.raises(RuntimeError, match="previous prefetch"):
        scheduler.schedule_forward(False)
    scheduler.complete_forward()


def test_early_exit_drains_future(scheduler):
    scheduler.schedule_forward(True)
    scheduler.complete_forward()
    assert scheduler._prefetch_future is None
    scheduler.schedule_forward(False)


def test_transfer_failure_propagates_and_resets_state(scheduler):
    scheduler.transport.prefetch.side_effect = ValueError("transfer failed")
    scheduler.schedule_forward(True)
    with pytest.raises(ValueError, match="transfer failed"):
        scheduler.complete_forward()
    assert scheduler._prefetch_future is None
    assert not scheduler._has_history


def test_future_remains_pending_until_transport_completes(scheduler):
    started, release = Event(), Event()

    def transfer(*args):
        started.set()
        assert release.wait(5)

    scheduler.transport.prefetch.side_effect = transfer
    scheduler.schedule_forward(True)
    try:
        assert started.wait(5)
        assert not scheduler._prefetch_future.done()
    finally:
        release.set()
        scheduler.complete_forward()


def test_noop_runtime():
    runtime = KVPPRuntime()
    runtime.prepare_forward(True)
    runtime.complete_forward()
    runtime.close()


@pytest.mark.parametrize("granularity", ["tensor", "layer"])
@pytest.mark.parametrize("base_offset", [0, 64])
def test_runtime_aliases_existing_storage_and_excludes_mtp(monkeypatch, granularity, base_offset):
    monkeypatch.setattr(torch, "npu", MagicMock(), raising=False)
    config = SimpleNamespace(size=2, broadcast_granularity=granularity)
    monkeypatch.setattr(module.KVPPConfig, "from_vllm_config", lambda _: config)
    plan = KVPPPhysicalCachePlan(
        {},
        {"main": 0, "indexer": 0},
        {"main": ("main", "indexer"), "mtp": ("mtp",)},
        {name: (KVPPBufferSpec(32, torch.int8, 32),) for name in ("main", "indexer", "mtp")},
        0,
    )
    monkeypatch.setattr(module, "create_kvpp_cache_allocation_plan", lambda *args: plan)
    monkeypatch.setattr(module, "get_kvpp_cache_specs", lambda _: {})
    monkeypatch.setattr(
        module,
        "get_kvpp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2, ranks=[8, 9], cpu_group=object(), device_group=object()),
    )
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda values, value, **kwargs: values.__setitem__(slice(None), [value, value]),
    )
    storage = torch.zeros(128 + base_offset, dtype=torch.int8)
    main = storage[base_offset : base_offset + 64].view(torch.float16)
    indexer = storage[base_offset + 64 :]
    main_impl = SimpleNamespace(layerwise_kv_cache_hook=None)
    mtp_impl = SimpleNamespace(layerwise_kv_cache_hook=None)
    runtime = KVPPRuntime.create_from_cache_layout(
        vllm_config=object(),
        kv_cache_config=SimpleNamespace(num_blocks=2),
        static_forward_context={"main": SimpleNamespace(impl=main_impl), "mtp": SimpleNamespace(impl=mtp_impl)},
        cache_layout=module.KVPPCacheLayout({"main": (main,), "indexer": (indexer,)}),
    )
    try:
        buffers = runtime.scheduler.transport._layer_buffers["main"]
        assert len(buffers) == (1 if granularity == "layer" else 2)
        assert sum(buffer.numel() for buffer in buffers) == 128
        assert all(buffer.untyped_storage().data_ptr() == storage.data_ptr() for buffer in buffers)
        assert mtp_impl.layerwise_kv_cache_hook is None
        assert main_impl.layerwise_kv_cache_hook is runtime.scheduler
        buffers[-1][-1] = 17
        assert storage[-1].item() == 17
    finally:
        runtime.close()
