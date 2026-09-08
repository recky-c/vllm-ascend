# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.distributed.kv_transfer.kv_pool.broadcast_transport as module


@pytest.mark.parametrize("parts", [1, 4])
def test_collective_uses_global_owner_and_waits_for_device_completion(monkeypatch, parts):
    events = []
    stream = SimpleNamespace(wait_event=lambda event: events.append("ready"))
    done = SimpleNamespace(record=lambda stream: events.append("record"), synchronize=lambda: events.append("complete"))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(stream=lambda stream: nullcontext(), Event=lambda: done), raising=False
    )
    broadcast = MagicMock(side_effect=lambda *args, **kwargs: SimpleNamespace(wait=lambda: events.append("wait")))
    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    group = SimpleNamespace(ranks=[8, 10], device_group=object())
    buffers = tuple(torch.zeros(16, dtype=torch.int8) for _ in range(parts))
    transport = module.BroadcastKVPPTransport(group, {"layer": 1}, {"layer": buffers})
    transport.prefetch("layer", object(), stream)
    assert events == ["ready"] + ["wait"] * parts + ["record", "complete"]
    for call, buffer in zip(broadcast.call_args_list, buffers):
        assert call.args[0] is buffer
        assert call.kwargs == {"src": 10, "group": group.device_group, "async_op": True}
