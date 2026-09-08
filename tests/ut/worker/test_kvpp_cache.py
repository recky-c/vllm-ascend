# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

import vllm_ascend.worker.kvpp_cache as module
from vllm_ascend.core.kv_cache_placement import KVPPBufferSpec, KVPPPhysicalCachePlan


def test_allocator_uses_target_ordinal_and_mtp_is_persistent(monkeypatch):
    names = [f"layer{i}" for i in range(7)]
    specs = {name: (KVPPBufferSpec(32 * (i + 1), torch.int8, 32),) for i, name in enumerate(names)}
    owners = {name: (0 if i in (1, 4) else 1) for i, name in enumerate(names[:-1])}
    plan = KVPPPhysicalCachePlan({}, owners, {name: (name,) for name in names}, specs, 0)
    monkeypatch.setattr(module, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(module, "get_kvpp_cache_specs", lambda config: {})
    monkeypatch.setattr(module, "create_kvpp_cache_allocation_plan", lambda *args: plan)
    caches = module.allocate_kvpp_cache(object(), SimpleNamespace(num_blocks=17), torch.device("cpu"))
    ptr = lambda i: caches[names[i]][0].untyped_storage().data_ptr()
    assert ptr(0) == ptr(2)
    assert ptr(3) == ptr(5)
    assert ptr(0) != ptr(3)
    assert len({ptr(0), ptr(3), ptr(1), ptr(4), ptr(6)}) == 5
    assert caches[names[0]][0].untyped_storage().nbytes() == 17 * 32 * 6
    assert caches[names[6]][0].untyped_storage().nbytes() == 17 * 32 * 7
    actual = sum(
        storage.nbytes()
        for storage in {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
            for parts in caches.values()
            for tensor in parts
        }.values()
    )
    assert actual == plan.get_physical_memory_bytes(17)
