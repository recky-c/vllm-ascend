from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    KVPPBufferMetadata,
    KVPPMTEPeerMetadata,
    MemFabricMTEKVPPTransport,
    _MTEDeviceBufferMetadata,
)
from vllm_ascend.worker.v2.kvpp import (
    KVPPPhase,
    KVPPScheduler,
    _active_pages,
    get_kvpp_managed_group_index,
)


def test_managed_group_allows_replicated_mtp_groups():
    groups = [
        SimpleNamespace(layer_names=["target.0", "mtp.4"]),
        SimpleNamespace(layer_names=["mtp.5"]),
    ]

    assert get_kvpp_managed_group_index(groups, {"target.0": 0}) == 0


def test_managed_group_rejects_target_layers_across_groups():
    groups = [
        SimpleNamespace(layer_names=["target.0", "mtp.4"]),
        SimpleNamespace(layer_names=["target.1", "mtp.5"]),
    ]

    with pytest.raises(ValueError, match="one KV cache group"):
        get_kvpp_managed_group_index(
            groups,
            {"target.0": 0, "target.1": 1},
        )


def test_active_pages_uses_only_pages_covered_by_sequence_lengths():
    block_table = torch.tensor([[7, 2, 9, 0], [4, 8, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)

    original_block_table = block_table.clone()
    pages = _active_pages(block_table, seq_lens, block_size=4, num_blocks=10)

    assert pages.page_ids.tolist() == [2, 4, 7, 8, 10, 10, 10, 10]
    assert pages.valid_mask.tolist() == [True, True, True, True, False, False,
                                        False, False]
    assert pages.page_ids.device == block_table.device
    assert pages.valid_mask.device == block_table.device
    assert pages.count_upper_bound == 4
    assert torch.equal(block_table, original_block_table)


def _active_page_tensor(*page_ids: int) -> KVPPActivePages:
    pages = torch.tensor(page_ids, dtype=torch.int32)
    return KVPPActivePages(
        pages,
        torch.ones_like(pages, dtype=torch.bool),
        count_upper_bound=len(page_ids),
    )


def test_mte_owner_stages_and_consumer_unpacks_same_active_pages(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def copy_op(
        anchor,
        local_offsets,
        staging_offsets,
        lengths,
        staging_base,
        source_rank,
        destination_rank,
        shm_id,
    ):
        calls.append(
            (
                anchor,
                tuple(local_offsets.tolist()),
                tuple(staging_offsets.tolist()),
                tuple(lengths.tolist()),
                staging_base,
                source_rank,
                destination_rank,
                shm_id,
            )
        )

    stream = SimpleNamespace()
    owner_anchor = torch.empty(1)
    owner = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    owner._layers = {"layer": (KVPPBufferMetadata(2000, 16, 16),)}
    owner._anchors = {"layer": owner_anchor}
    owner._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]), torch.tensor([16]), torch.tensor([16]), 16
        )
    }
    owner._local_metadata = KVPPMTEPeerMetadata(8000, 1024, 0)
    owner._peer_metadata = [
        owner._local_metadata,
        KVPPMTEPeerMetadata(8000, 1024, 1),
    ]

    owner.push_active_bundle(("layer",), _active_page_tensor(2, 3, 7), stream)
    assert calls == [
        (
            owner_anchor,
            (32, 48, 112),
            (0, 16, 32),
            (16, 16, 16),
            8000,
            -1,
            1,
            31,
        )
    ]

    calls.clear()
    consumer_anchor = torch.empty(1)
    consumer = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=1, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    consumer._layers = {"layer": (KVPPBufferMetadata(1000, 16, 16),)}
    consumer._anchors = {"layer": consumer_anchor}
    consumer._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]), torch.tensor([16]), torch.tensor([16]), 16
        )
    }
    consumer._local_metadata = KVPPMTEPeerMetadata(8000, 1024, 1)
    consumer._peer_metadata = [
        KVPPMTEPeerMetadata(8000, 1024, 0),
        consumer._local_metadata,
    ]

    consumer.receive_active_bundle(
        ("layer",), _active_page_tensor(2, 3, 7), stream
    )
    assert calls == [
        (
            consumer_anchor,
            (32, 48, 112),
            (0, 16, 32),
            (16, 16, 16),
            8000,
            1,
            -1,
            31,
        )
    ]


def test_mte_builds_one_device_batch_for_masked_pages_and_multiple_buffers(
    monkeypatch,
):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda *args, **kwargs: pytest.fail(
            "MTE capacity validation must not launch a device assertion"
        ),
    )
    calls = []

    def copy_op(anchor, local_offsets, staging_offsets, lengths,
                staging_base, source_rank, destination_rank, shm_id):
        calls.append(
            (
                tuple(local_offsets.tolist()),
                tuple(staging_offsets.tolist()),
                tuple(lengths.tolist()),
                staging_base,
                source_rank,
                destination_rank,
                shm_id,
            )
        )

    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    transport._anchors = {"layer": torch.empty(1)}
    transport._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0, 4000]),
            torch.tensor([32, 64]),
            torch.tensor([16, 8]),
            24,
        )
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 240, 0)
    transport._peer_metadata = [
        transport._local_metadata,
        KVPPMTEPeerMetadata(8000, 240, 1),
    ]
    pages = KVPPActivePages(
        torch.tensor([2, 2, 7, 10], dtype=torch.int32),
        torch.tensor([True, False, True, False]),
        count_upper_bound=2,
    )

    transport.push_active_bundle(("layer",), pages, SimpleNamespace())

    assert calls == [
        (
            (64, 64, 224, 320, 4128, 4128, 4448, 4640),
            (0, 0, 16, 16, 160, 160, 168, 168),
            (16, 0, 16, 0, 8, 0, 8, 0),
            8000,
            -1,
            1,
            31,
        )
    ]


def test_mte_bundle_assigns_disjoint_main_and_indexer_staging(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def copy_op(
        anchor,
        local_offsets,
        staging_offsets,
        lengths,
        staging_base,
        source_rank,
        destination_rank,
        shm_id,
    ):
        calls.append(
            (
                "main" if anchor is main_anchor else "indexer",
                tuple(staging_offsets.tolist()),
                source_rank,
                destination_rank,
            )
        )

    main_anchor = torch.empty(1)
    indexer_anchor = torch.empty(1)
    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"main": 0, "indexer": 0},
        10,
        copy_op=copy_op,
    )
    transport._anchors = {
        "main": main_anchor,
        "indexer": indexer_anchor,
    }
    transport._device_layers = {
        "main": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([16]),
            torch.tensor([16]),
            16,
        ),
        "indexer": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            8,
        ),
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 240, 0)
    transport._peer_metadata = [
        transport._local_metadata,
        KVPPMTEPeerMetadata(8000, 240, 1),
    ]

    transport.push_active_bundle(
        ("main", "indexer"),
        _active_page_tensor(2, 7),
        SimpleNamespace(),
    )

    # 240 / (16 + 8) reserves ten active pages. Main owns [0, 160),
    # and the indexer starts at 160 instead of overwriting main at zero.
    assert calls == [
        ("main", (0, 16), -1, 1),
        ("indexer", (160, 168), -1, 1),
    ]


def test_mte_bundle_uses_identical_layout_when_consumer_unpacks(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def copy_op(
        anchor,
        local_offsets,
        staging_offsets,
        lengths,
        staging_base,
        source_rank,
        destination_rank,
        shm_id,
    ):
        calls.append(
            (
                "main" if anchor is main_anchor else "indexer",
                tuple(staging_offsets.tolist()),
                source_rank,
                destination_rank,
            )
        )

    main_anchor = torch.empty(1)
    indexer_anchor = torch.empty(1)
    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=1, world_size=2),
        {"main": 0, "indexer": 0},
        10,
        copy_op=copy_op,
    )
    transport._anchors = {
        "main": main_anchor,
        "indexer": indexer_anchor,
    }
    transport._device_layers = {
        "main": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([16]),
            torch.tensor([16]),
            16,
        ),
        "indexer": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([8]),
            torch.tensor([8]),
            8,
        ),
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 240, 1)

    transport.receive_active_bundle(
        ("main", "indexer"),
        _active_page_tensor(2, 7),
        SimpleNamespace(),
    )

    assert calls == [
        ("main", (0, 16), 1, -1),
        ("indexer", (160, 168), 1, -1),
    ]


def test_mte_rejects_host_upper_bound_larger_than_staging_capacity():
    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=lambda *args: pytest.fail("copy must not be launched"),
    )
    transport._anchors = {"layer": torch.empty(1)}
    transport._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([16]),
            torch.tensor([16]),
            16,
        )
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 32, 0)
    transport._peer_metadata = [
        transport._local_metadata,
        KVPPMTEPeerMetadata(8000, 32, 1),
    ]
    pages = KVPPActivePages(
        torch.tensor([2, 3, 7], dtype=torch.int32),
        torch.tensor([True, True, True]),
        count_upper_bound=3,
    )

    with pytest.raises(
        RuntimeError,
        match="upper_bound=3, capacity=2",
    ):
        transport.push_active_bundle(("layer",), pages, SimpleNamespace())


def test_prepare_batch_preserves_attention_metadata():
    block_table = torch.tensor([[7, 2, 9, 0]], dtype=torch.int32)
    slot_mapping = torch.tensor([28, 29, 8, 9], dtype=torch.int64)
    original_block_table = block_table.clone()
    original_slot_mapping = slot_mapping.clone()
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )

    context.begin_forward(block_table, torch.tensor([5]))

    assert torch.equal(block_table, original_block_table)
    assert torch.equal(slot_mapping, original_slot_mapping)


def test_batch_and_forward_lifecycle():
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    context.begin_forward(block_table, torch.tensor([5]))
    context.enter_layer("layer")
    context.wait_for_layer("layer")
    context.leave_layer("layer")
    context.finish_forward()
    context.begin_forward(block_table, torch.tensor([5]))
    context.abort_batch()


def test_begin_forward_rejects_unfinished_forward():
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)

    context.begin_forward(block_table, torch.tensor([5]))

    with pytest.raises(RuntimeError, match="phase is FORWARD_ACTIVE"):
        context.begin_forward(block_table, torch.tensor([5]))

    context.abort_batch()


def test_forward_enforces_execution_layer_order():
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer.0": 0, "layer.1": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )
    context.begin_forward(
        torch.tensor([[7, 2]], dtype=torch.int32),
        torch.tensor([5]),
    )

    with pytest.raises(RuntimeError, match="expected layer.0"):
        context.enter_layer("layer.1")

    context.abort_batch()


def test_abort_batch_clears_partial_forward():
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer.0": 0, "layer.1": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    context.begin_forward(block_table, torch.tensor([5]))
    context.enter_layer("layer.0")
    context.wait_for_layer("layer.0")
    context.leave_layer("layer.0")

    context.abort_batch()
    context.begin_forward(block_table, torch.tensor([5]))
    assert context.selected_pages is not None
    context.abort_batch()


def test_owner_layer_uses_same_scheduler_lifecycle():
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    fake_transport = SimpleNamespace(initialize=lambda caches: None)
    context = KVPPScheduler(
        group=group,
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=fake_transport,
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    context.begin_forward(block_table, torch.tensor([5]))

    context.enter_layer("layer")
    context.wait_for_layer("layer")


def test_next_layer_is_prefetched_before_current_attention():
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    context = KVPPScheduler(
        group=group,
        layer_owners={"layer.0": 0, "layer.1": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    context.begin_forward(block_table, torch.tensor([5]))

    context.enter_layer("layer.0")
    context.wait_for_layer("layer.0")
    assert context._pending_layer == "layer.1"
    context.leave_layer("layer.0")

    assert context._pending_layer == "layer.1"
    context.enter_layer("layer.1")
    context.wait_for_layer("layer.1")
    context.leave_layer("layer.1")
    assert context._pending_layer is None


def test_sfa_execution_layers_bundle_main_and_indexer_caches():
    attn_layers = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    )
    indexer_layers = (
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.indexer.k_cache",
    )
    owners = {
        attn_layers[0]: 0,
        indexer_layers[0]: 0,
        attn_layers[1]: 1,
        indexer_layers[1]: 1,
    }
    context = KVPPScheduler(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners=owners,
        num_blocks=10,
        block_size=4,
        execution_layers=attn_layers,
        transport=SimpleNamespace(),
    )

    assert context.plan.layers == attn_layers
    assert context.plan.cache_bundles == {
        attn_layers[0]: (attn_layers[0], indexer_layers[0]),
        attn_layers[1]: (attn_layers[1], indexer_layers[1]),
    }
    context.begin_forward(
        torch.tensor([[7, 2]], dtype=torch.int32), torch.tensor([5])
    )
    context.enter_layer(attn_layers[0])
    context.wait_for_layer(attn_layers[0])
    assert context._pending_layer == attn_layers[1]
    context.leave_layer(attn_layers[0])


def test_sfa_cache_bundle_order_is_stable_across_owner_dict_orders():
    attn_layer = "model.layers.0.self_attn.attn"
    indexer_layer = "model.layers.0.self_attn.indexer.k_cache"
    expected_bundle = (attn_layer, indexer_layer)

    plans = []
    for owners in (
        {attn_layer: 0, indexer_layer: 0},
        {indexer_layer: 0, attn_layer: 0},
    ):
        scheduler = KVPPScheduler(
            group=SimpleNamespace(rank_in_group=0, world_size=1),
            layer_owners=owners,
            num_blocks=10,
            block_size=4,
            execution_layers=(attn_layer,),
            transport=SimpleNamespace(),
        )
        plans.append(scheduler.plan.cache_bundles[attn_layer])

    assert plans == [expected_bundle, expected_bundle]


def test_plan_validation_rejects_cross_rank_bundle_order_mismatch():
    attn_layer = "model.layers.0.self_attn.attn"
    indexer_layer = "model.layers.0.self_attn.indexer.k_cache"
    scheduler = KVPPScheduler(
        group=SimpleNamespace(world_size=2, cpu_group=object()),
        layer_owners={attn_layer: 0, indexer_layer: 0},
        num_blocks=10,
        block_size=4,
        execution_layers=(attn_layer,),
        transport=SimpleNamespace(),
    )

    def gather_mismatched(output, local_signature, group):
        output[:] = [
            local_signature,
            ((attn_layer, 0, (indexer_layer, attn_layer)),),
        ]

    with (
        patch(
            "vllm_ascend.worker.v2.kvpp.dist.all_gather_object",
            side_effect=gather_mismatched,
        ),
        pytest.raises(RuntimeError, match="execution plans differ"),
    ):
        scheduler._validate_plan_across_ranks()


def test_sfa_cache_bundle_rejects_mismatched_owners():
    attn_layer = "model.layers.0.self_attn.attn"
    indexer_layer = "model.layers.0.self_attn.indexer.k_cache"

    with pytest.raises(ValueError, match="spans owners"):
        KVPPScheduler(
            group=SimpleNamespace(rank_in_group=0, world_size=1),
            layer_owners={attn_layer: 0, indexer_layer: 1},
            num_blocks=10,
            block_size=4,
            execution_layers=(attn_layer,),
            transport=SimpleNamespace(),
        )
