from types import SimpleNamespace

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    KVPPMTEPeerMetadata,
    MemFabricMTEKVPPTransport,
    _MTEDeviceBufferMetadata,
)


def _pages() -> KVPPActivePages:
    return KVPPActivePages(
        torch.tensor([2, 2, 7, 10], dtype=torch.int32),
        torch.tensor([True, False, True, False]),
        count_upper_bound=2,
    )


def _transport(rank: int, owners=None, copy_op=None):
    return MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=rank, world_size=2),
        owners or {"main": 0, "indexer": 0},
        10,
        copy_op=copy_op or (lambda *args: None),
    )


def _install_layers(transport):
    transport._anchors = {"main": torch.empty(1), "indexer": torch.empty(1)}
    transport._device_layers = {
        "main": _MTEDeviceBufferMetadata(
            local_base_offsets=torch.tensor([0, 4000]),
            block_strides=torch.tensor([32, 64]),
            block_bytes=torch.tensor([16, 8]),
            staging_bytes_per_slot=24,
        ),
        "indexer": _MTEDeviceBufferMetadata(
            local_base_offsets=torch.tensor([8000]),
            block_strides=torch.tensor([16]),
            block_bytes=torch.tensor([8]),
            staging_bytes_per_slot=8,
        ),
    }


def test_mte_01_local_descriptors_have_offsets_and_zero_invalid_lengths():
    transport = _transport(0)
    _install_layers(transport)

    offsets, lengths = transport._local_descriptors("main", _pages())

    assert offsets.tolist() == [64, 64, 224, 320, 4128, 4128, 4448, 4640]
    assert lengths.tolist() == [16, 0, 16, 0, 8, 0, 8, 0]


def test_mte_02_bundle_staging_offsets_are_disjoint_and_use_active_ordinal():
    transport = _transport(0)
    _install_layers(transport)

    offsets = transport._bundle_staging_offsets(("main", "indexer"), _pages(), staging_bytes=320)

    # Ten active-page slots: main's two buffers occupy [0, 240), while the
    # indexer occupies [240, 320). Masked descriptors retain the preceding
    # ordinal but carry zero lengths in MTE-01.
    assert offsets["main"].tolist() == [0, 0, 16, 16, 160, 160, 168, 168]
    assert offsets["indexer"].tolist() == [240, 240, 248, 248]
    assert max(offsets["main"].tolist()) < min(offsets["indexer"].tolist())


def test_mte_03_owner_pushes_to_peer_and_consumer_receives_locally(monkeypatch):
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
        calls.append((source_rank, destination_rank, staging_base))

    owner = _transport(0, copy_op=copy_op)
    _install_layers(owner)
    owner._local_metadata = KVPPMTEPeerMetadata(8000, 320, 0)
    owner._peer_metadata = [
        owner._local_metadata,
        KVPPMTEPeerMetadata(9000, 320, 1),
    ]

    owner.push_active_bundle(("main", "indexer"), _pages(), SimpleNamespace())
    assert calls == [(-1, 1, 9000), (-1, 1, 9000)]

    calls.clear()
    consumer = _transport(1, copy_op=copy_op)
    _install_layers(consumer)
    consumer._local_metadata = KVPPMTEPeerMetadata(9000, 320, 1)

    consumer.receive_active_bundle(("main", "indexer"), _pages(), SimpleNamespace())
    assert calls == [(1, -1, 9000), (1, -1, 9000)]
