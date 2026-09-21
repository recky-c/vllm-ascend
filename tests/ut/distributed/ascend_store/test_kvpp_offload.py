"""Regression coverage for the two-stage P-side prefetch pipeline."""

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def make_worker():
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kvpp_offload = True
    worker.use_layerwise = True
    worker.num_layers = 6
    worker.current_layer = 0
    worker.next_layer_to_submit = 0
    worker.prefetch_layer_map = {4: 2, 5: 3}
    worker.layer_load_tasks = [[object()], [object()], [], [], [], []]
    worker.layer_save_tasks = [[] for _ in range(6)]
    worker.layer_load_finished_events = [threading.Event() for _ in range(6)]
    worker.kv_recv_thread = MagicMock()
    worker.kv_send_thread = MagicMock()
    worker._kvpp_offload_cancelled = threading.Event()
    return worker


def test_startup_primes_first_two_layers_without_attention_gate():
    worker = make_worker()
    worker.process_layer_data = MagicMock()
    worker.start_load_kv(SimpleNamespace(requests=[object()]))
    queued = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [task.layer_id for task in queued] == [0, 1]
    assert all(task.attention_start_gate is None for task in queued)


def test_empty_metadata_still_primes_reuse_lifetimes():
    worker = make_worker()
    worker.process_layer_data = MagicMock()
    worker.start_load_kv(SimpleNamespace(requests=[]))
    assert worker.next_layer_to_submit == 2


def test_h2d_advances_by_global_layer_not_number_of_owner_tasks():
    worker = make_worker()
    worker._submit_kvpp_layer_loads(1, gate=None)
    worker.layer_load_finished_events[0].set()
    worker.wait_for_layer_load()
    queued = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [task.layer_id for task in queued] == [0, 1, 2]
    assert queued[-1].attention_start_gate is not None
    assert worker.layer_load_finished_events[0].is_set()
    # Do not wait for future H2D on the compute thread.
    assert not worker.layer_load_finished_events[2].is_set()
    worker.current_layer = 1
    worker.layer_load_finished_events[1].set()
    worker.wait_for_layer_load()
    assert worker.next_layer_to_submit == 4


def test_peer_without_h2d_still_waits_for_previous_buffer_save():
    worker = make_worker()
    worker.next_layer_to_submit = 4
    worker._submit_kvpp_layer_loads(4, gate=None)
    task = worker.kv_recv_thread.add_request.call_args.args[0]
    assert task.transfer_tasks == []
    assert task.wait_for_save_layer == 2


def test_kvpp_and_attention_can_both_observe_load_completion():
    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    worker.layer_load_finished_events[0].set()
    worker.wait_for_layer_load()
    worker.wait_for_kvpp_cache("layer.0")
    assert worker.layer_load_finished_events[0].is_set()


def test_completed_event_does_not_hide_transfer_failure():
    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    worker.layer_load_finished_events[0].set()
    worker.kv_recv_thread.raise_if_failed.side_effect = RuntimeError("H2D failed")
    with pytest.raises(RuntimeError, match="H2D failed"):
        worker.wait_for_kvpp_cache("layer.0")


def test_only_owner_keeps_h2d_tasks_but_single_writer_saves_are_preserved():
    worker = make_worker()
    worker.kvpp_layer_owners = {i: i // 3 for i in range(6)}
    worker.kvpp_rank = 1
    worker.physical_layer_to_group_layers = {}
    worker._process_save_for_layer_batch = lambda requests, layer, *args: worker.layer_save_tasks[layer].append("save")
    worker._process_load_for_layer_batch = lambda requests, layer, *args: worker.layer_load_tasks[layer].append("load")
    worker._prepare_load_gvas = MagicMock()
    worker._alloc_gvas_for_save = MagicMock()
    worker._build_shared_save_data = MagicMock()
    worker._build_shared_load_data = MagicMock()
    worker.process_layer_data([object()])
    assert worker.layer_load_tasks == [[], [], [], ["load"], ["load"], ["load"]]
    assert worker.layer_save_tasks == [["save"] for _ in range(6)]
    assert worker.kv_recv_thread.final_layer_id == -1


def test_end_of_forward_releases_leases_but_only_completes_last_chunks():
    worker = make_worker()
    worker.current_layer = worker.num_layers - 1
    worker.sync_save_events = [MagicMock() for _ in range(worker.num_layers)]
    worker.layer_save_finished_events = [threading.Event() for _ in range(worker.num_layers)]
    worker.kv_send_thread = MagicMock()
    worker.m_store = MagicMock()
    requests = [
        SimpleNamespace(
            req_id="partial", is_last_chunk=False, load_keys=["shared"], load_spec=SimpleNamespace(can_load=True)
        ),
        SimpleNamespace(
            req_id="done", is_last_chunk=True, load_keys=["shared", "other"], load_spec=SimpleNamespace(can_load=True)
        ),
        SimpleNamespace(req_id="no_load", is_last_chunk=True, load_keys=None, load_spec=None),
    ]
    worker.save_kv_layer(SimpleNamespace(requests=requests))
    worker.m_store.batch_remove_lease.assert_called_once_with(["shared", "other"])
    worker.kv_recv_thread.set_finished_request.assert_called_once_with("done")
    assert worker.current_layer == worker.num_layers


@pytest.mark.parametrize("ready", [False, True])
def test_d2h_failure_is_visible_to_kvpp_even_if_load_completed(ready):
    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    if ready:
        worker.layer_load_finished_events[0].set()
    worker.kv_send_thread.raise_if_failed.side_effect = RuntimeError("D2H failed")
    with pytest.raises(RuntimeError, match="D2H failed"):
        worker.wait_for_kvpp_cache("layer.0")


def test_cancel_interrupts_pending_load_without_publishing_success():
    from concurrent.futures import ThreadPoolExecutor

    worker = make_worker()
    worker._extract_physical_layer_index = lambda _: 0
    started = threading.Event()

    def wait():
        started.set()
        worker.wait_for_kvpp_cache("layer.0")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wait)
        assert started.wait(timeout=2)
        worker.abort_kvpp_offload()
        with pytest.raises(RuntimeError, match="aborted"):
            future.result(timeout=2)
    assert not worker.layer_load_finished_events[0].is_set()
    with pytest.raises(RuntimeError, match="restart the worker"):
        worker.start_load_kv(SimpleNamespace(requests=[]))


def test_next_forward_resets_all_load_events_before_priming():
    worker = make_worker()
    worker.process_layer_data = MagicMock()
    for _ in range(2):
        for event in worker.layer_load_finished_events:
            event.set()
        worker.start_load_kv(SimpleNamespace(requests=[]))
        assert not any(event.is_set() for event in worker.layer_load_finished_events)
        assert worker.next_layer_to_submit == 2


@pytest.mark.parametrize("rank", [0, 1])
def test_two_forwards_cross_owner_boundary_and_reuse_buffers(rank):
    import numpy as np

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerRecvingThread
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LayerBatchReqMeta, LayerTransferTask

    worker = make_worker()
    worker.num_layers = 18
    worker.layer_load_finished_events = [threading.Event() for _ in range(18)]
    worker.layer_save_finished_events = [threading.Event() for _ in range(18)]
    worker.sync_save_events = [MagicMock() for _ in range(18)]
    worker._extract_physical_layer_index = int
    worker.prefetch_layer_map = {}
    for owner in (0, 1):
        width = 3 if owner == rank else 2
        for layer in range(owner * 9 + width, owner * 9 + 9):
            worker.prefetch_layer_map[layer] = layer - width

    def prepare(_):
        worker.layer_load_tasks = [
            [LayerTransferTask(layer_id=layer, block_ranges=[])] if layer // 9 == rank else [] for layer in range(18)
        ]

    worker.process_layer_data = prepare
    receiver = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    receiver.layer_load_finished_events = worker.layer_load_finished_events
    receiver.layer_save_finished_events = worker.layer_save_finished_events
    receiver.sync_save_events = worker.sync_save_events
    receiver.check_dependencies = worker._raise_if_kvpp_offload_failed
    receiver.request_queue = Queue()
    receiver.get_event = threading.Event()
    receiver.final_layer_id = -1
    receiver.max_transfer_blocks = receiver.max_transfer_bytes = 0
    receiver._stagger_h2d_submit = MagicMock()
    copies = []
    receiver._batch_copy_with_limits = lambda *args: copies.append(1) or 0
    builder = MagicMock()
    builder.build.side_effect = lambda task, **kw: LayerBatchReqMeta(
        req_ids=["r"],
        layer_id=task.layer_id,
        is_last_chunks=[False],
        addr_array=np.asarray([10]),
        size_array=np.asarray([16]),
        gvas_array=np.asarray([100]),
    )
    receiver.group_builders = [builder]
    gates = []

    def new_gate():
        gate = threading.Event()
        gates.append(gate)
        return gate

    futures = []
    with ThreadPoolExecutor(max_workers=1) as copies_executor, ThreadPoolExecutor(max_workers=1) as compute_executor:

        def submit(task):
            receiver.request_queue.put(task)
            futures.append(copies_executor.submit(receiver._handle_request, task))

        worker.kv_recv_thread.add_request.side_effect = submit

        def propagate_copy_error():
            for future in futures:
                if future.done():
                    future.result()

        worker.kv_recv_thread.raise_if_failed.side_effect = propagate_copy_error

        def forward():
            metadata = SimpleNamespace(requests=[])
            for _ in range(2):
                worker.start_load_kv(metadata)
                for layer in range(18):
                    worker.wait_for_layer_load()
                    worker.wait_for_kvpp_cache(str(layer))
                    gates[-1].set()
                    worker.save_kv_layer(metadata)
                assert all(event.is_set() for event in worker.layer_load_finished_events)
                assert not any(event.is_set() for event in worker.layer_save_finished_events)

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.reset_attention_compute_start_gate",
            side_effect=new_gate,
        ):
            future = compute_executor.submit(forward)
            try:
                future.result(timeout=10)
            finally:
                worker.abort_kvpp_offload()
        for future in futures:
            future.result(timeout=2)
    assert len(futures) == 36
    assert len(copies) == 18
    for event in worker.sync_save_events:
        assert event.record.call_count == 2
