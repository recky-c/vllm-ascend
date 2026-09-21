# SPDX-License-Identifier: Apache-2.0
"""Two-NPU data-path smoke; the host store is in memory, not a Memcache service.

Exercises real H2D/D2H, NPU fences, KVPP MTE and the layerwise worker together.
Requires a native KVPP build and MemFabric Hybrid 1.2. Run with two idle NPUs
selected by ASCEND_RT_VISIBLE_DEVICES and MEMFABRIC_HYBRID_HOME_PATH set.
"""

import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LayerBatchReqMeta, LayerTransferTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import MemFabricMTEKVPPTransport
from vllm_ascend.v1.core.kv_cache_placement import _get_allocation_groups
from vllm_ascend.worker.v2.kvpp import KVPPScheduler

NUM_LAYERS = 36
NUM_BLOCKS = 4
BLOCK_SIZE = 16


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_rank(rank, control_port, store_port):
    torch.npu.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{control_port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    os.environ["MF_CONFIG_STORE_URL"] = f"tcp://127.0.0.1:{store_port}"
    os.environ["ASCEND_KVPP_MTE_STAGING_BYTES"] = str(2 << 20)
    os.environ["ASCEND_KVPP_MTE_TIMEOUT_SECONDS"] = "60"
    names = [f"model.layers.{i}.attn" for i in range(NUM_LAYERS)]
    specs = {}
    for i, name in enumerate(names):
        specs[name] = MLAAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=32, dtype=torch.float16)
        if i % 2 == 0:
            specs[name + ".indexer.k_cache"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
            )
    layer_index = lambda name: int(name.split(".")[2])
    owners = {name: layer_index(name) // (NUM_LAYERS // 2) for name in specs}
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=BLOCK_SIZE, kv_cache_specs=specs))
    _, aliases = _get_allocation_groups([group], specs, owners, rank, 3)
    caches, predecessors = {}, {}
    for representative, members in aliases.items():
        tensor = torch.empty(
            (NUM_BLOCKS, BLOCK_SIZE, specs[representative].head_size), dtype=torch.float16, device="npu"
        )
        for name in members:
            caches[name] = tensor
        for before, after in zip(members, members[1:]):
            assert predecessors.setdefault(layer_index(after), layer_index(before)) == layer_index(before)
    host = {
        name: torch.full(
            tuple(tensor.shape), layer_index(name) * 4 + int(name.endswith(".k_cache")), dtype=torch.float16
        )
        for name, tensor in caches.items()
    }
    reference = {name: tensor.clone() for name, tensor in host.items()}
    component_names = list(caches)
    component_ids = {name: i for i, name in enumerate(component_names)}
    bundles = {i: [name for name in specs if layer_index(name) == i] for i in range(NUM_LAYERS)}
    copy_stream = torch.npu.Stream()
    copies = []
    futures = []
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kvpp_offload = worker.use_layerwise = True
    worker.num_layers = NUM_LAYERS
    worker.prefetch_layer_map = predecessors
    worker.layer_load_finished_events = [threading.Event() for _ in names]
    worker.layer_save_finished_events = [threading.Event() for _ in names]
    worker.sync_save_events = [torch.npu.Event() for _ in names]
    worker._kvpp_offload_cancelled = threading.Event()
    worker._extract_physical_layer_index = layer_index
    worker.kv_send_thread = SimpleNamespace(raise_if_failed=lambda: None)

    def prepare(_):
        worker.layer_load_tasks = [
            [LayerTransferTask(layer_id=i, block_ranges=[])] if owners[name] == rank else []
            for i, name in enumerate(names)
        ]

    worker.process_layer_data = prepare

    def build(task, **kwargs):
        ids = np.asarray([component_ids[name] for name in bundles[task.layer_id]], dtype=np.int64)
        return LayerBatchReqMeta(
            req_ids=["smoke"],
            layer_id=task.layer_id,
            is_last_chunks=[False],
            addr_array=ids,
            gvas_array=ids,
            size_array=np.ones_like(ids),
        )

    receiver = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    receiver.layer_load_finished_events = worker.layer_load_finished_events
    receiver.layer_save_finished_events = worker.layer_save_finished_events
    receiver.sync_save_events = worker.sync_save_events
    receiver.request_queue = Queue()
    receiver.get_event = threading.Event()
    receiver.final_layer_id = -1
    receiver.max_transfer_blocks = receiver.max_transfer_bytes = 0
    receiver.group_builders = [SimpleNamespace(build=build)]
    receiver.check_dependencies = worker._raise_if_kvpp_offload_failed
    receiver._stagger_h2d_submit = lambda _: None

    def copy_from_host(gvas, addresses, sizes, direction, *limits):
        with torch.npu.stream(copy_stream):
            for index in gvas:
                name = component_names[int(index)]
                assert owners[name] == rank
                caches[name].copy_(host[name])
                copies.append(name)
        copy_stream.synchronize()
        return 0

    receiver._batch_copy_with_limits = copy_from_host

    def check_errors():
        for future in futures:
            if future.done():
                future.result()

    with ThreadPoolExecutor(max_workers=1) as executor:

        def receive(task):
            torch.npu.set_device(rank)
            receiver._handle_request(task)

        def submit(task):
            receiver.request_queue.put(task)
            futures.append(executor.submit(receive, task))

        worker.kv_recv_thread = SimpleNamespace(add_request=submit, raise_if_failed=check_errors)
        process_group = SimpleNamespace(world_size=2, rank_in_group=rank, ranks=[0, 1], cpu_group=dist.group.WORLD)
        transport = MemFabricMTEKVPPTransport(process_group, owners, NUM_BLOCKS)
        scheduler = KVPPScheduler(
            process_group,
            owners,
            NUM_BLOCKS,
            BLOCK_SIZE,
            transport=transport,
            execution_layers=tuple(names),
            wait_for_cache=worker.wait_for_kvpp_cache,
            abort_cache=worker.abort_kvpp_offload,
        )
        scheduler.initialize_transport(caches)
        table = torch.tensor([[2, 0, 1], [2, 0, 1]], dtype=torch.int32, device="npu")
        metadata = SimpleNamespace(requests=[])
        try:
            for round_id, length in enumerate((17, 33, 34)):
                active_ids = [2, 0] if length == 17 else [2, 0, 1]
                for tensor in caches.values():
                    tensor.fill_(-777)
                torch.npu.synchronize()
                scheduler.begin_forward(table, [length, 1])
                worker.start_load_kv(metadata)
                for i, name in enumerate(names):
                    scheduler.enter_layer(name)
                    worker.wait_for_layer_load()
                    scheduler.wait_for_layer(name)
                    record_attention_compute_start()
                    for component in bundles[i]:
                        actual = caches[component].cpu()
                        torch.testing.assert_close(actual[active_ids], reference[component][active_ids], rtol=0, atol=0)
                        page = [2, 0, 1][(length - 1) // BLOCK_SIZE]
                        offset = (length - 1) % BLOCK_SIZE
                        value = 500 + round_id * NUM_LAYERS + i
                        caches[component][page, offset].fill_(value)
                        reference[component][page, offset].fill_(value)
                        if rank == 0:
                            # Rank 0 saves every layer, including owner-rank-1 layers.
                            host[component][active_ids] = caches[component].cpu()[active_ids]
                    scheduler.leave_layer(name)
                    worker.save_kv_layer(metadata)
                scheduler.finish_forward()
                shared = [host if rank == 0 else None]
                dist.broadcast_object_list(shared, src=0)
                host = shared[0]
                for component in host:
                    torch.testing.assert_close(host[component], reference[component], rtol=0, atol=0)
                assert not any(event.is_set() for event in worker.layer_save_finished_events)
            assert len(futures) == NUM_LAYERS * 3
            assert len(copies) == sum(owner == rank for owner in owners.values()) * 3
            for future in futures:
                future.result(timeout=10)
            dist.barrier()
            print(
                f"rank={rank}: 36 layers, sparse indexer, 3 forwards, exact KV and single-writer round trips passed",
                flush=True,
            )
        finally:
            scheduler.close()
    dist.destroy_process_group()


def test_layerwise_kvpp_real_mte_round_trip():
    pytest.importorskip("memfabric_hybrid")
    if not os.getenv("MEMFABRIC_HYBRID_HOME_PATH"):
        pytest.skip("MemFabric runtime environment is required")
    if torch.npu.device_count() < 2:
        pytest.skip("Two idle NPUs must be selected")
    mp.spawn(_run_rank, args=(_free_port(), _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    test_layerwise_kvpp_real_mte_round_trip()
