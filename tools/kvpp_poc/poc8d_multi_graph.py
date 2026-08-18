"""POC 8d: Multi graph-size alternating replay.

FULL_DECODE_ONLY captures graphs for different request capacities. This POC
verifies that two graphs (capacity 1 and capacity 2) can alternate replay
without conflict, each using its own event bundle.

Per doc section 22: each graph capacity gets independent events, not shared.
"""
import os
import sys

import torch
import torch_npu
import torch.distributed as dist

sys.path.insert(0, "/home/recky/vllm-ascend")
sys.path.insert(0, "/home/recky/vllm")

from dataclasses import dataclass
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    MemFabricMTEKVPPTransport,
)


@dataclass
class MockGroup:
    rank_in_group: int
    world_size: int
    cpu_group: object
    device_group: object


NUM_LAYERS = 4
LAYER_OWNERS = {i: i % 2 for i in range(NUM_LAYERS)}


def dummy_compute(size: int = 2048, iters: int = 8):
    a = torch.randn(size, size, device="npu", dtype=torch.float16)
    b = torch.randn(size, size, device="npu", dtype=torch.float16)
    for _ in range(iters):
        a = torch.matmul(a, b)
    return a


@dataclass
class GraphResources:
    """Per-graph-capacity event bundle (doc section 22)."""
    layer_events: dict
    retained: list


def make_resources(num_layers):
    @dataclass
    class LE:
        push_start: object
        push_done: object
        recv_start: object
        recv_done: object
    return GraphResources(
        layer_events={
            i: LE(*[torch.npu.Event() for _ in range(4)])
            for i in range(num_layers)
        },
        retained=[],
    )


def capture_graph(graph, capture_stream, comm_stream, sync_token,
                  transport, bundles, pages, resources, num_layers):
    retained = []
    with torch.npu.graph(graph, stream=capture_stream):
        for i in range(num_layers):
            ev = resources.layer_events[i]
            bundle = bundles[i]
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.push_start.record(capture_stream)
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.push_start)
                comp = transport.push_active_bundle(bundle, pages, comm_stream)
                ev.push_done.record(comm_stream)
            retained.append(comp)
            capture_stream.wait_event(ev.push_done)
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.recv_start.record(capture_stream)
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.recv_start)
                comp2 = transport.receive_active_bundle(bundle, pages, comm_stream)
                ev.recv_done.record(comm_stream)
            retained.append(comp2)
            dummy_compute()
            capture_stream.wait_event(ev.recv_done)
            dummy_compute()
    resources.retained.extend(retained)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    group = MockGroup(rank_in_group=rank, world_size=world_size,
                      cpu_group=dist.group.WORLD, device_group=dist.group.WORLD)

    num_blocks = 4
    block_size = 128
    num_kv_heads = 8
    head_dim = 128
    kv_shape = (num_blocks * block_size, num_kv_heads, head_dim)
    layer_names = [f"layer_{i}" for i in range(NUM_LAYERS)]
    kv_caches = {name: torch.zeros(kv_shape, dtype=torch.float16, device="npu")
                 for name in layer_names}
    layer_owners_str = {name: LAYER_OWNERS[i] for i, name in enumerate(layer_names)}

    transport = MemFabricMTEKVPPTransport(
        group=group, layer_owners=layer_owners_str, num_blocks=num_blocks)
    transport.initialize(kv_caches)
    print(f"[POC8D][rank={rank}] transport initialized", flush=True)

    # Two graphs with different "capacity" — we simulate capacity by using
    # different page sets. Graph A: 1 page, Graph B: 2 pages.
    pages_a = KVPPActivePages(
        torch.tensor([0], dtype=torch.int64, device="npu"),
        torch.tensor([True], dtype=torch.bool, device="npu"),
        count_upper_bound=1,
    )
    pages_b = KVPPActivePages(
        torch.tensor([0, 1], dtype=torch.int64, device="npu"),
        torch.tensor([True, True], dtype=torch.bool, device="npu"),
        count_upper_bound=2,
    )

    bundles = {i: (layer_names[i],) for i in range(NUM_LAYERS)}
    capture_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    sync_token = torch.zeros(1, dtype=torch.int32, device="npu")

    # Independent resources per graph capacity (doc section 22)
    res_a = make_resources(NUM_LAYERS)
    res_b = make_resources(NUM_LAYERS)

    torch.npu.synchronize()
    dist.barrier()

    # Capture graph A (capacity 1)
    graph_a = torch.npu.NPUGraph()
    print(f"[POC8D][rank={rank}] capturing graph A (cap=1)", flush=True)
    capture_graph(graph_a, capture_stream, comm_stream, sync_token,
                  transport, bundles, pages_a, res_a, NUM_LAYERS)
    print(f"[POC8D][rank={rank}] graph A captured", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # Capture graph B (capacity 2) — needs fresh sync_token state
    sync_token.zero_()
    graph_b = torch.npu.NPUGraph()
    print(f"[POC8D][rank={rank}] capturing graph B (cap=2)", flush=True)
    capture_graph(graph_b, capture_stream, comm_stream, sync_token,
                  transport, bundles, pages_b, res_b, NUM_LAYERS)
    print(f"[POC8D][rank={rank}] graph B captured", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # Alternate replay: G1, G2, G1, G2, G2, G1, ... at least 100 times
    REPLAY_SEQ = [graph_a, graph_b, graph_a, graph_b, graph_b, graph_a] * 17  # 102
    patterns_a = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}
    patterns_b = {0: 100.0, 1: 200.0, 2: 300.0, 3: 400.0}

    print(f"[POC8D][rank={rank}] starting {len(REPLAY_SEQ)} alternating replays", flush=True)
    for idx, g in enumerate(REPLAY_SEQ):
        is_a = (g is graph_a)
        pats = patterns_a if is_a else patterns_b
        pages = pages_a if is_a else pages_b
        # Owner fills
        for li in range(NUM_LAYERS):
            if rank == LAYER_OWNERS[li]:
                kv_caches[layer_names[li]].fill_(pats[li])
        torch.npu.synchronize()
        dist.barrier()
        g.replay()
        torch.npu.synchronize()
        dist.barrier()
        # Consumer verifies page 0
        for li in range(NUM_LAYERS):
            if rank != LAYER_OWNERS[li]:
                expected = pats[li]
                actual = kv_caches[layer_names[li]][:block_size, 0, 0][0].item()
                if not (abs(actual - expected) < 0.5):
                    print(f"[POC8D][rank={rank}] FAIL idx={idx} graph={'A' if is_a else 'B'} layer={li}: expected={expected} actual={actual}", flush=True)
                    sys.exit(1)
        if rank == 0 and idx % 20 == 0:
            print(f"[POC8D] replay {idx} ({'A' if is_a else 'B'}) OK", flush=True)

    print(f"[POC8D][rank={rank}] PASS ({len(REPLAY_SEQ)} alternating replays)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
