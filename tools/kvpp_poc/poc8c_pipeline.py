"""POC 8c: Barrier Bridge pipeline timing (3-4 layers, alternating owners).

This is the key difference from previous Attempt B: instead of running
ready barrier -> push -> done barrier -> receive all at once inside one
function, we split each KV transfer into two stages separated by model
computation, achieving real overlap:

    PUSH(i+1) || Attention + o_proj + MLP of i
    RECEIVE(i+1) || Q/KV projection of i+1

Layer/owner/scratch layout (per doc section 19):
    L0 owner rank0 -> slot0
    L1 owner rank1 -> slot1
    L2 owner rank0 -> slot0
    L3 owner rank1 -> slot1

Each layer gets a distinct KV pattern (L0=100, L1=200, L2=300, L3=400).
Dummy matmul simulates projection/attention/MLP compute so profiler can
see overlap.
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


# Simulate 4 layers with alternating owners and 2 scratch slots.
NUM_LAYERS = 4
LAYER_OWNERS = {i: i % 2 for i in range(NUM_LAYERS)}  # L0,L2 -> rank0; L1,L3 -> rank1
LAYER_PATTERNS = {0: 100.0, 1: 200.0, 2: 300.0, 3: 400.0}
SLOT_OF_LAYER = {0: 0, 1: 1, 2: 0, 3: 1}  # two scratch slots


def dummy_compute(label: str, size: int = 2048, iters: int = 8):
    """Simulate attention/MLP compute heavy enough to see overlap in profiler."""
    a = torch.randn(size, size, device="npu", dtype=torch.float16)
    b = torch.randn(size, size, device="npu", dtype=torch.float16)
    for _ in range(iters):
        a = torch.matmul(a, b)
    return a


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = torch.npu.current_device()
    print(f"[POC8C][rank={rank}] device={device}", flush=True)

    group = MockGroup(
        rank_in_group=rank,
        world_size=world_size,
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
    )

    # Build KV caches: one per layer. Owner has real KV; non-owner has scratch.
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
        group=group,
        layer_owners=layer_owners_str,
        num_blocks=num_blocks,
    )
    transport.initialize(kv_caches)
    print(f"[POC8C][rank={rank}] transport initialized", flush=True)

    # Active pages: 1 page (page_id=0) per layer
    page_ids = torch.tensor([0], dtype=torch.int64, device="npu")
    valid_mask = torch.tensor([True], dtype=torch.bool, device="npu")
    pages = KVPPActivePages(page_ids, valid_mask, count_upper_bound=1)

    # Events: 4 events per layer (push_start, push_done, recv_start, recv_done)
    @dataclass
    class LayerEvents:
        push_start: object
        push_done: object
        recv_start: object
        recv_done: object

    layer_events = {
        i: LayerEvents(
            torch.npu.Event(), torch.npu.Event(),
            torch.npu.Event(), torch.npu.Event(),
        )
        for i in range(NUM_LAYERS)
    }

    capture_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    sync_token = torch.zeros(1, dtype=torch.int32, device="npu")
    retained = []

    torch.npu.synchronize()
    dist.barrier()

    # Build bundles
    bundles = {i: (layer_names[i],) for i in range(NUM_LAYERS)}

    # === Graph capture implementing the pipeline DAG from doc section 16 ===
    #
    # For Layer 0 (first layer, no previous recv to wait):
    #   READY barrier(0) -> push_start(0) -> PUSH(0) on comm -> push_done(0)
    #   -> wait push_done(0) -> DONE barrier(0) -> recv_start(0) -> RECV(0) on comm
    #   -> recv_done(0)
    #
    # For Layer i (i>0):
    #   (Layer i-1's recv_done(i-1) already waited in previous wait_for_layer)
    #   READY barrier(i) -> push_start(i) -> PUSH(i) || dummy_compute(attn_i-1)
    #   -> wait push_done(i) -> DONE barrier(i) -> recv_start(i) -> RECV(i)
    #      || dummy_compute(qkv_i)
    #   -> recv_done(i)
    #
    # To match doc's enter_layer/wait_for_layer split, we structure capture as:
    #   for each layer i:
    #     [start_push(i)]      = READY barrier + push_start + PUSH on comm
    #     [finish_push_start_receive(i)] = wait push_done + DONE barrier + recv_start + RECV on comm
    #     dummy_compute("qkv_i")   # overlaps with RECV(i)
    #     [wait_receive(i)]    = wait recv_done
    #     dummy_compute("attn_i")  # next layer's PUSH will overlap this
    #
    # For the LAST layer, no next push to start.

    print(f"[POC8C][rank={rank}] starting graph capture", flush=True)
    graph = torch.npu.NPUGraph()

    with torch.npu.graph(graph, stream=capture_stream):
        for i in range(NUM_LAYERS):
            ev = layer_events[i]
            bundle = bundles[i]
            owner = LAYER_OWNERS[i]

            # --- _graph_start_push(i) ---
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.push_start.record(capture_stream)
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.push_start)
                comp = transport.push_active_bundle(bundle, pages, comm_stream)
                ev.push_done.record(comm_stream)
            retained.append(comp)

            # --- _graph_finish_push_start_receive(i) ---
            capture_stream.wait_event(ev.push_done)
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.recv_start.record(capture_stream)
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.recv_start)
                comp2 = transport.receive_active_bundle(bundle, pages, comm_stream)
                ev.recv_done.record(comm_stream)
            retained.append(comp2)

            # --- Q/KV projection of layer i overlaps with RECV(i) on comm ---
            dummy_compute(f"qkv_{i}")

            # --- wait_for_layer(i) ---
            capture_stream.wait_event(ev.recv_done)

            # --- Attention + o_proj + MLP of layer i ---
            # Next iteration's PUSH(i+1) will overlap this.
            dummy_compute(f"attn_{i}")

    print(f"[POC8C][rank={rank}] capture OK", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # === Replay with distinct per-layer patterns ===
    REPLAY_COUNT = 50  # 50 replays * 4 layers = 200 transfers
    for i in range(REPLAY_COUNT):
        # Owner fills each layer with its pattern
        for li in range(NUM_LAYERS):
            if rank == LAYER_OWNERS[li]:
                kv_caches[layer_names[li]].fill_(LAYER_PATTERNS[li])
        torch.npu.synchronize()
        dist.barrier()

        graph.replay()
        torch.npu.synchronize()
        dist.barrier()

        # Consumer verifies each layer's received data
        for li in range(NUM_LAYERS):
            if rank != LAYER_OWNERS[li]:
                expected = LAYER_PATTERNS[li]
                actual = kv_caches[layer_names[li]][:block_size, 0, 0][0].item()
                if not (abs(actual - expected) < 0.5):
                    print(
                        f"[POC8C][rank={rank}] FAIL replay {i} layer {li}: "
                        f"expected={expected}, actual={actual}",
                        flush=True,
                    )
                    sys.exit(1)
        if rank == 0 and i % 10 == 0:
            print(f"[POC8C] replay {i} OK", flush=True)

    print(f"[POC8C][rank={rank}] PASS ({REPLAY_COUNT} replays, {NUM_LAYERS} layers)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
