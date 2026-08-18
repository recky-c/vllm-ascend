"""POC 8b: Barrier Bridge + real MTE transport.

Uses real MemFabricMTEKVPPTransport with 2 ranks. Owner pushes KV via MTE on
comm_stream; non-owner receives on comm_stream. Barriers stay on capture
stream. Events bridge the two streams.

DAG (per layer):
    Capture Stream C                 Comm Stream M

    READY all_reduce
          |
    record push_start
          | ----------------------->
                                    wait push_start
                                    push_active_bundle (owner only)
                                    record push_done
          |<------------------------
    wait push_done
    DONE all_reduce
    record recv_start
          | ----------------------->
                                    wait recv_start
                                    receive_active_bundle (non-owner only)
                                    record recv_done
          |<------------------------
    wait recv_done
    validate

Each replay, owner fills KV with a distinct pattern; consumer verifies.
Then swap owner/consumer and repeat.
"""
import os
import sys

import torch
import torch_npu
import torch.distributed as dist

# Make vllm_ascend importable
sys.path.insert(0, "/home/recky/vllm-ascend")
sys.path.insert(0, "/home/recky/vllm")

from dataclasses import dataclass
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    KVPPBufferMetadata,
    MemFabricMTEKVPPTransport,
    build_kvpp_layer_metadata,
    flatten_kvpp_cache,
)


@dataclass
class MockGroup:
    """Minimal GroupCoordinator substitute for POC."""
    rank_in_group: int
    world_size: int
    cpu_group: object
    device_group: object


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"POC 8b requires world_size==2, got {world_size}"
    device = torch.npu.current_device()
    print(f"[POC8B][rank={rank}] device={device}", flush=True)

    group = MockGroup(
        rank_in_group=rank,
        world_size=world_size,
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
    )

    # --- Build fake KV caches ---
    num_blocks = 4
    block_size = 128
    num_kv_heads = 8
    head_dim = 128
    # KV cache shape: (num_blocks * block_size, num_kv_heads, head_dim)
    kv_shape = (num_blocks * block_size, num_kv_heads, head_dim)
    layer_name = "model.layers.0.self_attn.attn"
    # Owner (rank 0) has the real KV; non-owner has scratch that will be filled
    owner_kv = torch.zeros(kv_shape, dtype=torch.float16, device="npu")
    kv_caches = {layer_name: owner_kv}

    layer_owners = {layer_name: int(os.environ.get("POC8B_OWNER_RANK", "0"))}
    owner_rank = layer_owners[layer_name]
    print(f"[POC8B][rank={rank}] owner_rank={owner_rank}", flush=True)

    # --- Initialize transport ---
    transport = MemFabricMTEKVPPTransport(
        group=group,
        layer_owners=layer_owners,
        num_blocks=num_blocks,
    )
    transport.initialize(kv_caches)
    print(f"[POC8B][rank={rank}] transport initialized", flush=True)

    # --- Build active pages (1 active page, page_id=0) ---
    page_ids = torch.tensor([0], dtype=torch.int64, device="npu")
    valid_mask = torch.tensor([True], dtype=torch.bool, device="npu")
    pages = KVPPActivePages(
        page_ids=page_ids,
        valid_mask=valid_mask,
        count_upper_bound=1,
    )

    # --- Events ---
    capture_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    push_start = torch.npu.Event()
    push_done = torch.npu.Event()
    recv_start = torch.npu.Event()
    recv_done = torch.npu.Event()

    sync_token = torch.zeros(1, dtype=torch.int32, device="npu")

    bundle = (layer_name,)

    torch.npu.synchronize()
    dist.barrier()

    # --- Graph capture ---
    graph = torch.npu.NPUGraph()
    retained = []

    print(f"[POC8B][rank={rank}] starting graph capture (owner=rank0)", flush=True)

    with torch.npu.graph(graph, stream=capture_stream):
        # phase 1: READY barrier on capture stream
        dist.all_reduce(sync_token, group=dist.group.WORLD)
        push_start.record(capture_stream)

        # phase 2: push on comm stream (owner only)
        with torch.npu.stream(comm_stream):
            comm_stream.wait_event(push_start)
            completion = transport.push_active_bundle(bundle, pages, comm_stream)
            push_done.record(comm_stream)
        retained.append(completion)

        # phase 3: capture waits push_done before DONE barrier
        capture_stream.wait_event(push_done)
        dist.all_reduce(sync_token, group=dist.group.WORLD)

        # phase 4: recv_start -> receive on comm stream (non-owner only)
        recv_start.record(capture_stream)
        with torch.npu.stream(comm_stream):
            comm_stream.wait_event(recv_start)
            completion2 = transport.receive_active_bundle(bundle, pages, comm_stream)
            recv_done.record(comm_stream)
        retained.append(completion2)

        # phase 5: capture waits recv_done
        capture_stream.wait_event(recv_done)

    print(f"[POC8B][rank={rank}] capture OK", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # --- Replay with different patterns ---
    REPLAY_COUNT = 100

    for i in range(REPLAY_COUNT):
        pattern = float(i + 1) * 10.0  # 10, 20, 30, ...
        if rank == owner_rank:
            # Owner fills KV with pattern
            owner_kv.fill_(pattern)
            torch.npu.synchronize()
        dist.barrier()

        graph.replay()
        torch.npu.synchronize()
        dist.barrier()

        if rank != owner_rank:
            # Consumer: verify the received data matches owner's pattern.
            block_data = owner_kv[:block_size, 0, 0]
            actual = block_data[0].item()
            if not (abs(actual - pattern) < 0.5):
                print(
                    f"[POC8B][rank={rank}] FAIL replay {i}: "
                    f"expected={pattern}, actual={actual}",
                    flush=True,
                )
                sys.exit(1)
        if rank == 0 and i % 20 == 0:
            print(f"[POC8B] replay {i} OK", flush=True)

    print(f"[POC8B][rank={rank}] PASS (owner=rank{owner_rank}, {REPLAY_COUNT} replays)", flush=True)

    # Do not call transport.close() — it triggers double-free in MemFabric
    # shutdown; the process is exiting anyway and the graph correctness is
    # already verified.
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
