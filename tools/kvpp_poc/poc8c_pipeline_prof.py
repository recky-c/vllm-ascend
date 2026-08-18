"""POC 8c with profiler: verify real overlap in timeline.

Same as poc8c but runs a few replays under torch profiler to capture the
NPU timeline, proving PUSH/RECV on comm_stream overlap with compute on
capture stream.
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
LAYER_PATTERNS = {0: 100.0, 1: 200.0, 2: 300.0, 3: 400.0}
SLOT_OF_LAYER = {0: 0, 1: 1, 2: 0, 3: 1}


def dummy_compute(size: int = 2048, iters: int = 8):
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
    print(f"[POC8C-PROF][rank={rank}] transport initialized", flush=True)

    page_ids = torch.tensor([0], dtype=torch.int64, device="npu")
    valid_mask = torch.tensor([True], dtype=torch.bool, device="npu")
    pages = KVPPActivePages(page_ids, valid_mask, count_upper_bound=1)

    @dataclass
    class LayerEvents:
        push_start: object
        push_done: object
        recv_start: object
        recv_done: object

    layer_events = {
        i: LayerEvents(torch.npu.Event(), torch.npu.Event(),
                       torch.npu.Event(), torch.npu.Event())
        for i in range(NUM_LAYERS)
    }
    capture_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    sync_token = torch.zeros(1, dtype=torch.int32, device="npu")
    retained = []
    bundles = {i: (layer_names[i],) for i in range(NUM_LAYERS)}

    torch.npu.synchronize()
    dist.barrier()

    graph = torch.npu.NPUGraph()
    print(f"[POC8C-PROF][rank={rank}] capturing", flush=True)

    with torch.npu.graph(graph, stream=capture_stream):
        for i in range(NUM_LAYERS):
            ev = layer_events[i]
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

            dummy_compute()  # qkv_i overlaps RECV(i)
            capture_stream.wait_event(ev.recv_done)
            dummy_compute()  # attn_i; next PUSH overlaps

    print(f"[POC8C-PROF][rank={rank}] capture OK", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # Fill owner patterns
    for li in range(NUM_LAYERS):
        if rank == LAYER_OWNERS[li]:
            kv_caches[layer_names[li]].fill_(LAYER_PATTERNS[li])
    torch.npu.synchronize()
    dist.barrier()

    # Warmup replays (not profiled)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    dist.barrier()

    # Profiled replays
    from torch_npu.profiler import profile, ProfilerActivity
    from torch_npu.profiler.experimental_config import _ExperimentalConfig

    prof_dir = f"/tmp/poc8c_prof_rank{rank}"
    os.makedirs(prof_dir, exist_ok=True)
    print(f"[POC8C-PROF][rank={rank}] starting profiler -> {prof_dir}", flush=True)

    exp_config = _ExperimentalConfig(
        profiler_level="Level1",
        aic_metrics=0,
        data_simplification=False,
    )
    with profile(
        activities=[ProfilerActivity.NPU, ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
        experimental_config=exp_config,
    ) as prof:
        for _ in range(5):
            graph.replay()
    prof.export_chrome_trace(f"{prof_dir}/trace.json")
    torch.npu.synchronize()
    dist.barrier()
    print(f"[POC8C-PROF][rank={rank}] profiler done -> {prof_dir}/trace.json", flush=True)

    # Verify correctness
    for li in range(NUM_LAYERS):
        if rank != LAYER_OWNERS[li]:
            expected = LAYER_PATTERNS[li]
            actual = kv_caches[layer_names[li]][:block_size, 0, 0][0].item()
            assert abs(actual - expected) < 0.5, f"layer {li}: {actual} != {expected}"

    print(f"[POC8C-PROF][rank={rank}] PASS, profile at {prof_dir}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
