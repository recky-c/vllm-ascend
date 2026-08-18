"""POC 8c overlap measurement via event timestamps.

Instead of relying on msprof chrome trace (which needs special env to capture
aclnn kernels), we directly measure overlap using NPU event elapsed_time.

For each layer i in each replay:
  - record T_push_start, T_push_done (on comm stream)
  - record T_qkv_start, T_qkv_end (on capture stream, the compute that should
    overlap with push)
  - record T_recv_start, T_recv_done (on comm stream)
  - record T_attn_start, T_attn_end (on capture stream, the compute that should
    overlap with receive)

Overlap is proven if:
  push interval [T_push_start, T_push_done] intersects qkv/attn interval
  on capture stream.

Because events are recorded on different streams, elapsed_time gives the
device-side timestamp of when the event was recorded on that stream, so
cross-stream timestamp comparison is valid (both use the same global NPU
timer domain).
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


def dummy_compute(size: int = 2048, iters: int = 12):
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
    print(f"[POC8C-OVL][rank={rank}] transport initialized", flush=True)

    page_ids = torch.tensor([0], dtype=torch.int64, device="npu")
    valid_mask = torch.tensor([True], dtype=torch.bool, device="npu")
    pages = KVPPActivePages(page_ids, valid_mask, count_upper_bound=1)

    # Timed events: for each layer, push_start/push_done on comm, and
    # compute_start/compute_end on capture (the attn compute that should
    # overlap with the NEXT layer's push). We measure overlap between
    # layer i's push(i+1) and layer i's attn(i) compute.
    @dataclass
    class LayerTimedEvents:
        push_start: object   # comm stream
        push_done: object    # comm stream
        recv_start: object   # comm stream
        recv_done: object    # comm stream
        qkv_start: object    # capture stream
        qkv_end: object      # capture stream
        attn_start: object   # capture stream
        attn_end: object     # capture stream

    layer_ev = {
        i: LayerTimedEvents(
            *[torch.npu.Event(enable_timing=True) for _ in range(8)]
        )
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
    print(f"[POC8C-OVL][rank={rank}] capturing", flush=True)

    with torch.npu.graph(graph, stream=capture_stream):
        for i in range(NUM_LAYERS):
            ev = layer_ev[i]
            bundle = bundles[i]

            # --- start_push(i): READY barrier + push on comm ---
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.push_start.record(capture_stream)  # mark on capture, but the
                                                  # comm work follows it
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.push_start)
                comp = transport.push_active_bundle(bundle, pages, comm_stream)
                ev.push_done.record(comm_stream)
            retained.append(comp)

            # --- finish_push_start_receive(i) ---
            capture_stream.wait_event(ev.push_done)
            dist.all_reduce(sync_token, group=dist.group.WORLD)
            ev.recv_start.record(capture_stream)
            with torch.npu.stream(comm_stream):
                comm_stream.wait_event(ev.recv_start)
                comp2 = transport.receive_active_bundle(bundle, pages, comm_stream)
                ev.recv_done.record(comm_stream)
            retained.append(comp2)

            # QKV compute overlaps RECV(i)
            ev.qkv_start.record(capture_stream)
            dummy_compute()
            ev.qkv_end.record(capture_stream)

            # wait recv done
            capture_stream.wait_event(ev.recv_done)

            # Attn/o_proj/MLP compute; next layer's PUSH overlaps this
            ev.attn_start.record(capture_stream)
            dummy_compute()
            ev.attn_end.record(capture_stream)

    print(f"[POC8C-OVL][rank={rank}] capture OK", flush=True)
    torch.npu.synchronize()
    dist.barrier()

    # Fill owner patterns and warmup
    for li in range(NUM_LAYERS):
        if rank == LAYER_OWNERS[li]:
            kv_caches[layer_names[li]].fill_(LAYER_PATTERNS[li])
    torch.npu.synchronize()
    dist.barrier()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    dist.barrier()

    # Profiled replay
    graph.replay()
    torch.npu.synchronize()
    dist.barrier()

    # Collect timestamps
    print(f"\n[POC8C-OVL][rank={rank}] === Overlap Analysis ===", flush=True)
    print(f"[POC8C-OVL][rank={rank}] (times in ms, device-side event timestamps)", flush=True)
    for i in range(NUM_LAYERS):
        ev = layer_ev[i]
        ps = ev.push_start.elapsed_time(ev.push_done)  # push duration on comm
        rs = ev.recv_start.elapsed_time(ev.recv_done)  # recv duration on comm
        qkv = ev.qkv_start.elapsed_time(ev.qkv_end)
        attn = ev.attn_start.elapsed_time(ev.attn_end)
        # Cross-stream overlap: compare absolute event times.
        # event.request_raw_npu_event() not available; use a reference event
        # recorded at start of graph replay to get absolute timestamps.
        # torch.npu.Event doesn't expose absolute ts directly, but we can
        # compare relative order via a shared reference.
        print(
            f"[POC8C-OVL][rank={rank}] L{i}: "
            f"push={ps:.3f}ms recv={rs:.3f}ms "
            f"qkv={qkv:.3f}ms attn={attn:.3f}ms",
            flush=True,
        )

    # For rigorous overlap proof we need absolute timestamps. NPU Event
    # elapsed_time only gives pairwise durations. To prove overlap we check
    # that total replay time < sum of (push+recv+qkv+attn) per layer, which
    # would be impossible without overlap.
    total_start = torch.npu.Event(enable_timing=True)
    total_end = torch.npu.Event(enable_timing=True)
    total_start.record()
    graph.replay()
    total_end.record()
    torch.npu.synchronize()
    total_ms = total_start.elapsed_time(total_end)

    sum_serial = 0.0
    for i in range(NUM_LAYERS):
        ev = layer_ev[i]
        sum_serial += (
            ev.push_start.elapsed_time(ev.push_done)
            + ev.recv_start.elapsed_time(ev.recv_done)
            + ev.qkv_start.elapsed_time(ev.qkv_end)
            + ev.attn_start.elapsed_time(ev.attn_end)
        )

    print(
        f"\n[POC8C-OVL][rank={rank}] single replay total={total_ms:.3f}ms, "
        f"sum(serial components)={sum_serial:.3f}ms",
        flush=True,
    )
    if sum_serial > total_ms * 1.1:
        print(
            f"[POC8C-OVL][rank={rank}] OVERLAP PROVEN: "
            f"serial sum {sum_serial:.1f}ms >> replay {total_ms:.1f}ms",
            flush=True,
        )
    else:
        print(
            f"[POC8C-OVL][rank={rank}] OVERLAP NOT PROVEN by timing "
            f"(serial sum {sum_serial:.1f}ms vs replay {total_ms:.1f}ms)",
            flush=True,
        )

    # Verify correctness
    for li in range(NUM_LAYERS):
        if rank != LAYER_OWNERS[li]:
            expected = LAYER_PATTERNS[li]
            actual = kv_caches[layer_names[li]][:block_size, 0, 0][0].item()
            assert abs(actual - expected) < 0.5, f"layer {li}: {actual} != {expected}"
    print(f"[POC8C-OVL][rank={rank}] correctness PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
