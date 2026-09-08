# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun on reserved NPUs; no model weights are needed."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.core.kv_cache_placement import KVPPBufferSpec, KVPPPhysicalCachePlan, build_kvpp_layer_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.broadcast_transport import BroadcastKVPPTransport
from vllm_ascend.worker import kvpp_cache
from vllm_ascend.worker.v2.kvpp import KVPPScheduler


def run(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=120))
    if world % args.pipeline_parallel_size:
        raise ValueError("World size must be divisible by PP size.")
    width = world // args.pipeline_parallel_size
    for stage in range(args.pipeline_parallel_size):
        ranks = list(range(stage * width, (stage + 1) * width))
        device_group = dist.new_group(ranks, backend="hccl", timeout=datetime.timedelta(seconds=120))
        if rank in ranks:
            group = SimpleNamespace(ranks=ranks, device_group=device_group)
            local_rank = ranks.index(rank)
    names = tuple(f"model.layers.{i}.self_attn.attn" for i in range(2 * width + 3))
    patterns = ((512, 128), (656,), (512, 128, 256), (656, 128, 2), (512, 128, 128, 2), (656, 256))
    layouts = {}
    tensor_specs = {}
    owners = {}
    base_count, remainder = divmod(len(names), width)
    cursor = 0
    for owner in range(width):
        for name in names[cursor : cursor + base_count + int(owner < remainder)]:
            owners[name] = owner
        cursor += base_count + int(owner < remainder)
    for i, name in enumerate(names):
        specs = {name: tuple(KVPPBufferSpec(size, torch.int8, 32) for size in patterns[i % len(patterns)])}
        tensor_specs.update(specs)
        layouts[name] = build_kvpp_layer_layout((name,), specs, 17)
    max_size = max(size for _, size in layouts.values())
    # Guard bytes sit outside the span handed to production transport.
    storages = []
    mtp_names = tuple(f"mtp-{i}" for i in range(args.mtp_layers))
    tensor_specs.update({name: (KVPPBufferSpec(max_size * 2, torch.int8, 32),) for name in mtp_names})
    plan = KVPPPhysicalCachePlan({}, owners, {name: (name,) for name in (*names, *mtp_names)}, tensor_specs, local_rank)

    def guarded_zeros(size, *, dtype, device):
        storage = torch.full((size + 128,), -71, dtype=dtype, device=device)
        storages.append(storage)
        return storage[64:-64].zero_()

    # Inject synthetic specs and a group, while exercising the production allocator.
    with (
        patch.object(kvpp_cache, "create_kvpp_cache_allocation_plan", return_value=plan),
        patch.object(kvpp_cache, "get_kvpp_group", return_value=SimpleNamespace(rank_in_group=local_rank)),
        patch.object(torch, "zeros", side_effect=guarded_zeros),
    ):
        caches = kvpp_cache.allocate_kvpp_cache(
            object(), SimpleNamespace(num_blocks=17, kv_cache_groups=[]), torch.device("npu")
        )
    assert sum(storage.numel() - 128 for storage in storages) == plan.get_physical_memory_bytes(17)
    raw_buffers = {}
    transfers = {}
    for i, name in enumerate(names):
        layout, size = layouts[name]
        first = caches[name][0].view(torch.float16)
        raw = torch.empty(0, dtype=torch.int8, device="npu").set_(
            first.untyped_storage(), first.storage_offset() * first.element_size(), (size,), (1,)
        )
        raw_buffers[name] = raw
        transfers[name] = (
            (raw,)
            if args.granularity == "layer"
            else tuple(raw[offset : offset + length] for offset, length in layout[name])
        )
    mtp = [caches[name][0].fill_(37) for name in mtp_names]
    scheduler = KVPPScheduler(BroadcastKVPPTransport(group, owners, transfers), names)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    compute = torch.npu.Stream()
    try:
        with (output / f"rank-{rank}.jsonl").open("w") as log:
            for iteration in range(args.iterations):
                expected = {
                    name: (
                        (torch.arange(raw.numel(), dtype=torch.int32, device="npu") * 17 + i * 7 + iteration) % 127
                    ).to(torch.int8)
                    for i, (name, raw) in enumerate(raw_buffers.items())
                }
                torch.npu.current_stream().synchronize()
                # A cold first chunk writes owner history without broadcasting.
                with torch.npu.stream(compute):
                    scheduler.schedule_forward(False)
                    for i, name in enumerate(names):
                        scheduler.wait_for_layer(name)
                        if owners[name] == local_rank:
                            raw_buffers[name].copy_(expected[name])
                    scheduler.complete_forward()
                compute.synchronize()
                dist.barrier(group=group.device_group)
                time.sleep((rank % 3) * 0.003)
                snapshots = []
                with torch.npu.stream(compute):
                    scheduler.schedule_forward(True)
                    for i, name in enumerate(names):
                        scheduler.wait_for_layer(name)
                        # Clone queues a cache reader on a non-default stream.
                        # The next layer starts before this layer is overwritten.
                        snapshots.append(tuple(part.clone() for part in transfers[name]))
                        raw_buffers[name].fill_(-19)
                    scheduler.complete_forward()
                compute.synchronize()
                for i, parts in enumerate(snapshots):
                    name = names[i]
                    layout, _ = layouts[name]
                    expected_parts = (
                        (expected[name],)
                        if args.granularity == "layer"
                        else tuple(expected[name][offset : offset + length] for offset, length in layout[name])
                    )
                    assert all(torch.equal(part, reference) for part, reference in zip(parts, expected_parts))
                for storage in storages:
                    assert bool(torch.all(storage[:64] == -71).item())
                    assert bool(torch.all(storage[-64:] == -71).item())
                assert all(bool(torch.all(buffer == 37).item()) for buffer in mtp)
                log.write(
                    json.dumps(
                        {
                            "rank": rank,
                            "iteration": iteration,
                            "status": "passed",
                            "granularity": args.granularity,
                            "pp": args.pipeline_parallel_size,
                        }
                    )
                    + "\n"
                )
                log.flush()
    finally:
        scheduler.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularity", choices=("tensor", "layer"), required=True)
    parser.add_argument("--layout-matrix", choices=("mixed",), default="mixed")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pipeline-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mtp-layers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
