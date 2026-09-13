# SPDX-License-Identifier: Apache-2.0
"""Explicit two-device memcache whole-object roundtrip; requires a running MetaService."""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
import uuid
from pathlib import Path


def probe(rank, device, prefix, barrier, output):
    import importlib.metadata

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

    result = {"rank": rank, "device": device, "passed": False}
    try:
        torch.npu.set_device(device)
        backend = MemcacheBackend(None, device_id=device)
        blocks = 3
        bundles, components, expected = [], [], []
        for layer, sizes in enumerate(((1024, 512, 256), (1536, 512, 256), (2048, 512, 256))):
            host = (torch.arange(blocks * sum(sizes), dtype=torch.int32) + 17 * rank + layer).to(torch.uint8)
            raw = host.to(f"npu:{device}")
            bundles.append(raw)
            expected.append(host)
            cursor = 0
            for size in sizes:
                components.append(raw[cursor : cursor + blocks * size].view(blocks, size))
                cursor += blocks * size
        scratches = [torch.full((16384,), value, dtype=torch.uint8, device=f"npu:{device}") for value in (71, 93)]
        torch.npu.synchronize()
        ptrs, sizes = [raw.data_ptr() for raw in bundles], [raw.numel() for raw in bundles]
        backend.register_buffer(ptrs, sizes)
        keys = [f"{prefix}@owner:{rank}@block:{block}" for block in range(blocks)]
        addrs = [[part[block].data_ptr() for part in components] for block in range(blocks)]
        lengths = [[part.shape[1] for part in components] for _ in range(blocks)]
        result.update(
            keys=keys,
            registrations=list(zip(ptrs, sizes)),
            object_bytes=[sum(row) for row in lengths],
            memcache_version=importlib.metadata.version("memcache-hybrid"),
            memfabric_version=importlib.metadata.version("memfabric-hybrid"),
        )
        raw_put = backend.store.batch_put_from_layers(keys, addrs, lengths, 0)
        result["put_result"] = list(raw_put)
        assert list(raw_put) == [0] * blocks, raw_put
        barrier.wait(120)
        all_keys = [f"{prefix}@owner:{owner}@block:{block}" for owner in range(2) for block in range(blocks)]
        result["exists_result"] = backend.exists(all_keys)
        assert result["exists_result"] == [1] * 6, result["exists_result"]
        for iteration in range(2):
            if iteration:
                backend.put(keys, addrs, lengths)
            for raw in bundles:
                raw.zero_()
            torch.npu.synchronize()
            get_result = backend.get(keys, addrs, lengths)
            assert get_result == [0] * blocks, get_result
            torch.npu.synchronize()
            for actual, wanted in zip(bundles, expected):
                assert torch.equal(actual.cpu(), wanted), "Persistent bytes differ after pool load"
            for scratch, value in zip(scratches, (71, 93)):
                assert bool(torch.all(scratch.cpu() == value)), "Scratch was modified"
        result["sha256"] = [hashlib.sha256(raw.numpy().tobytes()).hexdigest() for raw in expected]
        barrier.wait(120)
        # Release SDK registrations while their NPU tensors are still alive.
        result["unregister_result"] = [backend.store.unregister_buffer(ptr, size) for ptr, size in zip(ptrs, sizes)]
        assert result["unregister_result"] == [0] * len(ptrs), result["unregister_result"]
        result["close_result"] = backend.store.close()
        assert result["close_result"] == 0, result["close_result"]
        result["passed"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        barrier.abort()
        raise
    finally:
        (Path(output) / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error("Pass two distinct visible logical devices")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    prefix = f"kvpp-probe-{uuid.uuid4().hex}"
    processes = [
        ctx.Process(target=probe, args=(rank, device, prefix, barrier, args.out)) for rank, device in enumerate(devices)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(10)
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"SDK probe failed: exitcodes={[process.exitcode for process in processes]}")
    print("PASS: two owner shards, six whole objects, repeated roundtrip, scratch unchanged", flush=True)


if __name__ == "__main__":
    main()
