"""Validate scheduler read leases, remote pointers, mutation and scratch reuse."""
import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_allocation import IpcAllocationPool
from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_remote_read_transport import IpcRemoteReadKVPPTransport
from vllm_ascend.worker.v2.kvpp import KVPPScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')
    torch.npu.set_stream(torch.npu.Stream())
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    group = SimpleNamespace(world_size=world, rank_in_group=rank, ranks=list(range(world)), cpu_group=dist.group.WORLD)
    pool = IpcAllocationPool(device)
    names = [f'model.layers.{i}.self_attn.attn' for i in range(world + 2)]
    caches = {}; owners = {}; slots = {}
    for index, layer in enumerate(names):
        owner = index % world
        slot = ('owner', index) if rank == owner else ('scratch', index % 2)
        if slot not in slots:
            slots[slot] = (pool.allocate(17 * 129).view(17,129),
                           pool.allocate(17 * 8 * 2).view(torch.bfloat16).view(17,8),
                           pool.allocate(17 * 2).view(torch.float16).view(17,1))
        for name, tensor in zip((layer, layer.replace('.attn','.indexer'), layer.replace('.attn','.scale')), slots[slot]):
            owners[name] = owner; caches[name] = tensor
    transport = IpcRemoteReadKVPPTransport(group,17,pool,owners,
                    '/home/recky/a3-kv-transfer-20260907/build-release/lib/libkv_copy.so')
    scheduler = KVPPScheduler(group,owners,caches,17,1,17,transport,attention_layer_names=tuple(names))
    checks = 0
    for epoch in range(4):
        scheduler.schedule_forward(torch.tensor([[0,1,2]],device=device,dtype=torch.int32),[3])
        for index, layer in enumerate(names):
            scheduler.wait_for_layer(layer)
            parts = tuple(caches[n] for n in (layer,layer.replace('.attn','.indexer'),layer.replace('.attn','.scale')))
            value = 1 + epoch*10 + index
            for tensor in parts:
                tensor.fill_(value if rank == owners[layer] else -3)
            remote = scheduler.remote_read_cache(layer,parts)
            if rank == world-1: time.sleep(0.005)
            for local, view in zip(parts,remote):
                # Device arithmetic exercises the imported address, and the
                # untouched local sentinel rules out a hidden staging copy.
                assert torch.all((view.float()+1).cpu() == value+1)
                if rank != owners[layer]:
                    assert view.data_ptr() != local.data_ptr()
                    assert torch.all(local.cpu() == -3)
                checks += 1
        scheduler.complete_forward()
    scheduler.close(); pool.close()
    out = Path('results')/args.label/f'rank-{rank}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(rank=rank,world=world,checks=checks,passed=True,
                                  remote_tensor_reads=transport.remote_tensor_reads,cleanup=True)))
    dist.destroy_process_group()


if __name__ == '__main__': main()
