"""Run the modified real KVPPScheduler with IPC-backed, physically aliased KV."""
import importlib.util
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
import torch.distributed as dist


def load(name, filename):
    if not (Path(__file__).parent / filename).exists():
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label')
    parser.add_argument('--backend', choices=['ipc_pull', 'ipc_broadcast', 'ipc_fullpage'], default='ipc_pull')
    args = parser.parse_args()
    allocation = load('vllm_ascend.distributed.kv_transfer.kv_pool.ipc_allocation', 'ipc_allocation.py')
    load('vllm_ascend.distributed.kv_transfer.kv_pool.ipc_lifecycle', 'ipc_lifecycle.py')
    backend = load('vllm_ascend.distributed.kv_transfer.kv_pool.ipc_pull_transport', 'ipc_pull_transport.py')
    transport_class = backend.IpcPullKVPPTransport
    if args.backend == 'ipc_broadcast':
        from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_broadcast_transport import IpcBroadcastKVPPTransport
        transport_class = IpcBroadcastKVPPTransport
    if args.backend == 'ipc_fullpage':
        from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_fullpage_broadcast_transport import IpcFullPageBroadcastKVPPTransport
        transport_class = IpcFullPageBroadcastKVPPTransport
    scheduler_module = load('vllm_ascend.worker.v2.kvpp', 'kvpp.py')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    torch.npu.set_device(device)
    # The transfer worker has its own thread-local current stream. Validate
    # that page-plan readiness is honored across that boundary.
    compute_stream = torch.npu.Stream()
    torch.npu.set_stream(compute_stream)
    torch_npu.npu.config.allow_internal_format = False
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    group = SimpleNamespace(world_size=world, rank_in_group=rank, ranks=list(range(world)), cpu_group=dist.group.WORLD)
    pool = allocation.IpcAllocationPool(device)
    page_count = 17
    layers = [f'model.layers.{i}.self_attn.attn' for i in range(10)]
    owners, caches, expected, layout_slots = {}, {}, {}, {}
    sequence = [0, 0]
    layer_bundles = {}
    for index, layer in enumerate(layers):
        layout = index % 2
        owner = index % world
        slot = ('owner', index) if owner == rank else ('scratch', layout, sequence[layout] % 2)
        if owner != rank:
            sequence[layout] += 1
        if slot not in layout_slots:
            raw = pool.allocate(16384)
            layout_slots[slot] = raw
        raw = layout_slots[slot]
        names = (layer, layer.replace('.attn', '.indexer'), layer.replace('.attn', '.scale'))
        specs = ((0, 129 + layout * 16, 160), (4096, 33 + layout * 8, 64), (8192, 12, 32))
        if args.backend == 'ipc_fullpage':
            specs = tuple((offset, length, length) for offset, length, stride in specs)
        layer_bundles[layer] = names
        for name, (offset, length, stride) in zip(names, specs):
            owners[name] = owner
            caches[name] = raw.as_strided((page_count, length), (stride, 1), offset)
            expected[name] = torch.zeros((page_count, length), dtype=torch.int8)
    class DelayedReaderTransport(transport_class):
        uses_direct_pull = True

        def _gather(self, value):
            result = super()._gather(value)
            # Delay a reader AFTER source publication but BEFORE remote load.
            if (rank == world - 1 and isinstance(value, tuple) and len(value) == 2
                    and hasattr(value[0], 'layer') and '.layers.3.' in value[0].layer):
                time.sleep(0.025)
            return result

    transport = DelayedReaderTransport(group, page_count, pool, owners,
                                            './build-release/lib/libkv_copy.so', cores=8)
    scheduler = scheduler_module.KVPPScheduler(group, owners, caches, page_count, 1, page_count,
                                              transport, attention_layer_names=tuple(layers))
    label = args.label or f'product-lifecycle-{world}'
    logpath = Path(f'results/{label}/rank-{rank}.jsonl')
    logpath.parent.mkdir(parents=True, exist_ok=True)
    log = logpath.open('w')
    def emit(row):
        row.update(rank=rank, world_size=world, compute_stream='nondefault')
        log.write(json.dumps(row) + '\n')
        log.flush()
    for forward in range(6):
        active = [(page + forward * 3) % page_count for page in [7, 1, 16, 2, 7, 9]]
        unique = list(dict.fromkeys(active))
        # New request / page-plan identities, but original physical positions.
        for index, layer in enumerate(layers):
            if owners[layer] == rank:
                for kind, name in enumerate(layer_bundles[layer]):
                    pattern = ((torch.arange(expected[name].numel()).reshape(expected[name].shape)
                                + forward * 29 + index * 11 + kind * 7) % 101).to(torch.int8)
                    expected[name].copy_(pattern)
                    caches[name].copy_(pattern)
        dist.barrier()
        scheduler.schedule_forward(torch.tensor([active], dtype=torch.int32, device=device), [len(active)])
        outputs = []
        for index, layer in enumerate(layers):
            # Delay the consumer before it consumes the next prefetch ticket.
            if rank == world - 1 and index == 3:
                time.sleep(0.025)
            scheduler.wait_for_layer(layer)
            ticket = scheduler._active_transfer_ticket
            assert ticket.owner_source_reusable
            for kind, name in enumerate(layer_bundles[layer]):
                reference = ((torch.arange(expected[name].numel()).reshape(expected[name].shape)
                              + forward * 29 + index * 11 + kind * 7) % 101).to(torch.int8)
                # Consumer reads are submitted on NPU before any byte is changed.
                outputs.append((caches[name][unique].clone(), reference[unique], layer, name))
                if args.backend == 'ipc_fullpage':
                    outputs.append((caches[name].clone(), reference, layer, name + '.all_pages'))
                # Actual local new-token write into the shared historical tail page.
                caches[name][unique[-1], -1].fill_(forward + 111)
                outputs.append((caches[name][unique[-1], -1:].clone(), torch.tensor([forward + 111], dtype=torch.int8),
                                layer, name + ':local-new'))
            # A final cache load after the attention-shaped reads, to exercise
            # the last-reader boundary used by the following layer hook.
            final_name = layer_bundles[layer][-1]
            outputs.append((caches[final_name][unique[0]].clone(),
                            ((torch.arange(expected[final_name].numel()).reshape(expected[final_name].shape)
                              + forward * 29 + index * 11 + 14) % 101).to(torch.int8)[unique[0]],
                            layer, final_name + ':last-reader'))
            emit({'phase': 'ticket', 'forward': forward, 'layer': layer,
                  'generation': repr(ticket.generation), 'actual_slots': ticket.target_slots,
                  'all_reader_acks': ticket.owner_source_reusable})
        scheduler.complete_forward()
        for output, reference, layer, name in outputs:
            if not torch.equal(output.cpu(), reference):
                raise AssertionError(f'data mismatch rank={rank} forward={forward} layer={layer} cache={name}')
        emit({'phase': 'forward', 'forward': forward, 'npu_checks': len(outputs), 'pass': True})
    # Abort/early-return drain with next layer already submitted.
    scheduler.schedule_forward(torch.tensor([active], dtype=torch.int32, device=device), [len(active)])
    scheduler.wait_for_layer(layers[0])
    scheduler.complete_forward()
    assert scheduler._prefetch_future is None and not transport.tickets
    emit({'phase': 'early-return-drain', 'pass': True})
    scheduler.close()
    caches.clear()
    layout_slots.clear()
    pool.close()
    dist.destroy_process_group()
    emit({'phase': 'cleanup', 'pass': True})
    log.close()


if __name__ == '__main__':
    main()
