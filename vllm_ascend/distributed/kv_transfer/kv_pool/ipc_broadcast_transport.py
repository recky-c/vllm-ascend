# SPDX-License-Identifier: Apache-2.0
"""Experimental packed HCCL broadcast with the same KVPP lifetime protocol."""
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_pull_transport import IpcPullKVPPTransport


class IpcBroadcastKVPPTransport(IpcPullKVPPTransport):
    def initialize_transport(self, caches: dict[str, Any], bundles: tuple[tuple[str, ...], ...],
                             max_active_pages: int) -> None:
        super().initialize_transport(caches, bundles, max_active_pages)
        # Use a separate communicator so interleaved TP/EP collectives do not
        # share ordering with the prefetch worker's broadcast stream.
        self.broadcast_group = dist.new_group(ranks=self.group.ranks, backend="hccl")
        max_page_bytes = max(sum(region["length"] for name in bundle for region in self.regions[name])
                             for bundle in bundles)
        self.wire = torch.empty(max_active_pages * max_page_bytes, dtype=torch.uint8, device=self.pool.device)

    def _copy_bundle(self, owner: int, bundle: tuple[str, ...], source_regions: Any,
                     stream: Any, retained: list[Any]) -> None:
        retained.append(self.wire)
        transfers = []
        packed = 0
        for name in bundle:
            remote, local = source_regions[name], self.regions[name]
            if len(remote) != len(local):
                raise RuntimeError("Peer cache bundle layout mismatch")
            for source, destination in zip(remote, local):
                if source["length"] != destination["length"]:
                    raise RuntimeError("Peer physical page length mismatch")
                length = source["length"]
                if self.rank == owner:
                    allocation = self.local_allocations[source["allocation"]]
                    descriptors = [(source["offset"] + page * source["stride"], packed + i * length, length)
                                   for i, page in enumerate(self.page_ids)]
                else:
                    allocation = self.local_allocations[destination["allocation"]]
                    descriptors = [(packed + i * length, destination["offset"] + page * destination["stride"], length)
                                   for i, page in enumerate(self.page_ids)]
                tensor = torch.tensor(descriptors, dtype=torch.int64, device=self.pool.device).reshape(-1, 3)
                retained.extend((allocation, tensor))
                transfers.append((allocation, tensor, len(descriptors)))
                packed += len(self.page_ids) * length
        if packed > self.wire.numel():
            raise RuntimeError("Broadcast payload exceeds initialization capacity")
        if not packed:
            return
        if self.rank == owner:
            for allocation, tensor, count in transfers:
                self.launch(self.cores, stream.npu_stream, allocation.pointer, self.wire.data_ptr(),
                            tensor.data_ptr(), count, 0)
        work = dist.broadcast(self.wire[:packed], src=self.group.ranks[owner],
                              group=self.broadcast_group, async_op=True)
        retained.append(work)
        work.wait()
        if self.rank != owner:
            for allocation, tensor, count in transfers:
                self.launch(self.cores, stream.npu_stream, self.wire.data_ptr(), allocation.pointer,
                            tensor.data_ptr(), count, 0)

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        # Parent drain refuses reclamation if a generation failed.
        dist.destroy_process_group(self.broadcast_group)
        self.wire = None
