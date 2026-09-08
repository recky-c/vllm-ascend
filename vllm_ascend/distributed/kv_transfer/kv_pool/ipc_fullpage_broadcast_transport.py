# SPDX-License-Identifier: Apache-2.0
"""Experimental direct broadcast of every physical page of a cache tensor."""
from typing import Any

import torch.distributed as dist

from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_pull_transport import IpcPullKVPPTransport


class IpcFullPageBroadcastKVPPTransport(IpcPullKVPPTransport):
    @staticmethod
    def region_extent(region: dict[str, int], num_pages: int) -> int:
        if num_pages <= 0 or region['length'] <= 0 or region['stride'] != region['length']:
            raise ValueError('Full-page direct broadcast requires contiguous nonempty physical pages')
        return num_pages * region['length']

    def initialize_transport(self, caches: dict[str, Any], bundles: tuple[tuple[str, ...], ...],
                             max_active_pages: int) -> None:
        super().initialize_transport(caches, bundles, max_active_pages)
        self.fullpage_views: dict[str, tuple[Any, ...]] = {}
        # Validate every peer before entering any device collective. Never
        # broadcast a strided envelope: gaps can belong to another cache view.
        for name, local_regions in self.regions.items():
            local_lengths = [self.region_extent(region, self.num_pages) for region in local_regions]
            for peer in self.peer_metadata:
                peer_lengths = [self.region_extent(region, self.num_pages) for region in peer['regions'][name]]
                if peer_lengths != local_lengths:
                    raise ValueError('Full-page broadcast peer cache geometry mismatch')
            self.fullpage_views[name] = tuple(
                self.local_allocations[region['allocation']].tensor.narrow(0, region['offset'], extent)
                for region, extent in zip(local_regions, local_lengths)
            )
        self.broadcast_group = dist.new_group(ranks=self.group.ranks, backend='hccl')

    def _copy_bundle(self, owner: int, bundle: tuple[str, ...], source_regions: Any,
                     stream: Any, retained: list[Any]) -> None:
        if not self.page_ids:
            return
        # Parent prefetch enforces source-ready, scratch last-use, generation
        # agreement and reader ACKs. Each view aliases the real attention cache;
        # no descriptor construction, pack, wire buffer or scatter is needed.
        for name in bundle:
            for view in self.fullpage_views[name]:
                retained.append(view)
                work = dist.broadcast(view, src=self.group.ranks[owner],
                                      group=self.broadcast_group, async_op=True)
                retained.append(work)
                work.wait()

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        dist.destroy_process_group(self.broadcast_group)
        self.fullpage_views.clear()
