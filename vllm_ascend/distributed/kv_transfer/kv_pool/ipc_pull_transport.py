# SPDX-License-Identifier: Apache-2.0
"""Conservative, opt-in A3 IPC pull transport for KVPP development.

This first integration keeps host collectives and host page validation. It is
not the optimized hot path: page IDs are copied to CPU once per forward plan.
No payload staging is allocated. Existing physical page positions are kept.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist

from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_allocation import IpcAllocationPool, IpcMapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_lifecycle import TransferGeneration, TransferTicket


class IpcPullKVPPTransport:
    uses_direct_pull = True

    def __init__(self, group: Any, num_physical_pages: int, pool: IpcAllocationPool,
                 owners: dict[str, int], kernel_library: str, cores: int = 8) -> None:
        if not 1 <= cores <= 40 or num_physical_pages <= 0:
            raise ValueError("Invalid core count or physical page count")
        self.group = group
        self.rank = group.rank_in_group
        self.num_pages = num_physical_pages
        self.pool = pool
        self.owners = owners
        self.cores = cores
        self.library = ctypes.CDLL(str(Path(kernel_library).resolve(strict=True)))
        self.launch = self.library.kv_copy_launch
        self.launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.launch.restype = None
        self.regions: dict[str, list[dict[str, int]]] = {}
        self.mappings: dict[tuple[int, int], IpcMapping] = {}
        self.local_allocations = {item.identifier: item for item in pool.allocations}
        self.forward = 0
        self.page_plan: Any = None
        self.page_ids: list[int] = []
        self.tickets: list[TransferTicket] = []
        self.last_uses: dict[tuple[int, int], Any] = {}
        self.closed = False
        self._failed_generation: tuple[str, int, str] | None = None
        self._pending_resources: list[Any] = []

    def _gather(self, value: Any) -> list[Any]:
        output = [None] * self.group.world_size
        dist.all_gather_object(output, value, group=self.group.cpu_group)
        return output

    def initialize_transport(self, caches: dict[str, Any], bundles: tuple[tuple[str, ...], ...],
                             max_active_pages: int) -> None:
        if self.pool.active_transports:
            raise RuntimeError("This allocation pool already belongs to a transport")
        self.pool.active_transports += 1
        pids = self._gather(self.pool.runtime.pid())
        epochs = self._gather(str(uuid4()) if self.rank == 0 else None)
        self.engine_epoch = epochs[0]
        self.caches = caches
        for name, cache in caches.items():
            tensors = (cache,) if isinstance(cache, torch.Tensor) else tuple(cache)
            self.regions[name] = []
            for tensor in tensors:
                if tensor.ndim == 0 or tensor.shape[0] % self.num_pages:
                    raise ValueError(f"KV cache {name} cannot be divided into physical pages")
                rows = tensor.shape[0] // self.num_pages
                page = tensor[:rows]
                if rows == 0 or not page.is_contiguous():
                    raise ValueError(f"KV cache {name} has a non-contiguous or empty page")
                stride = tensor.stride(0) * tensor.element_size() * rows
                length = page.numel() * page.element_size()
                if length > stride:
                    raise ValueError("Overlapping physical pages")
                allocation, offset = self.pool.resolve(tensor, (self.num_pages - 1) * stride + length)
                self.regions[name].append({"allocation": allocation.identifier, "epoch": allocation.epoch,
                                           "offset": offset, "stride": stride, "length": length})
        for bundle in bundles:
            if len({self.owners[name] for name in bundle}) != 1:
                raise ValueError("All caches in a transfer bundle must have the same owner")
            ranges: dict[int, list[tuple[int, int]]] = {}
            for name in bundle:
                for region in self.regions[name]:
                    start = region["offset"]
                    end = start + (self.num_pages - 1) * region["stride"] + region["length"]
                    ranges.setdefault(region["allocation"], []).append((start, end))
            for allocation_ranges in ranges.values():
                allocation_ranges.sort()
                if any(right[0] < left[1] for left, right in zip(allocation_ranges, allocation_ranges[1:])):
                    raise ValueError("Overlapping cache regions in one bundle")
        # One source export per underlying allocation, even when several cache
        # views or logical layers alias that storage.
        source_ids = {region["allocation"] for name, regions in self.regions.items()
                      if self.owners[name] == self.rank for region in regions}
        exports = {identifier: self.local_allocations[identifier].export(
            [pid for rank, pid in enumerate(pids) if rank != self.rank]) for identifier in source_ids}
        self.peer_metadata = self._gather({"exports": exports, "regions": self.regions})
        for peer, metadata in enumerate(self.peer_metadata):
            if peer != self.rank:
                for identifier, exported in metadata["exports"].items():
                    self.mappings[peer, identifier] = IpcMapping(self.pool.runtime, exported)
        self._gather("mappings-ready")

    def prefetch(self, layer: str, bundle: tuple[str, ...], pages: Any,
                 source_ready: Any, scratch_ready: Any, stream: Any) -> TransferTicket:
        if self._failed_generation is not None:
            raise RuntimeError(f"IPC transport has a failed generation: {self._failed_generation!r}")
        try:
            return self._prefetch_impl(layer, bundle, pages, source_ready, scratch_ready, stream)
        except BaseException:
            # A later bundle submission or peer ACK can fail after an earlier
            # kernel launch. Keep descriptors/maps alive and refuse manual
            # reclamation when a distributed read lease cannot be proven done.
            self._failed_generation = (self.engine_epoch, self.forward, layer)
            raise

    def _prefetch_impl(self, layer: str, bundle: tuple[str, ...], pages: Any,
                       source_ready: Any, scratch_ready: Any, stream: Any) -> TransferTicket:
        if self.closed:
            raise RuntimeError("Transport is closed")
        if pages is not self.page_plan:
            # The worker's default stream need not be the model compute
            # stream. Metadata must be complete before the host validation
            # reads device page IDs/masks, not only before the copy kernel.
            source_ready.synchronize()
            if any(ticket.last_cache_use is None for ticket in self.tickets):
                raise RuntimeError("Previous forward has an unreleased cache user")
            self.forward += 1
            self.page_plan = pages
            # Explicit conservative control-plane baseline, not claimed to be
            # free of a host synchronization. Preserve the device plan by ref.
            values = pages.physical_page_ids.detach().cpu().tolist()
            mask = pages.valid_page_mask.detach().cpu().tolist()
            self.page_ids = [int(value) for value, valid in zip(values, mask) if valid]
            if len(values) != len(mask) or len(set(self.page_ids)) != len(self.page_ids):
                raise ValueError("Malformed or duplicate physical page plan")
            if any(page < 0 or page >= self.num_pages for page in self.page_ids):
                raise ValueError("Physical page ID out of range")
        owner = self.owners[layer]
        source_regions = self.peer_metadata[owner]["regions"]
        epochs = tuple(sorted({(region["allocation"], region["epoch"])
                               for name in bundle for region in source_regions[name]}))
        generation = TransferGeneration(self.engine_epoch, self.forward, layer, bundle, self.forward, epochs)
        slots = tuple(sorted({(region["allocation"], region["epoch"])
                              for name in bundle for region in self.regions[name]}))
        # Last-use events are indexed by actual allocation identity. Different
        # layouts may have a different non-owner sequence and alias pattern.
        for slot in slots:
            if any(slot in ticket.target_slots and ticket.last_cache_use is None for ticket in self.tickets):
                raise RuntimeError(f"Scratch allocation {slot} still has an unrecorded cache reader")
        if self.rank == owner:
            source_ready.synchronize()
        readiness = self._gather((generation, tuple(self.page_ids)))
        if any(item != readiness[0] for item in readiness):
            raise RuntimeError(f"KVPP generation/page-plan mismatch: {generation!r}")
        retained: list[Any] = [pages, self.pool, self.caches, source_ready, scratch_ready]
        self._pending_resources = retained
        with torch.npu.stream(stream):
            stream.wait_event(source_ready)
            stream.wait_event(scratch_ready)
            for slot in slots:
                if slot in self.last_uses:
                    stream.wait_event(self.last_uses[slot])
            self._copy_bundle(owner, bundle, source_regions, stream, retained)
            ready = torch.npu.Event()
            ready.record(stream)
        ticket = TransferTicket(generation, slots, owner,
                                frozenset(rank for rank in range(self.group.world_size) if rank != owner),
                                tuple(retained), local_cache_ready=ready)
        # Host ACK follows device completion of the WHOLE bundle. The owner
        # cannot write a shared tail page until every reader has acknowledged.
        ready.synchronize()
        acknowledgements = self._gather((self.rank, generation))
        for rank, acknowledged_generation in acknowledgements:
            if rank != owner:
                ticket.acknowledge_reader(rank, acknowledged_generation)
        self.tickets.append(ticket)
        self._pending_resources = []
        return ticket

    def _copy_bundle(self, owner: int, bundle: tuple[str, ...], source_regions: Any,
                     stream: Any, retained: list[Any]) -> None:
        if self.rank != owner:
            for name in bundle:
                remote = source_regions[name]
                local = self.regions[name]
                if len(remote) != len(local):
                    raise RuntimeError("Peer cache bundle layout mismatch")
                for source, destination in zip(remote, local):
                    if source["length"] != destination["length"]:
                        raise RuntimeError("Peer physical page length mismatch")
                    mapping = self.mappings[owner, source["allocation"]]
                    allocation = self.local_allocations[destination["allocation"]]
                    descriptors = [(source["offset"] + page * source["stride"],
                                    destination["offset"] + page * destination["stride"],
                                    source["length"]) for page in self.page_ids]
                    tensor = torch.tensor(descriptors, dtype=torch.int64,
                                          device=self.pool.device).reshape(-1, 3)
                    retained.extend((mapping, allocation, tensor))
                    if descriptors:
                        self.launch(self.cores, stream.npu_stream, mapping.pointer, allocation.pointer,
                                    tensor.data_ptr(), len(descriptors), 0)

    def release_after_last_cache_use(self, ticket: TransferTicket, compute_stream: Any) -> None:
        event = torch.npu.Event()
        event.record(compute_stream)
        ticket.release_after_last_cache_use(event)
        for slot in ticket.target_slots:
            self.last_uses[slot] = event

    def drain(self) -> None:
        if self._failed_generation is not None:
            raise RuntimeError(f"Cannot reclaim IPC resources after failed generation {self._failed_generation!r}; "
                               "the process group must be aborted before process teardown")
        for ticket in self.tickets:
            ticket.drain()
        self.tickets.clear()
        self.page_plan = None

    def close(self) -> None:
        if self.closed:
            return
        self.drain()
        self._gather("drained")
        for mapping in self.mappings.values():
            mapping.close()
        self.mappings.clear()
        self._gather("imports-closed")
        self.pool.close_exports()
        self._gather("exports-closed")
        self.pool.active_transports -= 1
        self.closed = True
