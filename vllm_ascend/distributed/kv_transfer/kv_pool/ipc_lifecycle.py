# SPDX-License-Identifier: Apache-2.0
"""Host-side lease accounting for an explicit KV transfer.

Completion of a local receive and release of an owner's remote read lease are
different conditions. This module has no device or process-group side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class TransferGeneration:
    engine_epoch: str
    forward: int
    layer: str
    bundle: tuple[str, ...]
    page_plan: int
    allocation_epochs: tuple[tuple[int, int], ...]


@dataclass
class TransferTicket:
    generation: TransferGeneration
    # IDs come from actual allocation identity; never from layer parity.
    target_slots: tuple[tuple[int, int], ...]
    owner_rank: int
    reader_ranks: frozenset[int]
    retained_resources: tuple[Any, ...]
    local_cache_ready: Any | None = None
    owner_source_reusable: bool = False
    last_cache_use: Any | None = None
    _reader_acks: set[int] = field(default_factory=set)
    _released: bool = False

    def acknowledge_reader(self, rank: int, generation: TransferGeneration) -> None:
        if generation != self.generation:
            raise RuntimeError(f"Stale KVPP acknowledgement: {generation!r}; expected {self.generation!r}")
        if rank not in self.reader_ranks or rank in self._reader_acks:
            raise RuntimeError(f"Unexpected or duplicate KVPP reader acknowledgement: {rank}")
        self._reader_acks.add(rank)
        self.owner_source_reusable = self._reader_acks == set(self.reader_ranks)

    def wait_local_cache_ready(self, compute_stream: Any) -> None:
        if self.local_cache_ready is None:
            raise RuntimeError("Local KV completion dependency has not been established")
        compute_stream.wait_event(self.local_cache_ready)

    def wait_owner_source_reusable(self) -> None:
        if not self.owner_source_reusable:
            missing = self.reader_ranks.difference(self._reader_acks)
            raise RuntimeError(f"Owner KV source still has readers: {sorted(missing)}")

    def release_after_last_cache_use(self, event: Any) -> None:
        if self.local_cache_ready is None or not self.owner_source_reusable:
            raise RuntimeError("Cannot release a transfer before receive/source completion")
        if self.last_cache_use is not None:
            raise RuntimeError("Last cache use was already recorded")
        self.last_cache_use = event

    def drain(self) -> None:
        if self.last_cache_use is None:
            raise RuntimeError("Cannot reclaim KV resources before the last cache reader")
        self.local_cache_ready.synchronize()
        self.last_cache_use.synchronize()
        self.retained_resources = ()
        self._released = True
