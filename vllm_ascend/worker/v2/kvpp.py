from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    MemFabricMTEKVPPTransport,
)


@dataclass(frozen=True)
class KVPPExecutionPlan:
    """Pure layer-to-cache execution topology, independent of transport."""

    cache_bundles: dict[str, tuple[str, ...]]

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.cache_bundles)

    @classmethod
    def build(
        cls,
        layer_owners: dict[str, int],
        execution_layers: tuple[str, ...] | None,
    ) -> "KVPPExecutionPlan":
        layers = tuple(execution_layers or layer_owners)
        if not layers:
            raise ValueError("KVPP requires at least one executable attention layer.")

        if execution_layers is None:
            cache_bundles = {layer_name: (layer_name,) for layer_name in layers}
        else:
            cache_layers_by_index: dict[int, list[str]] = {}
            for cache_layer_name in layer_owners:
                cache_layers_by_index.setdefault(
                    extract_layer_index(cache_layer_name), []
                ).append(cache_layer_name)

            cache_bundles: dict[str, tuple[str, ...]] = {}
            claimed_indices: set[int] = set()
            for layer_name in layers:
                if layer_name not in layer_owners:
                    raise ValueError(
                        f"KVPP execution layer {layer_name} has no KV cache owner."
                    )
                layer_index = extract_layer_index(layer_name)
                if layer_index in claimed_indices:
                    raise ValueError(
                        "KVPP received multiple executable attention layers for "
                        f"transformer layer {layer_index}."
                    )
                claimed_indices.add(layer_index)
                cache_bundles[layer_name] = tuple(cache_layers_by_index[layer_index])

            unclaimed = {
                cache_layer_name
                for layer_index, cache_layer_names in cache_layers_by_index.items()
                if layer_index not in claimed_indices
                for cache_layer_name in cache_layer_names
            }
            if unclaimed:
                raise ValueError(
                    "KVPP cache layers have no executable attention owner: "
                    f"{sorted(unclaimed)}."
                )

        for layer_name, cache_layer_names in cache_bundles.items():
            bundle_owners = {layer_owners[name] for name in cache_layer_names}
            if len(bundle_owners) != 1:
                raise ValueError(
                    f"KVPP cache bundle for {layer_name} spans owners "
                    f"{sorted(bundle_owners)}."
                )
        return cls(cache_bundles=cache_bundles)


def get_kvpp_managed_group_indices(
    kv_cache_groups: list[Any],
    layer_owners: dict[str, int],
) -> tuple[int, int | None]:
    """Return (target_group_index, draft_group_index) for KVPP-managed groups.

    Target KVPP layers must all live in one group. Draft (MTP/Eagle) layers
    may live in a separate ``is_eagle_group`` group, or absent when no
    speculator is configured. Each managed layer must belong to exactly one
    of these two groups.
    """
    managed_layers = set(layer_owners)
    configured_layers = {
        layer_name
        for group in kv_cache_groups
        for layer_name in group.layer_names
    }
    missing_layers = managed_layers - configured_layers
    if missing_layers:
        raise ValueError(
            "KVPP-owned cache layers are missing from KV cache groups: "
            f"{sorted(missing_layers)}."
        )

    target_indices: set[int] = set()
    draft_indices: set[int] = set()
    for index, group in enumerate(kv_cache_groups):
        if not managed_layers.intersection(group.layer_names):
            continue
        if getattr(group, "is_eagle_group", False):
            draft_indices.add(index)
        else:
            target_indices.add(index)

    if len(target_indices) != 1:
        raise ValueError(
            "KVPP currently requires all managed Target KV layers to use "
            "one KV cache group, but found managed Target groups "
            f"{sorted(target_indices)}."
        )
    if len(draft_indices) > 1:
        raise ValueError(
            "KVPP found multiple managed draft KV cache groups: "
            f"{sorted(draft_indices)}."
        )
    return target_indices.pop(), (draft_indices.pop() if draft_indices else None)


def get_kvpp_managed_group_index(
    kv_cache_groups: list[Any],
    layer_owners: dict[str, int],
) -> int:
    """Legacy single-group accessor; see ``get_kvpp_managed_group_indices``."""
    target_index, _ = get_kvpp_managed_group_indices(kv_cache_groups, layer_owners)
    return target_index


def _active_pages(
    block_table: torch.Tensor,
    seq_lens: Any,
    block_size: int,
    num_blocks: int,
) -> KVPPActivePages:
    """Return fixed-shape device pages read by the current batch.

    The original block table is read only. Invalid columns and duplicate page
    IDs become masked slots instead of being compacted through the host.
    """
    if isinstance(seq_lens, torch.Tensor) and seq_lens.device.type != "cpu":
        raise ValueError(
            "KVPP sequence lengths must remain on the host so the active-page "
            "capacity bound never introduces a device-to-host synchronization."
        )
    table_columns = block_table.shape[1]
    host_lengths = torch.as_tensor(
        seq_lens, dtype=torch.int64, device="cpu"
    ).flatten()
    pages_per_request_host = torch.div(
        host_lengths + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).clamp_(min=0, max=table_columns)
    count_upper_bound = min(
        num_blocks, int(pages_per_request_host.sum().item())
    )
    lengths = host_lengths.to(device=block_table.device)
    table = block_table[: lengths.shape[0]].to(dtype=torch.int64)
    columns = torch.arange(
        table.shape[1], dtype=torch.int64, device=block_table.device
    )
    pages_per_request = torch.div(
        lengths + block_size - 1, block_size, rounding_mode="floor"
    )
    covered = columns.unsqueeze(0) < pages_per_request.unsqueeze(1)
    valid = covered & (table >= 0) & (table < num_blocks)
    sentinel = torch.full_like(table, num_blocks)
    sorted_pages = torch.sort(torch.where(valid, table, sentinel).flatten()).values
    unique = torch.ones_like(sorted_pages, dtype=torch.bool)
    if sorted_pages.numel() > 1:
        unique[1:] = sorted_pages[1:] != sorted_pages[:-1]
    valid_mask = unique & (sorted_pages < num_blocks)
    return KVPPActivePages(
        sorted_pages,
        valid_mask,
        count_upper_bound=count_upper_bound,
    )


class KVPPPhase(Enum):
    IDLE = auto()
    FORWARD_ACTIVE = auto()
    LAYER_ENTERED = auto()
    LAYER_WAITED = auto()


class KVPPScheduler:
    """Schedule stream-ordered layer prefetch over an injected transport.

    Owned layers use persistent KV caches. Non-owned layers are already bound
    by vLLM's planner to one of two alternating full-size scratch caches.
    Active pages are pushed into the same physical block IDs, preserving the
    original block table and slot mapping. The dual buffers let layer N+1 be
    filled while layer N attention still reads its own scratch cache.
    """

    def __init__(
        self,
        group: GroupCoordinator,
        layer_owners: dict[str, int],
        num_blocks: int,
        block_size: int,
        transport: MemFabricMTEKVPPTransport,
        execution_layers: tuple[str, ...] | None = None,
    ) -> None:
        self.group = group
        self.layer_owners = layer_owners
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.transport = transport
        self.plan = KVPPExecutionPlan.build(
            self.layer_owners, execution_layers
        )
        self._phase = KVPPPhase.IDLE
        self._next_layer_index = 0
        self._selected_pages: KVPPActivePages | None = None
        self._comm_stream: Any | None = None
        self._current_layer: str | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._transfer_future: Future[None] | None = None
        self._pending_layer: str | None = None
        self._device_id: int | None = None
        self._graph_sync_token: torch.Tensor | None = None

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        self.transport.initialize(kv_caches)
        if self.group.world_size > 1:
            self._device_id = torch.npu.current_device()
            self._comm_stream = torch.npu.Stream()
            self._graph_sync_token = torch.zeros(
                1, dtype=torch.int32, device="npu"
            )
            # One transfer may be in flight. Serializing jobs also preserves
            # point-to-point notification order when layer ownership changes.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kvpp-prefetch"
            )

    @property
    def selected_pages(self) -> KVPPActivePages | None:
        return self._selected_pages

    def begin_forward(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> None:
        active_pages = _active_pages(
            block_table, seq_lens, self.block_size, self.num_blocks
        )
        self._begin_forward(active_pages)

    def _begin_forward(self, active_pages: KVPPActivePages) -> None:
        if self._phase is not KVPPPhase.IDLE:
            raise RuntimeError(
                f"KVPP cannot begin a forward while phase is {self._phase.name}."
            )
        if self._current_layer is not None or self._pending_layer is not None:
            raise RuntimeError("KVPP cannot start with layer work pending.")
        self._selected_pages = active_pages
        self._next_layer_index = 0
        self._phase = KVPPPhase.FORWARD_ACTIVE

    def finish_forward(self) -> None:
        if self._phase is KVPPPhase.IDLE:
            return
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP forward ended while layer {self._current_layer} was active."
            )
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP forward ended with transfer for {self._pending_layer} pending."
            )
        if self._next_layer_index != len(self.plan.layers):
            raise RuntimeError(
                "KVPP forward executed "
                f"{self._next_layer_index}/{len(self.plan.layers)} layers."
            )
        self._selected_pages = None
        self._next_layer_index = 0
        self._phase = KVPPPhase.IDLE

    def abort_batch(self) -> None:
        """Drain in-flight work and reset state after a failed/short forward."""
        if self._transfer_future is not None:
            with suppress(BaseException):
                self._transfer_future.result()
        self._transfer_future = None
        self._pending_layer = None
        self._current_layer = None
        self._next_layer_index = 0
        self._selected_pages = None
        self._phase = KVPPPhase.IDLE

    def enter_layer(self, layer_name: str) -> None:
        """Start owner page pushes while Q/KV projection runs on compute."""
        if self._selected_pages is None:
            raise RuntimeError("KVPP batch metadata was not prepared before forward.")
        if self._phase is not KVPPPhase.FORWARD_ACTIVE:
            raise RuntimeError("KVPP forward was not started before entering a layer.")
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP layer {self._current_layer} was not completed before {layer_name}."
            )
        if self._next_layer_index >= len(self.plan.layers):
            raise RuntimeError(
                f"KVPP forward entered unexpected extra layer {layer_name}."
            )
        expected_layer = self.plan.layers[self._next_layer_index]
        if layer_name != expected_layer:
            raise RuntimeError(
                f"KVPP forward expected {expected_layer}, "
                f"but entered {layer_name}."
            )
        self._next_layer_index += 1

        self._current_layer = layer_name
        self._phase = KVPPPhase.LAYER_ENTERED
        if self._pending_layer is None:
            self._start_prefetch(layer_name)
        elif self._pending_layer != layer_name:
            raise RuntimeError(
                f"KVPP prefetched {self._pending_layer}, but forward entered "
                f"{layer_name}."
            )

    def wait_for_layer(self, layer_name: str) -> None:
        """Order cache use, then prefetch the next layer before attention."""
        if (
            self._phase is not KVPPPhase.LAYER_ENTERED
            or self._current_layer != layer_name
        ):
            raise RuntimeError(f"No pending KVPP transfer for layer {layer_name}.")
        if self._pending_layer != layer_name:
            raise RuntimeError(
                f"No pending KVPP prefetch for layer {layer_name}."
            )
        if self._transfer_future is not None:
            # This blocks only for the residual transfer time because this
            # layer was prefetched while the preceding layer was executing.
            layer_index = extract_layer_index(layer_name)
            with torch.profiler.record_function(
                f"kvpp.wait.previous_layer.layer_{layer_index}"
            ):
                self._transfer_future.result()
        self._phase = KVPPPhase.LAYER_WAITED
        # The current transfer is remotely visible now. Its buffer is read by
        # the attention about to be submitted, while the other scratch buffer
        # receives the next layer immediately.
        self._pending_layer = None
        self._transfer_future = None
        if self._next_layer_index < len(self.plan.layers):
            self._start_prefetch(self.plan.layers[self._next_layer_index])

    def leave_layer(self, layer_name: str) -> None:
        """Mark the current layer's attention submission complete.

        The call site is after all attention kernels that consume historical
        KV have been submitted, but before o_proj and the layer MLP/MoE. With
        dual scratch buffers the next transfer was already submitted just
        before this attention. Compute-stream ordering protects reuse when the
        buffer cycles back two layers later.
        """
        if (
            self._phase is not KVPPPhase.LAYER_WAITED
            or self._current_layer != layer_name
        ):
            raise RuntimeError(
                f"KVPP attention for layer {layer_name} finished before its "
                "transfer was consumed."
            )

        self._current_layer = None
        self._phase = KVPPPhase.FORWARD_ACTIVE

    def _start_prefetch(self, layer_name: str) -> None:
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP transfer for {self._pending_layer} is still pending."
            )
        self._pending_layer = layer_name
        self._transfer_future = None
        if self.group.world_size <= 1:
            return
        if self._executor is None or self._comm_stream is None:
            raise RuntimeError("KVPP prefetch worker was not initialized.")

        if torch.npu.is_current_stream_capturing():
            self._run_graph_prefetch(layer_name, self._selected_pages)
            return

        # All ranks publish a local safe point. Alternating layers use distinct
        # buffers. When a buffer cycles back after two layers, this event is
        # ordered after all earlier attention work on the compute stream, so
        # the owner cannot overwrite a buffer that is still being read.
        scratch_ready = torch.npu.Event()
        scratch_ready.record(torch.npu.current_stream())
        pages = self._selected_pages
        assert pages is not None
        self._transfer_future = self._executor.submit(
            self._run_prefetch, layer_name, pages, scratch_ready
        )

    def _run_graph_prefetch(
        self,
        layer_name: str,
        pages: KVPPActivePages | None,
    ) -> None:
        """Capture a host-event-free transfer sequence for FULL graph replay.

        The two device collectives replace eager mode's CPU ready/done
        protocol.  Keeping all operations on the capture stream is deliberate:
        it provides correct cross-rank ordering without recording an NPU event
        that a background Python thread would later synchronize.
        """
        if pages is None:
            raise RuntimeError("KVPP graph capture has no active-page metadata.")
        if self._graph_sync_token is None or self.group.device_group is None:
            raise RuntimeError("KVPP graph synchronization was not initialized.")

        stream = torch.npu.current_stream()
        owner_rank = self.layer_owners[layer_name]
        local_rank = self.group.rank_in_group
        dist.all_reduce(self._graph_sync_token, group=self.group.device_group)
        if local_rank == owner_rank:
            self.transport.push_active_bundle(
                self.plan.cache_bundles[layer_name], pages, stream
            )
        dist.all_reduce(self._graph_sync_token, group=self.group.device_group)
        if local_rank != owner_rank:
            self.transport.receive_active_bundle(
                self.plan.cache_bundles[layer_name], pages, stream
            )

    def _run_prefetch(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        scratch_ready: Any,
    ) -> None:
        """Run safe-point and completion notification off the compute thread."""
        if self._comm_stream is None:
            raise RuntimeError("KVPP prefetch transport is not initialized.")
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)

        owner_rank = self.layer_owners[layer_name]
        local_rank = self.group.rank_in_group
        owner_global_rank = self.group.ranks[owner_rank]
        layer_index = extract_layer_index(layer_name)
        ready_tag = 0x4B560000 + layer_index * 2
        done_tag = ready_tag + 1
        token = torch.ones(1, dtype=torch.uint8, device="cpu")

        with torch.profiler.record_function(
            f"kvpp.comm_total.layer_{layer_index}"
        ):
            scratch_ready.synchronize()

            if local_rank != owner_rank:
                dist.send(
                    token,
                    dst=owner_global_rank,
                    group=self.group.cpu_group,
                    tag=ready_tag,
                )
                dist.recv(
                    token,
                    src=owner_global_rank,
                    group=self.group.cpu_group,
                    tag=done_tag,
                )
                with torch.profiler.record_function(
                    f"kvpp.transport_receive.layer_{layer_index}"
                ):
                    with torch.npu.stream(self._comm_stream):
                        receive_completion = (
                            self.transport.receive_active_bundle(
                                self.plan.cache_bundles[layer_name],
                                pages,
                                self._comm_stream,
                            )
                        )
                    receive_completion.wait()
                return

            for peer_rank, peer_global_rank in enumerate(self.group.ranks):
                if peer_rank == owner_rank:
                    continue
                dist.recv(
                    token,
                    src=peer_global_rank,
                    group=self.group.cpu_group,
                    tag=ready_tag,
                )

            with torch.profiler.record_function(
                f"kvpp.transport_push.layer_{layer_index}"
            ):
                with torch.npu.stream(self._comm_stream):
                    completion = self.transport.push_active_bundle(
                        self.plan.cache_bundles[layer_name],
                        pages,
                        self._comm_stream,
                    )
                # Only the communication worker waits on the host. The compute
                # thread continues until this layer first writes/reads its
                # paged KV cache.
                completion.wait()

            for peer_rank, peer_global_rank in enumerate(self.group.ranks):
                if peer_rank == owner_rank:
                    continue
                dist.send(
                    token,
                    dst=peer_global_rank,
                    group=self.group.cpu_group,
                    tag=done_tag,
                )

    def close(self) -> None:
        """Drain overlap work and release transport-owned resources."""
        self.abort_batch()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.transport.close()


class KVPPDraftPhase(Enum):
    IDLE = auto()
    BATCH_ACTIVE = auto()


class KVPPDraftController:
    """Batch-level KV refresh for MTP/Eagle draft layers under KVPP.

    Unlike :class:`KVPPScheduler` (Target layers, per-layer streaming with
    dual rotating scratch), the draft controller refreshes every non-owner
    draft KV cache **once per ``propose()``**, synchronously, before the
    first MTP attention. The same scratch is then read and written by the
    prefill and every speculative step; no further transfer happens until
    the next batch. Owner ranks write new KV directly to persistent cache;
    non-owner ranks write to their per-layer dedicated scratch. Nothing is
    written back from scratch to owner at batch end.
    """

    def __init__(
        self,
        group: GroupCoordinator,
        layer_owners: dict[str, int],
        draft_layer_names: tuple[str, ...],
        num_blocks: int,
        block_size: int,
        transport: MemFabricMTEKVPPTransport,
    ) -> None:
        if not draft_layer_names:
            raise ValueError("KVPPDraftController requires at least one draft layer.")
        self.group = group
        self.layer_owners = layer_owners
        self.draft_layer_names = draft_layer_names
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.transport = transport
        self._phase = KVPPDraftPhase.IDLE
        self._active_pages: KVPPActivePages | None = None
        self._graph_sync_token: torch.Tensor | None = None

    def initialize(self) -> None:
        if self.group.world_size > 1:
            self._graph_sync_token = torch.zeros(
                1, dtype=torch.int32, device="npu"
            )

    def begin_batch(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> None:
        """Refresh every non-owner draft scratch once for this batch.

        Synchronous: returns only after owner pushes are remotely visible on
        the current stream. Owners and single-rank groups no-op.
        """
        if self._phase is not KVPPDraftPhase.IDLE:
            raise RuntimeError(
                f"KVPP draft batch already active (phase={self._phase.name})."
            )
        if self.group.world_size <= 1:
            self._phase = KVPPDraftPhase.BATCH_ACTIVE
            return

        pages = _active_pages(
            block_table, seq_lens, self.block_size, self.num_blocks
        )
        self._active_pages = pages
        self._phase = KVPPDraftPhase.BATCH_ACTIVE

        if torch.npu.is_current_stream_capturing():
            self._run_graph_refresh(pages)
        else:
            self._run_eager_refresh(pages)

    def _run_eager_refresh(self, pages: KVPPActivePages) -> None:
        stream = torch.npu.current_stream()
        comm_stream = torch.npu.Stream()
        comm_stream.wait_stream(stream)
        local_rank = self.group.rank_in_group
        with torch.npu.stream(comm_stream):
            for layer_name in self.draft_layer_names:
                owner_rank = self.layer_owners[layer_name]
                bundle = (layer_name,)
                if local_rank == owner_rank:
                    completion = self.transport.push_active_bundle(
                        bundle, pages, comm_stream
                    )
                else:
                    completion = self.transport.receive_active_bundle(
                        bundle, pages, comm_stream
                    )
                completion.wait()
        stream.wait_stream(comm_stream)

    def _run_graph_refresh(self, pages: KVPPActivePages) -> None:
        if self._graph_sync_token is None or self.group.device_group is None:
            raise RuntimeError(
                "KVPP draft graph synchronization was not initialized."
            )
        stream = torch.npu.current_stream()
        local_rank = self.group.rank_in_group
        for layer_name in self.draft_layer_names:
            owner_rank = self.layer_owners[layer_name]
            bundle = (layer_name,)
            dist.all_reduce(
                self._graph_sync_token, group=self.group.device_group
            )
            if local_rank == owner_rank:
                self.transport.push_active_bundle(bundle, pages, stream)
            dist.all_reduce(
                self._graph_sync_token, group=self.group.device_group
            )
            if local_rank != owner_rank:
                self.transport.receive_active_bundle(bundle, pages, stream)

    def finish_batch(self) -> None:
        """End the draft batch. Scratch is discarded; nothing written back."""
        if self._phase is KVPPDraftPhase.IDLE:
            return
        self._active_pages = None
        self._phase = KVPPDraftPhase.IDLE

    def abort_batch(self) -> None:
        """Recover after a failed/aborted propose."""
        self._active_pages = None
        self._phase = KVPPDraftPhase.IDLE

    def close(self) -> None:
        """Release only local state. Transport is owned by the Target scheduler."""
        self.abort_batch()


class KVPPGraphCaptureController:
    """Own the graph-only page inputs and scheduler lifecycle."""

    def __init__(self, scheduler: KVPPScheduler, cache_group_index: int) -> None:
        self.scheduler = scheduler
        self.cache_group_index = cache_group_index
        self._active = False
        self._staged_inputs: KVPPActivePages | None = None
        self._capture_pages: dict[int, KVPPActivePages] = {}

    def stage(
        self,
        block_tables: tuple[torch.Tensor, ...],
        num_requests: int,
    ) -> None:
        if self._active:
            raise RuntimeError("A KVPP graph capture forward is already active.")
        block_table = block_tables[self.cache_group_index]
        descriptor_count = block_table.numel()
        pages = self._capture_pages.get(num_requests)
        if pages is not None:
            if pages.page_ids.numel() != descriptor_count:
                raise RuntimeError(
                    "KVPP graph capture changed descriptor capacity for batch "
                    f"size {num_requests}: captured={pages.page_ids.numel()}, "
                    f"new={descriptor_count}."
                )
        else:
            pages = KVPPActivePages(
                page_ids=torch.zeros(
                    descriptor_count,
                    dtype=torch.int64,
                    device=block_table.device,
                ),
                valid_mask=torch.zeros(
                    descriptor_count,
                    dtype=torch.bool,
                    device=block_table.device,
                ),
                count_upper_bound=min(
                    self.scheduler.num_blocks, descriptor_count
                ),
            )
            self._capture_pages[num_requests] = pages
        self._staged_inputs = pages

    def begin(self) -> None:
        if self._staged_inputs is None:
            return
        pages = self._staged_inputs
        self._staged_inputs = None
        self.scheduler._begin_forward(pages)
        self._active = True

    def finish(self, success: bool) -> None:
        if not self._active:
            return
        try:
            if success:
                self.scheduler.finish_forward()
            else:
                self.scheduler.abort_batch()
        finally:
            self._active = False

    def prepare_replay(self, graph_num_requests: int) -> None:
        runtime_pages = self.scheduler.selected_pages
        if runtime_pages is None:
            raise RuntimeError("KVPP graph replay has no runtime page metadata.")
        graph_pages = self._capture_pages.get(graph_num_requests)
        if graph_pages is None:
            raise RuntimeError(
                "KVPP graph inputs were not captured for batch size "
                f"{graph_num_requests}."
            )
        runtime_count = runtime_pages.page_ids.numel()
        graph_capacity = graph_pages.page_ids.numel()
        if runtime_count > graph_capacity:
            raise RuntimeError(
                "KVPP runtime page descriptors exceed the selected graph "
                f"capacity: runtime={runtime_count}, graph={graph_capacity}."
            )
        graph_pages.page_ids[:runtime_count].copy_(runtime_pages.page_ids)
        graph_pages.valid_mask.zero_()
        graph_pages.valid_mask[:runtime_count].copy_(runtime_pages.valid_mask)
        self.scheduler.abort_batch()
