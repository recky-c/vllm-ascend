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
            for cache_layer_name in sorted(
                layer_owners,
                key=lambda name: (extract_layer_index(name), name),
            ):
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


def get_kvpp_managed_group_index(
    kv_cache_groups: list[Any],
    layer_owners: dict[str, int],
) -> int:
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

    managed_group_indices = {
        index
        for index, group in enumerate(kv_cache_groups)
        if managed_layers.intersection(group.layer_names)
    }
    if len(managed_group_indices) != 1:
        raise ValueError(
            "KVPP currently requires all managed Target KV layers to use "
            "one KV cache group, but found managed groups "
            f"{sorted(managed_group_indices)}."
        )
    return managed_group_indices.pop()


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


@dataclass
class KVPPGraphLayerEvents:
    """Per-layer device-side events for the Barrier Bridge graph path.

    The capture stream records ``push_start`` after the READY barrier and
    ``recv_start`` after the DONE barrier; the comm stream waits on them before
    launching the MTE push/receive. The comm stream records ``push_done`` /
    ``recv_done`` when the MTE transfer completes; the capture stream waits on
    them before the DONE barrier (push_done) and before attention (recv_done).

    This establishes the cross-rank happens-before chain required for
    correctness::

        OWNER PUSH -> push_done -> capture wait -> DONE BARRIER
                    -> recv_start -> NON-OWNER RECEIVE
    """

    push_start: Any
    push_done: Any
    recv_start: Any
    recv_done: Any


@dataclass
class KVPPGraphResources:
    """Per-graph-capacity event bundle (doc section 22).

    Each captured graph (one per request capacity) owns an independent set of
    layer events so alternating replays of different-capacity graphs do not
    share state.
    """

    layer_events: dict[str, KVPPGraphLayerEvents]
    retained_completions: list[Any]


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
        self._active_graph_resources: KVPPGraphResources | None = None

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        self._validate_plan_across_ranks()
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

    def _validate_plan_across_ranks(self) -> None:
        """Fail before transport setup if ranks disagree on bundle layout."""
        if self.group.world_size <= 1:
            return
        signature = tuple(
            (
                layer_name,
                self.layer_owners[layer_name],
                self.plan.cache_bundles[layer_name],
            )
            for layer_name in self.plan.layers
        )
        peer_signatures: list[Any] = [None] * self.group.world_size
        dist.all_gather_object(
            peer_signatures,
            signature,
            group=self.group.cpu_group,
        )
        mismatched_ranks = [
            rank
            for rank, peer_signature in enumerate(peer_signatures)
            if peer_signature != signature
        ]
        if mismatched_ranks:
            raise RuntimeError(
                "KVPP execution plans differ across ranks; refusing to start "
                "with incompatible cache-bundle layouts from ranks "
                f"{mismatched_ranks}."
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

    def create_graph_resources(self) -> KVPPGraphResources:
        """Allocate per-layer events for one graph capture (doc section 33).

        Events are created before capture begins and reused across replays of
        the same graph. Different graph capacities get independent resources.
        """
        layer_events: dict[str, KVPPGraphLayerEvents] = {}
        for layer_name in self.plan.layers:
            layer_events[layer_name] = KVPPGraphLayerEvents(
                push_start=torch.npu.Event(),
                push_done=torch.npu.Event(),
                recv_start=torch.npu.Event(),
                recv_done=torch.npu.Event(),
            )
        return KVPPGraphResources(
            layer_events=layer_events,
            retained_completions=[],
        )

    def set_active_graph_resources(
        self, resources: KVPPGraphResources
    ) -> None:
        """Bind the resources for the graph currently being captured/replayed."""
        self._active_graph_resources = resources

    def clear_active_graph_resources(self) -> None:
        """Unbind graph resources after capture/replay ends."""
        self._active_graph_resources = None

    def _require_active_graph_resources(self) -> KVPPGraphResources:
        if self._active_graph_resources is None:
            raise RuntimeError(
                "KVPP graph resources were not set before graph capture/replay."
            )
        return self._active_graph_resources

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

        if torch.npu.is_current_stream_capturing():
            # Graph path: _start_prefetch only started the push. Finish the
            # push, run the DONE barrier, and start the receive here so the
            # receive overlaps with this layer's Q/KV projection.
            pages = self._selected_pages
            assert pages is not None
            self._graph_finish_push_start_receive(layer_name, pages)

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

        if torch.npu.is_current_stream_capturing():
            # Graph path: wait for the receive to complete on the comm stream
            # before attention reads the KV. The push for this layer was
            # started in the previous layer's wait_for_layer (or in
            # _start_prefetch for layer 0), and the receive was started in
            # enter_layer. Now we wait for recv_done.
            self._graph_wait_receive(layer_name)
        elif self._transfer_future is not None:
            # Eager path: this blocks only for the residual transfer time
            # because this layer was prefetched while the preceding layer
            # was executing.
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

        Barrier Bridge graph path: no extra synchronization is needed here.
        Scratch reuse safety is guaranteed naturally because receive(Layer i+2)
        only starts at enter_layer(i+2), by which time Layer i's attention has
        long finished (doc section 17). The next layer's PUSH only writes
        remote staging, not local scratch (doc section 17).
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
            # Graph path: only start the push. The push is launched on the
            # comm stream and overlaps with the previous layer's attention/
            # o_proj/MLP. The receive is started later in enter_layer.
            pages = self._selected_pages
            assert pages is not None
            self._graph_start_push(layer_name, pages)
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

    def _graph_start_push(
        self,
        layer_name: str,
        pages: KVPPActivePages,
    ) -> None:
        """Phase 1 of Barrier Bridge: READY barrier + launch push on comm stream.

        Collective stays on the capture stream. The MTE push runs on the
        independent KVPP comm stream so it overlaps with the previous layer's
        attention/o_proj/MLP compute. This function returns immediately without
        waiting for the push to complete — that wait happens in
        ``_graph_finish_push_start_receive``.
        """
        resources = self._require_active_graph_resources()
        events = resources.layer_events[layer_name]
        if self._graph_sync_token is None or self.group.device_group is None:
            raise RuntimeError("KVPP graph synchronization was not initialized.")
        if self._comm_stream is None:
            raise RuntimeError("KVPP comm stream was not initialized.")

        capture_stream = torch.npu.current_stream()

        # READY barrier on capture stream: ensures every non-owner is ready
        # to receive before the owner pushes.
        dist.all_reduce(
            self._graph_sync_token, group=self.group.device_group
        )

        # Signal comm stream that the ready barrier completed.
        events.push_start.record(capture_stream)

        # MTE push on comm stream (overlaps with capture stream compute).
        with torch.npu.stream(self._comm_stream):
            self._comm_stream.wait_event(events.push_start)
            completion = self.transport.push_active_bundle(
                self.plan.cache_bundles[layer_name], pages, self._comm_stream
            )
            events.push_done.record(self._comm_stream)

        resources.retained_completions.append(completion)

    def _graph_finish_push_start_receive(
        self,
        layer_name: str,
        pages: KVPPActivePages,
    ) -> None:
        """Phase 2: wait push_done, DONE barrier, launch receive on comm stream.

        Core happens-before chain (doc section 26)::

            OWNER PUSH -> push_done -> capture wait -> DONE BARRIER
                        -> recv_start -> NON-OWNER RECEIVE

        The receive runs on the comm stream and overlaps with this layer's
        Q/KV projection compute on the capture stream.
        """
        resources = self._require_active_graph_resources()
        events = resources.layer_events[layer_name]
        capture_stream = torch.npu.current_stream()

        # Local PUSH must complete before this rank enters the cross-rank
        # DONE barrier so the push is remotely visible.
        capture_stream.wait_event(events.push_done)

        # DONE barrier on capture stream.
        dist.all_reduce(
            self._graph_sync_token, group=self.group.device_group
        )

        # Signal comm stream that the done barrier completed.
        events.recv_start.record(capture_stream)

        # MTE receive on comm stream (overlaps with capture stream QKV).
        with torch.npu.stream(self._comm_stream):
            self._comm_stream.wait_event(events.recv_start)
            completion = self.transport.receive_active_bundle(
                self.plan.cache_bundles[layer_name], pages, self._comm_stream
            )
            events.recv_done.record(self._comm_stream)

        resources.retained_completions.append(completion)

    def _graph_wait_receive(self, layer_name: str) -> None:
        """Phase 3: wait for the receive to complete before attention."""
        resources = self._require_active_graph_resources()
        events = resources.layer_events[layer_name]
        torch.npu.current_stream().wait_event(events.recv_done)

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


class KVPPGraphCaptureController:
    """Own the graph-only page inputs and scheduler lifecycle."""

    def __init__(self, scheduler: KVPPScheduler, cache_group_index: int) -> None:
        self.scheduler = scheduler
        self.cache_group_index = cache_group_index
        self._active = False
        self._staged_inputs: KVPPActivePages | None = None
        self._staged_graph_num_requests: int | None = None
        self._capture_pages: dict[int, KVPPActivePages] = {}
        self._graph_resources: dict[int, KVPPGraphResources] = {}

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
        # Create per-capacity graph event resources if not yet allocated.
        if num_requests not in self._graph_resources:
            self._graph_resources[num_requests] = (
                self.scheduler.create_graph_resources()
            )
        self._staged_graph_num_requests = num_requests

    def begin(self) -> None:
        if self._staged_inputs is None:
            return
        pages = self._staged_inputs
        self._staged_inputs = None
        graph_num_requests = self._staged_graph_num_requests
        self._staged_graph_num_requests = None
        if graph_num_requests is None:
            raise RuntimeError(
                "KVPP graph capture began without a staged graph capacity."
            )
        resources = self._graph_resources.get(graph_num_requests)
        if resources is None:
            raise RuntimeError(
                "KVPP graph resources were not allocated for batch size "
                f"{graph_num_requests}."
            )
        self.scheduler.set_active_graph_resources(resources)
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
            self.scheduler.clear_active_graph_resources()
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
        resources = self._graph_resources.get(graph_num_requests)
        if resources is None:
            raise RuntimeError(
                "KVPP graph resources were not allocated for batch size "
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
        self.scheduler.set_active_graph_resources(resources)
