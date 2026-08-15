from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPActivePages,
    KVPPTransport,
)


@dataclass(frozen=True)
class KVPPBatchPlan:
    """Immutable transport inputs for one model-runner batch."""

    epoch: int
    active_pages: KVPPActivePages
    num_requests: int


@dataclass(frozen=True)
class KVPPExecutionPlan:
    """Pure layer-to-cache execution topology, independent of transport."""

    layers: tuple[str, ...]
    cache_bundles: dict[str, tuple[str, ...]]
    owners: dict[str, int]
    indices: dict[str, int]

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

        owners: dict[str, int] = {}
        for layer_name, cache_layer_names in cache_bundles.items():
            bundle_owners = {layer_owners[name] for name in cache_layer_names}
            if len(bundle_owners) != 1:
                raise ValueError(
                    f"KVPP cache bundle for {layer_name} spans owners "
                    f"{sorted(bundle_owners)}."
                )
            owners[layer_name] = bundle_owners.pop()

        return cls(
            layers=layers,
            cache_bundles=cache_bundles,
            owners=owners,
            indices={layer_name: index for index, layer_name in enumerate(layers)},
        )


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


def select_kvpp_managed_caches(
    kv_caches: dict[str, Any],
    layer_owners: dict[str, int],
) -> dict[str, Any]:
    missing_layers = set(layer_owners) - set(kv_caches)
    if missing_layers:
        raise ValueError(
            "KVPP-owned cache tensors were not initialized: "
            f"{sorted(missing_layers)}."
        )
    return {
        layer_name: kv_caches[layer_name]
        for layer_name in layer_owners
    }


def _active_pages(
    block_table: torch.Tensor,
    seq_lens: Any,
    block_size: int,
    num_blocks: int,
    *,
    for_graph_capture: bool = False,
) -> KVPPActivePages:
    """Return fixed-shape device pages read by the current batch.

    The original block table is read only. Invalid columns and duplicate page
    IDs become masked slots instead of being compacted through the host.
    """
    if (
        not for_graph_capture
        and isinstance(seq_lens, torch.Tensor)
        and seq_lens.device.type != "cpu"
    ):
        raise ValueError(
            "KVPP sequence lengths must remain on the host so the active-page "
            "capacity bound never introduces a device-to-host synchronization."
        )
    table_columns = block_table.shape[1]
    if for_graph_capture:
        # Keep lengths as a graph input.  The fixed capacity is conservative,
        # while valid_mask removes unused columns for each replay batch.
        lengths = torch.as_tensor(
            seq_lens, dtype=torch.int64, device=block_table.device
        ).flatten()
        count_upper_bound = min(num_blocks, block_table.numel())
    else:
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


@dataclass
class KVPPScheduler:
    """Schedule stream-ordered layer prefetch over an injected transport.

    Owned layers use persistent KV caches. Non-owned layers are already bound
    by vLLM's planner to one of two alternating full-size scratch caches.
    Active pages are pushed into the same physical block IDs, preserving the
    original block table and slot mapping. The dual buffers let layer N+1 be
    filled while layer N attention still reads its own scratch cache.
    """

    group: GroupCoordinator
    layer_owners: dict[str, int]
    num_blocks: int
    block_size: int
    transport: KVPPTransport
    execution_layers: tuple[str, ...] | None = None
    plan: KVPPExecutionPlan = field(init=False)
    _batch_plan: KVPPBatchPlan | None = field(default=None, init=False)
    _batch_epoch: int = field(default=0, init=False)
    _forward_sequence: int = field(default=0, init=False)
    _forward_id: int | None = field(default=None, init=False)
    _next_layer_index: int = field(default=0, init=False)
    _selected_pages: KVPPActivePages | None = None
    _comm_stream: Any | None = None
    _current_layer: str | None = None
    _executor: ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False
    )
    _transfer_future: Future[None] | None = field(
        default=None, init=False, repr=False
    )
    _pending_layer: str | None = field(default=None, init=False)
    _transfer_waited: bool = field(default=False, init=False)
    _device_id: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.plan = KVPPExecutionPlan.build(
            self.layer_owners, self.execution_layers
        )

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        self.transport.initialize(kv_caches)
        if self.group.world_size > 1:
            self._device_id = torch.npu.current_device()
            self._comm_stream = torch.npu.Stream()
            # One transfer may be in flight. Serializing jobs also preserves
            # point-to-point notification order when layer ownership changes.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kvpp-prefetch"
            )

    def prepare_batch(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> None:
        """Compatibility entry point for one-batch, one-forward execution."""
        self.begin_batch(block_table, seq_lens)
        self.begin_forward()

    def begin_batch(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> KVPPBatchPlan:
        if self._batch_plan is not None:
            raise RuntimeError(
                f"KVPP batch epoch {self._batch_plan.epoch} is still active."
            )
        if self._current_layer is not None or self._pending_layer is not None:
            raise RuntimeError("KVPP cannot start a batch with layer work pending.")

        active_pages = _active_pages(
            block_table, seq_lens, self.block_size, self.num_blocks
        )
        self._batch_epoch += 1
        self._batch_plan = KVPPBatchPlan(
            epoch=self._batch_epoch,
            active_pages=active_pages,
            num_requests=int(torch.as_tensor(seq_lens).numel()),
        )
        self._selected_pages = active_pages
        return self._batch_plan

    def begin_graph_capture(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> KVPPBatchPlan:
        """Begin a batch whose active-page selection is captured in the graph."""
        if self._batch_plan is not None:
            raise RuntimeError(
                f"KVPP batch epoch {self._batch_plan.epoch} is still active."
            )
        if self._current_layer is not None or self._pending_layer is not None:
            raise RuntimeError("KVPP cannot start a batch with layer work pending.")

        active_pages = _active_pages(
            block_table,
            seq_lens,
            self.block_size,
            self.num_blocks,
            for_graph_capture=True,
        )
        self._batch_epoch += 1
        self._batch_plan = KVPPBatchPlan(
            epoch=self._batch_epoch,
            active_pages=active_pages,
            num_requests=seq_lens.numel(),
        )
        self._selected_pages = active_pages
        return self._batch_plan

    def begin_forward(self) -> int:
        if self._batch_plan is None:
            raise RuntimeError("KVPP batch must begin before its forward pass.")
        if self._forward_id is not None:
            raise RuntimeError(f"KVPP forward {self._forward_id} is still active.")
        self._forward_sequence += 1
        self._forward_id = self._forward_sequence
        self._next_layer_index = 0
        return self._forward_id

    def finish_forward(self) -> None:
        if self._forward_id is None:
            return
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP forward {self._forward_id} ended while layer "
                f"{self._current_layer} was active."
            )
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP forward {self._forward_id} ended with transfer for "
                f"{self._pending_layer} pending."
            )
        if self._next_layer_index != len(self.plan.layers):
            raise RuntimeError(
                f"KVPP forward {self._forward_id} executed "
                f"{self._next_layer_index}/{len(self.plan.layers)} layers."
            )
        self._forward_id = None
        self._next_layer_index = 0

    def finish_batch(self) -> None:
        if self._forward_id is not None:
            raise RuntimeError(
                f"KVPP batch cannot finish while forward {self._forward_id} "
                "is active."
            )
        if self._current_layer is not None or self._pending_layer is not None:
            raise RuntimeError("KVPP batch cannot finish with layer work pending.")
        self._batch_plan = None
        self._selected_pages = None

    def abort_batch(self) -> None:
        """Drain in-flight work and reset state after a failed/short forward."""
        if self._transfer_future is not None:
            with suppress(BaseException):
                self._transfer_future.result()
        self._transfer_future = None
        self._pending_layer = None
        self._current_layer = None
        self._transfer_waited = False
        self._forward_id = None
        self._next_layer_index = 0
        self._batch_plan = None
        self._selected_pages = None

    def enter_layer(self, layer_name: str) -> None:
        """Start owner page pushes while Q/KV projection runs on compute."""
        if self._selected_pages is None:
            raise RuntimeError("KVPP batch metadata was not prepared before forward.")
        if self._forward_id is None:
            raise RuntimeError("KVPP forward was not started before entering a layer.")
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP layer {self._current_layer} was not completed before {layer_name}."
            )
        if self._next_layer_index >= len(self.plan.layers):
            raise RuntimeError(
                f"KVPP forward {self._forward_id} entered unexpected extra layer "
                f"{layer_name}."
            )
        expected_layer = self.plan.layers[self._next_layer_index]
        if layer_name != expected_layer:
            raise RuntimeError(
                f"KVPP forward {self._forward_id} expected {expected_layer}, "
                f"but entered {layer_name}."
            )
        self._next_layer_index += 1

        self._current_layer = layer_name
        self._transfer_waited = False
        if self._pending_layer is None:
            self._start_prefetch(layer_name)
        elif self._pending_layer != layer_name:
            raise RuntimeError(
                f"KVPP prefetched {self._pending_layer}, but forward entered "
                f"{layer_name}."
            )

    def wait_for_layer(self, layer_name: str) -> None:
        """Order cache use, then prefetch the next layer before attention."""
        if self._current_layer != layer_name:
            raise RuntimeError(f"No pending KVPP transfer for layer {layer_name}.")
        if self._pending_layer != layer_name:
            raise RuntimeError(
                f"No pending KVPP prefetch for layer {layer_name}."
            )
        if self._transfer_future is not None:
            # This blocks only for the residual transfer time because this
            # layer was prefetched while the preceding layer was executing.
            layer_index = self.plan.indices[layer_name]
            with torch.profiler.record_function(
                f"kvpp.wait.previous_layer.layer_{layer_index}"
            ):
                self._transfer_future.result()
        self._transfer_waited = True
        # The current transfer is remotely visible now. Its buffer is read by
        # the attention about to be submitted, while the other scratch buffer
        # receives the next layer immediately.
        self._pending_layer = None
        self._transfer_future = None
        layer_index = self.plan.indices[layer_name]
        next_index = layer_index + 1
        if next_index < len(self.plan.layers):
            self._start_prefetch(self.plan.layers[next_index])

    def leave_layer(self, layer_name: str) -> None:
        """Mark the current layer's attention submission complete.

        The call site is after all attention kernels that consume historical
        KV have been submitted, but before o_proj and the layer MLP/MoE. With
        dual scratch buffers the next transfer was already submitted just
        before this attention. Compute-stream ordering protects reuse when the
        buffer cycles back two layers later.
        """
        if self._current_layer != layer_name or not self._transfer_waited:
            raise RuntimeError(
                f"KVPP attention for layer {layer_name} finished before its "
                "transfer was consumed."
            )

        self._current_layer = None
        self._transfer_waited = False

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

        owner_rank = self.plan.owners[layer_name]
        local_rank = self.group.rank_in_group
        owner_global_rank = self.group.ranks[owner_rank]
        layer_index = self.plan.indices[layer_name]
        ready_tag = 0x4B560000 + layer_index * 2
        done_tag = ready_tag + 1
        token = torch.ones(1, dtype=torch.uint8, device="cpu")

        with torch.profiler.record_function(
            f"kvpp.comm_total.layer_{layer_index}"
        ):
            with torch.profiler.record_function(
                f"kvpp.scratch_ready.layer_{layer_index}"
            ):
                scratch_ready.synchronize()

            if local_rank != owner_rank:
                with torch.profiler.record_function(
                    f"kvpp.ready_send.layer_{layer_index}"
                ):
                    dist.send(
                        token,
                        dst=owner_global_rank,
                        group=self.group.cpu_group,
                        tag=ready_tag,
                    )
                with torch.profiler.record_function(
                    f"kvpp.done_recv.layer_{layer_index}"
                ):
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

            with torch.profiler.record_function(
                f"kvpp.ready_recv.layer_{layer_index}"
            ):
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

            with torch.profiler.record_function(
                f"kvpp.done_send.layer_{layer_index}"
            ):
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
