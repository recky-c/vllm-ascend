# SPDX-License-Identifier: Apache-2.0
"""KVPP runtime: layer ownership and stream-ordered active-page transport.

This module owns the *usage* side of KV layer parallelism: per-batch active
page selection, the per-layer state machine that overlaps transfer with
attention, and the boundary to the transport backend.

Allocation (who owns which layer, how scratch is sized) is handled in
``vllm_ascend.core.kvpp_allocation`` and the vLLM allocation hook. By the
time ``KVPPContext`` is constructed, every non-owned layer's
``forward_context[layer_name].kv_cache`` is already bound by vLLM's
standard ``bind_kv_cache`` to one of two alternating full-size scratch
tensors (expanded via ``KVCacheTensor.shared_by``). Active pages are
pushed into the same physical block IDs, preserving the original block
table and slot mapping. The dual buffers let layer N+1 be filled while
layer N attention still reads its own scratch cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, get_kvpp_group
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.core.kvpp_allocation import KVPP_REPLICATED_OWNER
from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPActivePages,
    KVPPCompletion,
    KVPPTransport,
    NullCompletion,
    NullTransport,
)


def _active_pages(
    block_table: torch.Tensor,
    seq_lens: Any,
    block_size: int,
    num_blocks: int,
) -> KVPPActivePages:
    """Return compacted device pages read by the current batch.

    The original block table is read only. Pages are sorted, deduplicated, and
    cropped to ``count_upper_bound`` so the MTE kernel only iterates over the
    slots that can actually be active. ``count_upper_bound`` is derived only
    from the host-resident sequence lengths and bounds the number of valid,
    unique pages without a device reduction.
    """
    if isinstance(seq_lens, torch.Tensor) and seq_lens.device.type != "cpu":
        raise ValueError(
            "KVPP sequence lengths must remain on the host so the active-page "
            "capacity bound never introduces a device-to-host synchronization."
        )
    host_lengths = torch.as_tensor(
        seq_lens, dtype=torch.int64, device="cpu"
    ).flatten()
    table_columns = block_table.shape[1]
    pages_per_request_host = torch.div(
        host_lengths + block_size - 1, block_size, rounding_mode="floor"
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
    if count_upper_bound == 0:
        page_ids = sorted_pages[:0]
        mask = valid_mask[:0]
    else:
        page_ids = sorted_pages[:count_upper_bound]
        mask = valid_mask[:count_upper_bound]
    return KVPPActivePages(
        page_ids,
        mask,
        count_upper_bound=count_upper_bound,
    )


@dataclass
class KVPPContext:
    """Layer ownership and stream-ordered active-page transport.

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
    execution_layers: tuple[str, ...] | None = None
    transport: KVPPTransport | None = None
    _selected_pages: KVPPActivePages | None = None
    _comm_stream: Any | None = None
    _current_layer: str | None = None
    _ordered_layers: tuple[str, ...] = field(init=False)
    _layer_indices: dict[str, int] = field(init=False)
    _layer_bundles: dict[str, tuple[str, ...]] = field(init=False)
    _execution_owners: dict[str, int] = field(init=False)
    _transfer_waited: bool = field(default=False, init=False)
    # previous_layer overlap state. A transfer for the next layer is submitted
    # while the current layer's attention is still running, so it must be
    # tracked separately from the current layer.
    _pending_layer: str | None = field(default=None, init=False)
    _transfer_future: Any = field(default=None, init=False)
    _executor: Any = field(default=None, init=False)
    _device_id: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = NullTransport()

        if self.execution_layers is None:
            self._ordered_layers = tuple(self.layer_owners)
            self._layer_bundles = {
                layer_name: (layer_name,) for layer_name in self._ordered_layers
            }
        else:
            self._ordered_layers = tuple(self.execution_layers)
            if not self._ordered_layers:
                raise ValueError("KVPP requires at least one executable attention layer.")
            cache_layers_by_index: dict[int, list[str]] = {}
            for cache_layer_name in self.layer_owners:
                cache_layers_by_index.setdefault(
                    extract_layer_index(cache_layer_name), []
                ).append(cache_layer_name)
            self._layer_bundles = {}
            claimed_indices: set[int] = set()
            for layer_name in self._ordered_layers:
                if layer_name not in self.layer_owners:
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
                self._layer_bundles[layer_name] = tuple(
                    cache_layers_by_index[layer_index]
                )
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

        self._execution_owners = {}
        for layer_name, cache_layer_names in self._layer_bundles.items():
            owners = {self.layer_owners[name] for name in cache_layer_names}
            if len(owners) != 1:
                raise ValueError(
                    f"KVPP cache bundle for {layer_name} spans owners {sorted(owners)}."
                )
            self._execution_owners[layer_name] = owners.pop()
        self._layer_indices = {
            layer_name: index
            for index, layer_name in enumerate(self._ordered_layers)
        }

    @property
    def _replicated_layer_count(self) -> int:
        """Number of layers replicated on every rank (skip transport)."""
        return sum(
            1 for owner in self._execution_owners.values()
            if owner == KVPP_REPLICATED_OWNER
        )

    def _is_replicated(self, layer_name: str) -> bool:
        """Whether ``layer_name`` is replicated on every KVPP rank."""
        return self._execution_owners.get(layer_name) == KVPP_REPLICATED_OWNER

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        if self.transport is None:
            self.transport = NullTransport()
        self.transport.initialize(kv_caches)
        if self.group.world_size > 1:
            self._device_id = torch.npu.current_device()
            self._comm_stream = torch.npu.Stream()
            # One transfer may be in flight at a time. Serializing jobs
            # preserves point-to-point notification order between layers.
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kvpp-transport"
            )

    def prepare_batch(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> None:
        self._selected_pages = _active_pages(
            block_table, seq_lens, self.block_size, self.num_blocks
        )
        # Notify the transport of the compacted page set so it can cache any
        # page-dependent state (e.g. the int8 valid-mask view used by the
        # descriptorless MTE kernel) once per batch instead of per layer.
        if self.transport is not None:
            self.transport.prepare_batch(self._selected_pages)

    def begin_layer(self, layer_name: str) -> None:
        """Mark that forward has entered a new layer, and block until this
        layer's transfer is remotely visible.

        This is the only point where the compute thread waits for the
        transport. The wait happens before any KV use so the scratch buffer
        is guaranteed safe to read. Two cases:

        * ``_pending_layer == layer_name``: the transfer was prefetched by
          the previous layer's ``prepare_for_attention``. Wait for it.
        * ``_pending_layer is None``: no prefetch was submitted (first layer,
          or a path that skipped it). Start it now via the fallback and wait.

        Replicated layers (owner == ``KVPP_REPLICATED_OWNER``, e.g. layer 0)
        skip the wait entirely: every rank holds a persistent copy, so there
        is no transfer to wait for. ``_transfer_waited`` is still set so the
        later ``prepare_for_attention`` / ``finish_layer_attention`` guards
        pass.

        After the wait returns, the current scratch buffer is safe to read
        for attention. The next layer's transfer is submitted later in
        ``prepare_for_attention`` (before the SFA kernel) so it overlaps with
        the SFA kernel, o_proj, and the layer MLP/MoE.
        """
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP layer {self._current_layer} was not completed before {layer_name}."
            )
        self._current_layer = layer_name
        self._transfer_waited = False

        if self._is_replicated(layer_name):
            # Replicated layer: no transfer, no wait. Mark as waited so the
            # state machine accepts the subsequent prepare/finish calls.
            self._transfer_waited = True
            return

        if self._pending_layer is None:
            self._ensure_transfer_started(layer_name)
        elif self._pending_layer != layer_name:
            raise RuntimeError(
                f"KVPP prefetched {self._pending_layer}, but forward entered "
                f"{layer_name}."
            )

        if self._transfer_future is not None and self.group.world_size > 1:
            layer_index = self._layer_indices[layer_name]
            with torch.profiler.record_function(
                f"kvpp.wait.previous_layer.layer_{layer_index}"
            ):
                self._transfer_future.result()
        self._transfer_waited = True
        self._pending_layer = None
        self._transfer_future = None

    def _ensure_transfer_started(self, layer_name: str) -> None:
        """Fallback: start a transfer that was not prefetched.

        Normally every layer's transfer is submitted by the previous layer's
        ``prepare_for_attention``. The only expected caller is ``begin_layer``
        when no prefetch is in flight (first layer, or a path that skipped
        the prefetch). Encapsulated here so the transport bootstrap logic is
        isolated from the state machine.
        """
        self._start_overlap_transfer(layer_name)

    def prepare_for_attention(self, layer_name: str) -> None:
        """Submit the next layer's transfer so it overlaps the SFA kernel.

        Called immediately before ``_execute_sparse_flash_attention_process``.
        By this point all KV writes for the current layer have been submitted
        on the compute stream, so for an owner rank the persistent cache is
        up to date and can be pushed; for a consumer rank the scratch buffer
        was already waited on in ``begin_layer`` and is being read by the
        upcoming SFA kernel. The next layer's transfer uses the *other*
        scratch buffer, so the two never collide.

        This method is non-blocking: it only submits work to the transport
        worker thread and returns immediately. The actual wait for the next
        layer's transfer happens in the next layer's ``begin_layer``.
        """
        if self._current_layer != layer_name:
            raise RuntimeError(f"No active KVPP layer for {layer_name}.")
        if not self._transfer_waited:
            raise RuntimeError(
                f"KVPP prepare_for_attention for {layer_name} called before "
                "begin_layer wait completed."
            )
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP transfer for {self._pending_layer} is still pending."
            )

        layer_index = self._layer_indices[layer_name]
        next_index = layer_index + 1
        if next_index < len(self._ordered_layers):
            self._start_overlap_transfer(
                self._ordered_layers[next_index]
            )

    def finish_layer_attention(self, layer_name: str) -> None:
        """Mark the current layer's attention submission complete.

        The call site is after the SFA kernel and ``_v_up_proj`` have been
        submitted, but before ``o_proj`` and the layer MLP/MoE. With dual
        scratch buffers the next transfer was already submitted in
        ``prepare_for_attention`` and uses the other scratch buffer.
        Compute-stream ordering protects reuse when this buffer cycles back
        two layers later.
        """
        if self._current_layer != layer_name or not self._transfer_waited:
            raise RuntimeError(
                f"KVPP attention for layer {layer_name} finished before its "
                "transfer was consumed."
            )
        self._current_layer = None
        self._transfer_waited = False

    def _start_overlap_transfer(self, layer_name: str) -> None:
        """Submit the transfer for ``layer_name`` off the compute thread.

        Records a compute-stream event that the transport worker waits on
        before writing into the scratch buffer (ensuring the previous
        reader has finished). The actual owner/consumer protocol and
        ``transport.push/receive`` run inside ``_run_overlap_transfer``
        on the transport worker thread.

        Replicated layers have no transfer: ``_pending_layer`` is set so the
        state machine tracks the in-flight layer, but ``_transfer_future``
        stays ``None`` and no work is submitted. The matching ``begin_layer``
        wait is then a no-op.
        """
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP transfer for {self._pending_layer} is still pending."
            )
        self._pending_layer = layer_name
        self._transfer_future = None
        if self._is_replicated(layer_name):
            return
        if self.group.world_size <= 1:
            return
        if self._executor is None or self._comm_stream is None:
            raise RuntimeError("KVPP overlap worker was not initialized.")

        # All ranks publish a local safe point. Alternating layers use
        # distinct scratch buffers. When a buffer cycles back after two
        # layers, this event is ordered after all earlier attention work
        # on the compute stream, so the owner cannot overwrite a buffer
        # that is still being read.
        scratch_ready = torch.npu.Event()
        scratch_ready.record(torch.npu.current_stream())
        pages = self._selected_pages
        assert pages is not None
        self._transfer_future = self._executor.submit(
            self._run_overlap_transfer, layer_name, pages, scratch_ready
        )

    def _run_overlap_transfer(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        scratch_ready: Any,
    ) -> None:
        """Run safe-point and completion notification off the compute thread.

        Owner rank: wait for all consumers' scratch-ready, push active
        pages to every peer, then notify done.
        Consumer rank: notify owner scratch-ready, wait for owner done,
        then receive active pages into local scratch.

        The transport is ``NullTransport`` when ``kvpp_size <= 1`` or when no
        real backend was wired; in that case push/receive are no-ops and the
        CPU ready/done protocol is skipped. Replicated layers (owner ==
        ``KVPP_REPLICATED_OWNER``) never reach here because
        ``_start_overlap_transfer`` returns early for them.
        """
        if self.transport is None or self._comm_stream is None:
            raise RuntimeError("KVPP overlap transport is not initialized.")
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)

        if isinstance(self.transport, NullTransport):
            # No-op path: nothing to transfer, return immediately.
            return

        owner_rank = self._execution_owners[layer_name]
        if owner_rank == KVPP_REPLICATED_OWNER:
            # Defensive: replicated layers should never reach here because
            # _start_overlap_transfer skips them. No transfer, no protocol.
            return

        local_rank = self.group.rank_in_group
        owner_global_rank = self.group.ranks[owner_rank]
        layer_index = self._layer_indices[layer_name]
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
                # Consumer: tell owner our scratch is safe to write, wait for
                # owner's push to finish, then unpack into local scratch.
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
                        receive_completions = [
                            self.transport.receive_active_pages(
                                cache_layer_name, pages, self._comm_stream
                            )
                            for cache_layer_name in self._layer_bundles[layer_name]
                        ]
                    for completion in receive_completions:
                        completion.wait()
                return

            # Owner: wait for every consumer's scratch-ready, push to all
            # peers, then notify each consumer that the push is done.
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
                    completions = [
                        self.transport.push_active_pages(
                            cache_layer_name, pages, self._comm_stream
                        )
                        for cache_layer_name in self._layer_bundles[layer_name]
                    ]
                # Only the communication worker waits on the host. The compute
                # thread continues until this layer first writes/reads its
                # paged KV cache.
                for completion in completions:
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
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None


def make_kvpp_context(
    group: GroupCoordinator,
    layer_owners: dict[str, int],
    num_blocks: int,
    block_size: int,
    execution_layers: tuple[str, ...] | None = None,
    transport: KVPPTransport | None = None,
) -> KVPPContext:
    """Construct a :class:`KVPPContext`.

    When ``transport`` is ``None`` the context falls back to
    :class:`NullTransport`, which is only appropriate for ``kvpp_size <= 1``.
    Production callers should pass the result of
    :func:`create_kvpp_transport` (MTE by default).
    """
    return KVPPContext(
        group=group,
        layer_owners=layer_owners,
        num_blocks=num_blocks,
        block_size=block_size,
        execution_layers=execution_layers,
        transport=transport,
    )


def initialize_kvpp_for_runner(
    runner: Any,
    kv_caches_dict: dict[str, Any] | None,
) -> KVPPContext | None:
    """Build and bind a :class:`KVPPContext` onto a model runner.

    Returns ``None`` when KVPP is inactive (``kvpp_size <= 1``). Otherwise:

    * Validates the KV cache config carries KVPP placement metadata.
    * Discovers the SFA attention implementations that will drive
      per-layer transfers (GLM5.2 path).
    * Constructs the :class:`KVPPContext`, initializes its transport with
      the retained cache tensors, and injects the context into every
      discovered attention impl so their ``forward`` can call
      ``begin_layer`` / ``prepare_for_attention`` /
      ``finish_layer_attention``.

    All KVPP-specific discovery and wiring lives here so the model runner
    only has to call this single function.
    """
    kvpp_size = runner.vllm_config.parallel_config.kvpp_size
    if kvpp_size <= 1:
        return None

    kv_cache_config = runner.kv_cache_config
    layer_owners = kv_cache_config.kvpp_layer_owners
    if layer_owners is None:
        raise RuntimeError("KVPP cache placement is missing from KVCacheConfig.")

    kvpp_group = get_kvpp_group()
    if kv_cache_config.kvpp_rank != kvpp_group.rank_in_group:
        raise RuntimeError(
            "KVPP cache placement rank does not match the worker's "
            f"communication rank: config={kv_cache_config.kvpp_rank}, "
            f"group={kvpp_group.rank_in_group}."
        )

    if len(kv_cache_config.kv_cache_groups) != 1:
        raise RuntimeError("KVPP currently requires exactly one KV cache group.")

    # Block tables index kernel blocks. Transport descriptors must use that
    # same address space when one external block expands to multiple blocks.
    blocks_per_kv_block = runner.block_tables.blocks_per_kv_block[0]
    num_kernel_blocks = kv_cache_config.num_blocks * blocks_per_kv_block
    block_size = runner.block_tables.kernel_block_sizes[0]

    # Discover SFA attention impls that will drive layer transfers. SFA
    # indexer cache layers have their own AttentionImpl, but execute as
    # part of the main SFA forward and are bundled by transformer index.
    kvpp_impls: dict[str, Any] = {}
    for layer_name, module in runner.compilation_config.static_forward_context.items():
        if layer_name not in layer_owners:
            continue
        impl = getattr(module, "impl", None)
        if impl is not None and hasattr(impl, "kvpp_context"):
            kvpp_impls[layer_name] = impl
    if not kvpp_impls:
        raise RuntimeError(
            "KVPP did not find an SFA attention implementation to "
            "drive layer transfers."
        )

    from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
        create_kvpp_transport,
    )

    transport = create_kvpp_transport(
        kvpp_group,
        layer_owners,
        num_blocks=num_kernel_blocks,
    )
    kvpp_context = KVPPContext(
        kvpp_group,
        layer_owners,
        num_blocks=num_kernel_blocks,
        block_size=block_size,
        execution_layers=tuple(kvpp_impls),
        transport=transport,
    )
    if kv_caches_dict is None:
        raise RuntimeError("KVPP cache tensors were not retained during allocation.")
    kvpp_context.initialize_transport(kv_caches_dict)
    for impl in kvpp_impls.values():
        impl.kvpp_context = kvpp_context
    return kvpp_context
