from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
    ) -> KVPPExecutionPlan:
        layers = tuple(execution_layers or layer_owners)
        if not layers:
            raise ValueError("KVPP requires at least one executable attention layer.")

        cache_bundles: dict[str, tuple[str, ...]]
        if execution_layers is None:
            cache_bundles = {layer_name: (layer_name,) for layer_name in layers}
        else:
            cache_layers_by_index: dict[int, list[str]] = {}
            for cache_layer_name in sorted(
                layer_owners,
                key=lambda name: (extract_layer_index(name), name),
            ):
                cache_layers_by_index.setdefault(extract_layer_index(cache_layer_name), []).append(cache_layer_name)

            cache_bundles = {}
            claimed_indices: set[int] = set()
            for layer_name in layers:
                if layer_name not in layer_owners:
                    raise ValueError(f"KVPP execution layer {layer_name} has no KV cache owner.")
                layer_index = extract_layer_index(layer_name)
                if layer_index in claimed_indices:
                    raise ValueError(
                        f"KVPP received multiple executable attention layers for transformer layer {layer_index}."
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
                raise ValueError(f"KVPP cache layers have no executable attention owner: {sorted(unclaimed)}.")

        for layer_name, cache_layer_names in cache_bundles.items():
            bundle_owners = {layer_owners[name] for name in cache_layer_names}
            if len(bundle_owners) != 1:
                raise ValueError(f"KVPP cache bundle for {layer_name} spans owners {sorted(bundle_owners)}.")
        return cls(cache_bundles=cache_bundles)


def validate_local_mtp_layers(
    layer_names: tuple[str, ...],
    draft_layer_names: set[str],
    *,
    is_last_pp_rank: bool,
) -> None:
    """Fail fast on wrong MTP placement for this PP stage.

    The last PP stage must hold the MTP cache (replicated across KVPP
    ranks); non-last stages must not hold any MTP cache.
    """
    local_mtp_layers = set(draft_layer_names).intersection(layer_names)
    if is_last_pp_rank and not local_mtp_layers:
        raise RuntimeError(
            "MTP is enabled on the last pipeline stage, but no MTP KV "
            "cache layers were found in the local cache groups."
        )
    if not is_last_pp_rank and local_mtp_layers:
        raise RuntimeError(
            f"Non-last pipeline stages must not hold MTP KV cache layers, but found {sorted(local_mtp_layers)}."
        )


def _collect_kvpp_attention_impls(
    static_forward_context: dict[str, Any],
    layer_owners: dict[str, int],
) -> dict[str, Any]:
    impls: dict[str, Any] = {}
    for layer_name, module in static_forward_context.items():
        if layer_name not in layer_owners:
            continue
        impl = getattr(module, "impl", None)
        if impl is not None and hasattr(impl, "layerwise_kv_cache_hook"):
            impls[layer_name] = impl
    if not impls:
        raise RuntimeError("KVPP did not find an MLA or SFA attention implementation to drive layer transfers.")
    return impls


def _bound_kv_caches(
    static_forward_context: dict[str, Any],
    layer_owners: dict[str, int],
) -> dict[str, Any]:
    managed_kv_caches: dict[str, Any] = {}
    for layer_name in layer_owners:
        module = static_forward_context.get(layer_name)
        if module is None or not hasattr(module, "kv_cache"):
            raise RuntimeError(f"KVPP could not find the bound cache for logical layer {layer_name!r}.")
        managed_kv_caches[layer_name] = module.kv_cache
    return managed_kv_caches


def validate_v1_mtp_layers(
    layer_names: tuple[str, ...],
    layer_owners: dict[str, int],
    *,
    speculative_config: Any,
    hf_config: Any,
    is_last_pp_rank: bool,
) -> None:
    """Fail fast on wrong MTP placement for a V1 pipeline stage."""
    if speculative_config is None or speculative_config.method != "mtp":
        return
    mtp_start = getattr(hf_config, "num_hidden_layers", None)
    num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", None)
    if mtp_start is None or not num_mtp_layers:
        return
    mtp_end = mtp_start + num_mtp_layers
    local_mtp_layers = {name for name in layer_names if mtp_start <= extract_layer_index(name) < mtp_end}
    if is_last_pp_rank and not local_mtp_layers:
        raise RuntimeError(
            "MTP is enabled on the last pipeline stage, but no MTP KV "
            "cache layers were found in the local cache groups."
        )
    if not is_last_pp_rank and local_mtp_layers:
        raise RuntimeError(
            f"Non-last pipeline stages must not hold MTP KV cache layers, but found {sorted(local_mtp_layers)}."
        )
    owned_mtp_layers = local_mtp_layers & set(layer_owners)
    if owned_mtp_layers:
        raise RuntimeError(
            "MTP attention layers must be replicated outside KVPP, "
            "but these layers have KVPP owners: "
            f"{sorted(owned_mtp_layers)}."
        )


def _kvpp_layer_names(kv_cache_config: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(layer_name for group in kv_cache_config.kv_cache_groups for layer_name in group.layer_names)
    )


class KVPPRuntime:
    """Model-runner facing KVPP glue: placement, scheduler, and lifecycle."""

    def __init__(
        self,
        scheduler: KVPPScheduler | None = None,
        cache_group_index: int | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.cache_group_index = cache_group_index

    @classmethod
    def _placement(
        cls,
        vllm_config: Any,
        kv_cache_config: Any,
    ) -> tuple[tuple[str, ...], dict[str, int], int] | None:
        from vllm_ascend.kvpp_config import KVPPConfig
        from vllm_ascend.v1.core.kv_cache_placement import get_kvpp_layer_owners

        if KVPPConfig.from_vllm_config(vllm_config).size <= 1:
            return None
        layer_names = _kvpp_layer_names(kv_cache_config)
        layer_owners = get_kvpp_layer_owners(vllm_config, layer_names)
        cache_group_index = get_kvpp_managed_group_index(kv_cache_config.kv_cache_groups, layer_owners)
        return layer_names, layer_owners, cache_group_index

    @classmethod
    def _assemble(
        cls,
        *,
        layer_owners: dict[str, int],
        cache_group_index: int,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any],
        num_kernel_blocks: int,
        block_size: int,
    ) -> KVPPRuntime:
        from vllm_ascend.distributed.parallel_state import get_kvpp_group

        kvpp_group = get_kvpp_group()
        kvpp_impls = _collect_kvpp_attention_impls(static_forward_context, layer_owners)
        scheduler = KVPPScheduler(
            group=kvpp_group,
            layer_owners=layer_owners,
            num_blocks=num_kernel_blocks,
            block_size=block_size,
            transport=MemFabricMTEKVPPTransport(kvpp_group, layer_owners, num_kernel_blocks),
            execution_layers=tuple(kvpp_impls),
        )
        scheduler.initialize_transport(kv_caches)
        for impl in kvpp_impls.values():
            impl.layerwise_kv_cache_hook = scheduler
        return cls(scheduler=scheduler, cache_group_index=cache_group_index)

    @classmethod
    def try_build(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        block_tables: Any,
        static_forward_context: dict[str, Any],
        speculator: Any | None,
        is_last_pp_rank: bool,
    ) -> KVPPRuntime:
        placement = cls._placement(vllm_config, kv_cache_config)
        if placement is None:
            return cls()
        layer_names, layer_owners, cache_group_index = placement

        if speculator is not None:
            draft_layer_names = speculator.draft_attn_layer_names or set()
            managed_draft_layers = set(layer_owners).intersection(draft_layer_names)
            if managed_draft_layers:
                raise RuntimeError(
                    "MTP attention layers must be replicated outside KVPP, "
                    "but these layers have KVPP owners: "
                    f"{sorted(managed_draft_layers)}."
                )
            validate_local_mtp_layers(
                layer_names,
                draft_layer_names,
                is_last_pp_rank=is_last_pp_rank,
            )

        blocks_per_kv_block = block_tables.blocks_per_kv_block[cache_group_index]
        return cls._assemble(
            layer_owners=layer_owners,
            cache_group_index=cache_group_index,
            static_forward_context=static_forward_context,
            kv_caches=_bound_kv_caches(static_forward_context, layer_owners),
            num_kernel_blocks=kv_cache_config.num_blocks * blocks_per_kv_block,
            block_size=block_tables.kernel_block_sizes[cache_group_index],
        )

    @classmethod
    def try_build_v1(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any],
        block_tables: Any,
        is_last_pp_rank: bool,
    ) -> KVPPRuntime:
        placement = cls._placement(vllm_config, kv_cache_config)
        if placement is None:
            return cls()
        layer_names, layer_owners, cache_group_index = placement
        validate_v1_mtp_layers(
            layer_names,
            layer_owners,
            speculative_config=vllm_config.speculative_config,
            hf_config=vllm_config.model_config.hf_config,
            is_last_pp_rank=is_last_pp_rank,
        )
        block_table = block_tables[cache_group_index]
        return cls._assemble(
            layer_owners=layer_owners,
            cache_group_index=cache_group_index,
            static_forward_context=static_forward_context,
            kv_caches={name: kv_caches[name] for name in layer_owners},
            num_kernel_blocks=(kv_cache_config.num_blocks * block_table.blocks_per_phys_block),
            block_size=block_table.logical_block_size,
        )

    def begin_forward(
        self,
        block_tables: tuple[torch.Tensor, ...],
        seq_lens: Any,
    ) -> None:
        if self.scheduler is None:
            return
        assert self.cache_group_index is not None
        self.scheduler.begin_forward(block_tables[self.cache_group_index], seq_lens)

    def begin_v1_forward(
        self,
        input_batch: Any,
        num_reqs: int,
        seq_lens: Any,
    ) -> None:
        if self.scheduler is None:
            return
        assert self.cache_group_index is not None
        block_table = input_batch.block_table[self.cache_group_index]
        self.scheduler.begin_forward(
            block_table.get_device_tensor(num_reqs),
            seq_lens,
        )

    def finish_forward(self, *, dummy_skip_attn: bool = False) -> None:
        if self.scheduler is None:
            return
        if dummy_skip_attn:
            self.scheduler.reset_forward()
            return
        self.scheduler.finish_forward()

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
            self.cache_group_index = None


def get_kvpp_managed_group_index(
    kv_cache_groups: list[Any],
    layer_owners: dict[str, int],
) -> int:
    managed_layers = set(layer_owners)
    configured_layers = {layer_name for group in kv_cache_groups for layer_name in group.layer_names}
    missing_layers = managed_layers - configured_layers
    if missing_layers:
        raise ValueError(f"KVPP-owned cache layers are missing from KV cache groups: {sorted(missing_layers)}.")

    managed_group_indices = {
        index for index, group in enumerate(kv_cache_groups) if managed_layers.intersection(group.layer_names)
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
    host_lengths = torch.as_tensor(seq_lens, dtype=torch.int64, device="cpu").flatten()
    pages_per_request_host = torch.div(
        host_lengths + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).clamp_(min=0, max=table_columns)
    count_upper_bound = min(num_blocks, int(pages_per_request_host.sum().item()))
    lengths = host_lengths.to(device=block_table.device)
    table = block_table[: lengths.shape[0]].to(dtype=torch.int64)
    columns = torch.arange(table.shape[1], dtype=torch.int64, device=block_table.device)
    pages_per_request = torch.div(lengths + block_size - 1, block_size, rounding_mode="floor")
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
        self.plan = KVPPExecutionPlan.build(self.layer_owners, execution_layers)
        self._phase = KVPPPhase.IDLE
        self._next_layer_index = 0
        self._selected_pages: KVPPActivePages | None = None
        self._comm_stream: Any | None = None
        self._current_layer: str | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._transfer_future: Future[None] | None = None
        self._pending_layer: str | None = None
        self._device_id: int | None = None

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        self._validate_plan_across_ranks()
        self.transport.initialize(kv_caches)
        if self.group.world_size > 1:
            self._device_id = torch.npu.current_device()
            self._comm_stream = torch.npu.Stream()
            # One transfer may be in flight. Serializing jobs also preserves
            # point-to-point notification order when layer ownership changes.
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpp-prefetch")

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
        mismatched_ranks = [rank for rank, peer_signature in enumerate(peer_signatures) if peer_signature != signature]
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
        active_pages = _active_pages(block_table, seq_lens, self.block_size, self.num_blocks)
        self._begin_forward(active_pages)

    def _begin_forward(self, active_pages: KVPPActivePages) -> None:
        if self._phase is not KVPPPhase.IDLE:
            raise RuntimeError(f"KVPP cannot begin a forward while phase is {self._phase.name}.")
        if self._current_layer is not None or self._pending_layer is not None:
            raise RuntimeError("KVPP cannot start with layer work pending.")
        self._selected_pages = active_pages
        self._next_layer_index = 0
        self._phase = KVPPPhase.FORWARD_ACTIVE

    def finish_forward(self) -> None:
        if self._phase is KVPPPhase.IDLE:
            return
        if self._current_layer is not None:
            raise RuntimeError(f"KVPP forward ended while layer {self._current_layer} was active.")
        if self._pending_layer is not None:
            raise RuntimeError(f"KVPP forward ended with transfer for {self._pending_layer} pending.")
        if self._next_layer_index != len(self.plan.layers):
            raise RuntimeError(f"KVPP forward executed {self._next_layer_index}/{len(self.plan.layers)} layers.")
        self._selected_pages = None
        self._next_layer_index = 0
        self._phase = KVPPPhase.IDLE

    def reset_forward(self) -> None:
        """Drop an unfinished forward (dummy skip-attn) or drain before close."""
        if self._transfer_future is not None:
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
            raise RuntimeError(f"KVPP layer {self._current_layer} was not completed before {layer_name}.")
        if self._next_layer_index >= len(self.plan.layers):
            raise RuntimeError(f"KVPP forward entered unexpected extra layer {layer_name}.")
        expected_layer = self.plan.layers[self._next_layer_index]
        if layer_name != expected_layer:
            raise RuntimeError(f"KVPP forward expected {expected_layer}, but entered {layer_name}.")
        self._next_layer_index += 1

        self._current_layer = layer_name
        self._phase = KVPPPhase.LAYER_ENTERED
        if self._pending_layer is None:
            self._start_prefetch(layer_name)
        elif self._pending_layer != layer_name:
            raise RuntimeError(f"KVPP prefetched {self._pending_layer}, but forward entered {layer_name}.")

    def wait_for_layer(self, layer_name: str) -> None:
        """Order cache use, then prefetch the next layer before attention."""
        if self._phase is not KVPPPhase.LAYER_ENTERED or self._current_layer != layer_name:
            raise RuntimeError(f"No pending KVPP transfer for layer {layer_name}.")
        if self._pending_layer != layer_name:
            raise RuntimeError(f"No pending KVPP prefetch for layer {layer_name}.")

        if self._transfer_future is not None:
            # Eager path: this blocks only for the residual transfer time
            # because this layer was prefetched while the preceding layer
            # was executing.
            layer_index = extract_layer_index(layer_name)
            with torch.profiler.record_function(f"kvpp.wait.previous_layer.layer_{layer_index}"):
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
        if self._phase is not KVPPPhase.LAYER_WAITED or self._current_layer != layer_name:
            raise RuntimeError(f"KVPP attention for layer {layer_name} finished before its transfer was consumed.")

        self._current_layer = None
        self._phase = KVPPPhase.FORWARD_ACTIVE

    def _start_prefetch(self, layer_name: str) -> None:
        if self._pending_layer is not None:
            raise RuntimeError(f"KVPP transfer for {self._pending_layer} is still pending.")
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
        self._transfer_future = self._executor.submit(self._run_prefetch, layer_name, pages, scratch_ready)

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

        with torch.profiler.record_function(f"kvpp.comm_total.layer_{layer_index}"):
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
                with torch.profiler.record_function(f"kvpp.transport_receive.layer_{layer_index}"):
                    with torch.npu.stream(self._comm_stream):
                        receive_completion = self.transport.receive_active_bundle(
                            self.plan.cache_bundles[layer_name],
                            pages,
                            self._comm_stream,
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

            with torch.profiler.record_function(f"kvpp.transport_push.layer_{layer_index}"):
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
        self.reset_forward()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.transport.close()
