# SPDX-License-Identifier: Apache-2.0
"""MemFabric Hybrid MTE backend for KV layer parallelism.

MemFabric Hybrid 1.2 cannot export an existing KV tensor as a remote MTE GVA.
This backend therefore allocates one bounded symmetric active-page staging
segment per rank. Layer owners copy selected persistent pages directly to each
consumer's segment with one batched AscendC GM->UB->remote-GM launch. Consumers
unpack the same device-resident page descriptors into their existing scratch
cache before attention. No full layer cache is copied or staged.

The AscendC kernel is *descriptorless*: it receives the raw page IDs, valid
mask, and per-buffer layout constants (base offsets, strides, byte sizes,
staging buffer offsets) and computes every local/staging offset internally.
This eliminates the per-layer ``cumsum``/``Mul``/``Add``/``Cast`` chain that
the previous pre-computed-descriptor design paid on the metadata stream, and
removes the duplicated staging-offset calculation across peers.

Replicated layers (owner == ``KVPP_REPLICATED_OWNER``, e.g. layer 0) are
skipped at the runtime layer and never reach this transport; the defensive
checks below return an already-complete completion just in case.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger

from vllm_ascend.core.kvpp_allocation import KVPP_REPLICATED_OWNER
from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPActivePages,
    KVPPBufferMetadata,
    KVPPCompletion,
    build_kvpp_layer_metadata,
    flatten_kvpp_cache,
)


_DEFAULT_STAGING_BYTES = 256 << 20
_DEFAULT_SHM_ID = 31
_SHM_ID_LIMIT = 64
_SHM_ALIGNMENT = 2 << 20


@dataclass(frozen=True)
class KVPPMTEPeerMetadata:
    """Backend-specific symmetric address published by one rank."""

    staging_addr: int
    staging_bytes: int
    rank: int


@dataclass(frozen=True)
class MemFabricMTECompletion:
    """Stream event recorded after all MTE kernels in one phase."""

    event: Any
    resources: tuple[Any, ...] = ()

    @classmethod
    def record(
        cls, stream: Any, resources: tuple[Any, ...] = ()
    ) -> "MemFabricMTECompletion":
        event = torch.npu.Event()
        event.record(stream)
        return cls(event, resources)

    def wait(self) -> None:
        self.event.synchronize()

    def wait_on_stream(self, stream: Any) -> None:
        stream.wait_event(self.event)


@dataclass(frozen=True)
class _MTELayerLayout:
    """Per-layer device constants consumed by the descriptorless kernel.

    ``local_base_offsets`` is the only per-layer field that changes with the
    anchor tensor. ``block_strides`` and ``block_bytes`` are identical across
    layers that share a KV layout, but are kept per-layer to avoid a layout
    lookup at launch time. Each tensor holds ``num_buffers`` int64 values.
    """

    anchor: torch.Tensor
    local_base_offsets: torch.Tensor
    block_strides: torch.Tensor
    block_bytes: torch.Tensor
    staging_buffer_offsets: torch.Tensor
    num_buffers: int
    staging_bytes_per_slot: int


class MemFabricMTEKVPPTransport:
    """Move active physical pages through bounded symmetric MTE staging."""

    # Owner pushes land in transport-owned staging rather than the attention
    # scratch tensor. The runtime may therefore overlap that push with the
    # previous reader and wait for scratch safety only before receive/unpack.
    uses_staging_buffer = True

    def __init__(
        self,
        group: GroupCoordinator,
        layer_owners: dict[str, int],
        num_blocks: int,
        *,
        shm_module: Any | None = None,
        copy_op: Callable[..., None] | None = None,
    ) -> None:
        self.group = group
        self.layer_owners = layer_owners
        self.num_blocks = num_blocks
        self._shm_module = shm_module
        self._memfabric_module: Any | None = None
        self._global_initialized = False
        self._copy_op = copy_op
        self._memory: Any | None = None
        self._local_metadata: KVPPMTEPeerMetadata | None = None
        self._peer_metadata: list[KVPPMTEPeerMetadata] = []
        self._layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
        self._layouts: dict[str, _MTELayerLayout] = {}
        self._shm_id = _DEFAULT_SHM_ID
        self._batch_pages: KVPPActivePages | None = None
        self._batch_valid_mask_int8: torch.Tensor | None = None

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        if not os.getenv("MEMFABRIC_HYBRID_HOME_PATH"):
            raise RuntimeError(
                "KVPP MTE requires the MemFabric Hybrid environment. Source "
                "/usr/local/memfabric_hybrid/set_env.sh before launching vLLM."
            )
        if self._shm_module is None:
            try:
                import memfabric_hybrid  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "KVPP MTE requires memfabric_hybrid.shm."
                ) from exc
            self._memfabric_module = memfabric_hybrid
            self._shm_module = memfabric_hybrid.shm
        if self._copy_op is None:
            import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401

            namespace = getattr(torch.ops, "_C_ascend", None)
            self._copy_op = getattr(namespace, "kvpp_mte_copy", None)
            if self._copy_op is None:
                raise RuntimeError(
                    "KVPP MTE custom operator is unavailable. Rebuild "
                    "vllm-ascend after sourcing MemFabric Hybrid 1.2.0."
                )

        store_url = os.getenv("MF_CONFIG_STORE_URL") or os.getenv(
            "ASCEND_MF_STORE_URL"
        )
        if not store_url:
            raise RuntimeError(
                "KVPP MTE requires MF_CONFIG_STORE_URL (or the deprecated "
                "ASCEND_MF_STORE_URL compatibility variable)."
            )
        staging_bytes = int(
            os.getenv("ASCEND_KVPP_MTE_STAGING_BYTES", _DEFAULT_STAGING_BYTES)
        )
        if staging_bytes <= 0 or staging_bytes % _SHM_ALIGNMENT:
            raise ValueError(
                "ASCEND_KVPP_MTE_STAGING_BYTES must be a positive multiple "
                f"of {_SHM_ALIGNMENT} bytes, got {staging_bytes}."
            )
        shm_id = int(os.getenv("ASCEND_KVPP_MTE_SHM_ID", _DEFAULT_SHM_ID))
        if not 0 <= shm_id < _SHM_ID_LIMIT:
            raise ValueError(
                f"ASCEND_KVPP_MTE_SHM_ID must be in [0, {_SHM_ID_LIMIT}), "
                f"got {shm_id}."
            )
        self._shm_id = shm_id

        config = self._shm_module.ShmConfig()
        config.start_store = self.group.rank_in_group == 0
        timeout = int(os.getenv("ASCEND_KVPP_MTE_TIMEOUT_SECONDS", "120"))
        config.init_timeout = timeout
        config.create_timeout = timeout
        config.operation_timeout = timeout
        device_id = torch.npu.current_device()
        if self._memfabric_module is not None:
            ret = self._memfabric_module.initialize(0)
            if ret != 0:
                raise RuntimeError(
                    "KVPP MemFabric global initialization failed: "
                    f"error={ret}."
                )
            self._global_initialized = True
        ret = self._shm_module.initialize(
            store_url,
            self.group.world_size,
            self.group.rank_in_group,
            device_id,
            config,
        )
        if ret != 0:
            raise RuntimeError(
                f"KVPP MemFabric SHM initialization failed: error={ret}."
            )
        self._memory = self._shm_module.create(
            shm_id,
            self.group.world_size,
            self.group.rank_in_group,
            staging_bytes,
            self._shm_module.ShmDataOpType.MTE,
        )
        if self._memory is None:
            raise RuntimeError("KVPP MemFabric SHM creation returned no memory.")
        operation = int(self._memory.query_support_data_operation())
        if operation != int(self._shm_module.ShmDataOpType.MTE.value):
            raise RuntimeError(
                "KVPP MemFabric SHM does not support MTE: "
                f"reported operation={operation}."
            )
        self._memory.barrier()

        self._layers = build_kvpp_layer_metadata(kv_caches, self.num_blocks)
        self._layouts = {}
        for layer_name, buffers in self._layers.items():
            anchor = flatten_kvpp_cache(kv_caches[layer_name])[0]
            device = anchor.device
            block_bytes_values = [b.block_bytes for b in buffers]
            staging_bytes_per_slot = sum(block_bytes_values)
            max_active_pages = staging_bytes // staging_bytes_per_slot
            if max_active_pages == 0:
                raise RuntimeError(
                    "KVPP MTE staging segment cannot hold one page across "
                    f"every cache buffer for layer {layer_name}: "
                    f"page_bytes={staging_bytes_per_slot}, "
                    f"capacity={staging_bytes}."
                )
            # Per-buffer staging start offsets: buffer i begins after
            # (max_active_pages * block_bytes[0..i-1]) bytes. This is a
            # constant for the layer and does not change per batch, so it is
            # computed once here and passed verbatim to every kernel launch.
            per_buffer_capacity = [
                b * max_active_pages for b in block_bytes_values
            ]
            cumulative = 0
            staging_buffer_offsets_values: list[int] = []
            for capacity in per_buffer_capacity:
                staging_buffer_offsets_values.append(cumulative)
                cumulative += capacity
            self._layouts[layer_name] = _MTELayerLayout(
                anchor=anchor,
                local_base_offsets=torch.tensor(
                    [b.base_addr - anchor.data_ptr() for b in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_strides=torch.tensor(
                    [b.block_stride_bytes for b in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_bytes=torch.tensor(
                    block_bytes_values,
                    dtype=torch.int64,
                    device=device,
                ),
                staging_buffer_offsets=torch.tensor(
                    staging_buffer_offsets_values,
                    dtype=torch.int64,
                    device=device,
                ),
                num_buffers=len(buffers),
                staging_bytes_per_slot=staging_bytes_per_slot,
            )
        self._local_metadata = KVPPMTEPeerMetadata(
            staging_addr=int(self._memory.gva),
            staging_bytes=staging_bytes,
            rank=self.group.rank_in_group,
        )
        peers: list[KVPPMTEPeerMetadata | None] = [None] * self.group.world_size
        dist.all_gather_object(
            peers, self._local_metadata, group=self.group.cpu_group
        )
        if any(peer is None for peer in peers):
            raise RuntimeError("KVPP MTE did not receive every peer GVA.")
        self._peer_metadata = [peer for peer in peers if peer is not None]
        # Staging buffer offsets are computed once from the local staging size
        # and reused for every peer. Asymmetric staging segments would make
        # those offsets wrong, so reject the deployment early instead of
        # silently corrupting transfers.
        for peer in self._peer_metadata:
            if peer.staging_bytes != self._local_metadata.staging_bytes:
                raise RuntimeError(
                    "KVPP MTE requires symmetric staging segments: local="
                    f"{self._local_metadata.staging_bytes}, peer={peer.rank}="
                    f"{peer.staging_bytes}. Set ASCEND_KVPP_MTE_STAGING_BYTES "
                    "to the same value on every rank."
                )
        logger.info(
            "KVPP MemFabric MTE initialized: rank=%d, gva=%#x, "
            "staging_bytes=%d, shm_id=%d",
            self.group.rank_in_group,
            self._local_metadata.staging_addr,
            staging_bytes,
            shm_id,
        )

    def prepare_batch(self, pages: KVPPActivePages) -> None:
        """Reset batch-scoped metadata reused by every layer and peer.

        The int8 view is created lazily by the first launch on the communication
        stream. Keeping it for the batch avoids one cast/allocation per layer
        and per destination without introducing a cross-stream dependency.
        """
        self._batch_pages = pages
        self._batch_valid_mask_int8 = None

    def _check_staging_capacity(
        self, layer_name: str, pages: KVPPActivePages
    ) -> None:
        layout = self._layouts[layer_name]
        max_active_pages = (
            self._local_metadata.staging_bytes
            // layout.staging_bytes_per_slot
            if self._local_metadata is not None
            else 0
        )
        if pages.count_upper_bound > max_active_pages:
            raise RuntimeError(
                "KVPP MTE active-page upper bound exceeds staging capacity: "
                f"upper_bound={pages.count_upper_bound}, "
                f"capacity={max_active_pages}, "
                f"staging_bytes={self._local_metadata.staging_bytes}. "
                "Increase ASCEND_KVPP_MTE_STAGING_BYTES."
            )

    def _launch(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        *,
        staging_base: int,
        source_rank: int = -1,
        destination_rank: int = -1,
    ) -> None:
        assert self._copy_op is not None
        layout = self._layouts[layer_name]
        capacity = pages.page_ids.numel()
        if self._batch_pages is not pages:
            raise RuntimeError(
                "KVPP MTE received pages that were not prepared for this batch."
            )
        if self._batch_valid_mask_int8 is None:
            # Created on the active communication stream and retained until
            # prepare_batch() starts the next batch.
            self._batch_valid_mask_int8 = pages.valid_mask.to(dtype=torch.int8)
        valid_mask_int8 = self._batch_valid_mask_int8
        self._copy_op(
            layout.anchor,
            pages.page_ids,
            valid_mask_int8,
            layout.local_base_offsets,
            layout.block_strides,
            layout.block_bytes,
            layout.staging_buffer_offsets,
            capacity,
            layout.num_buffers,
            staging_base,
            source_rank,
            destination_rank,
            self._shm_id,
        )

    def push_active_pages(
        self, layer_name: str, pages: KVPPActivePages, stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank == KVPP_REPLICATED_OWNER:
            return MemFabricMTECompletion.record(stream)
        if owner_rank != self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None or not self._peer_metadata:
            raise RuntimeError("KVPP MTE transport was not initialized.")
        self._check_staging_capacity(layer_name, pages)

        # All peers share the same page_ids / valid_mask / layout tensors;
        # only staging_base and destination_rank differ. The kernel computes
        # every offset internally, so there is no per-peer Python work.
        for peer_rank, peer in enumerate(self._peer_metadata):
            if peer_rank == owner_rank:
                continue
            self._launch(
                layer_name,
                pages,
                staging_base=peer.staging_addr,
                destination_rank=peer.rank,
            )
        return MemFabricMTECompletion.record(stream, (pages.page_ids,))

    def receive_active_pages(
        self, layer_name: str, pages: KVPPActivePages, stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank == KVPP_REPLICATED_OWNER:
            return MemFabricMTECompletion.record(stream)
        if owner_rank == self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None:
            raise RuntimeError("KVPP MTE transport was not initialized.")
        self._check_staging_capacity(layer_name, pages)

        self._launch(
            layer_name,
            pages,
            staging_base=self._local_metadata.staging_addr,
            source_rank=self._local_metadata.rank,
        )
        return MemFabricMTECompletion.record(stream, (pages.page_ids,))

    def close(self) -> None:
        if self._memory is not None:
            self._memory.destroy()
            self._memory = None
        if self._global_initialized:
            self._shm_module.uninitialize()
            assert self._memfabric_module is not None
            self._memfabric_module.uninitialize()
            self._global_initialized = False
        self._peer_metadata.clear()
        self._local_metadata = None
        self._layers.clear()
        self._layouts.clear()
        self._batch_pages = None
        self._batch_valid_mask_int8 = None
