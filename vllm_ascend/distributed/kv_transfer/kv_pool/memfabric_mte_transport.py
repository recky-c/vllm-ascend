# SPDX-License-Identifier: Apache-2.0
"""MemFabric Hybrid MTE backend for KV layer parallelism.

MemFabric Hybrid 1.2 cannot export an existing KV tensor as a remote MTE GVA.
This backend therefore allocates one bounded symmetric active-page staging
segment per rank. Layer owners copy selected persistent pages directly to each
consumer's segment with one batched AscendC GM->UB->remote-GM launch. Consumers
unpack the same device-resident page descriptors into their existing scratch
cache before attention. No full layer cache is copied or staged.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger

_DEFAULT_STAGING_BYTES = 256 << 20
_DEFAULT_SHM_ID = 31
_SHM_ID_LIMIT = 64
_SHM_ALIGNMENT = 2 << 20


def _store_url_for_kvpp_group(store_url: str, group: GroupCoordinator) -> str:
    """Give each KVPP process group its own MemFabric config store.

    A PP deployment has one KVPP group per pipeline stage. MemFabric's SHM
    initializer identifies participants only by ``(store_url, world_size,
    rank_id)``; reusing one URL would therefore merge stage-local ranks from
    different PP stages. KVPP groups are contiguous slices of the global rank
    grid, so their ordinal can safely select a stage-local TCP port.
    """
    ranks = tuple(int(rank) for rank in group.ranks)
    if not ranks:
        raise ValueError("KVPP process group must contain at least one rank.")
    if len(ranks) != group.world_size:
        raise ValueError(
            "KVPP process-group rank count does not match its world size: "
            f"ranks={ranks}, world_size={group.world_size}."
        )
    first_rank = min(ranks)
    expected_ranks = tuple(range(first_rank, first_rank + group.world_size))
    if tuple(sorted(ranks)) != expected_ranks or first_rank % group.world_size:
        raise ValueError(
            f"KVPP MemFabric store isolation requires contiguous, aligned process groups, got ranks={ranks}."
        )
    group_index = first_rank // group.world_size
    if group_index == 0:
        return store_url

    parsed = urlsplit(store_url)
    if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Multiple KVPP groups require MF_CONFIG_STORE_URL to use the tcp://host:port form, got {store_url!r}."
        )
    port = parsed.port + group_index
    if port > 65535:
        raise ValueError(
            f"KVPP MemFabric derived store port exceeds 65535: base_port={parsed.port}, group_index={group_index}."
        )
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


@dataclass(frozen=True)
class KVPPBufferMetadata:
    """Address and logical-page layout for one KV cache tensor."""

    base_addr: int
    block_stride_bytes: int
    block_bytes: int


@dataclass(frozen=True)
class KVPPActivePages:
    """Fixed-shape device representation of active physical KV pages."""

    page_ids: torch.Tensor
    valid_mask: torch.Tensor
    count_upper_bound: int

    def __post_init__(self) -> None:
        if self.page_ids.device != self.valid_mask.device:
            raise ValueError("KVPP active page tensors must share one device.")
        if self.page_ids.dim() != 1 or self.valid_mask.dim() != 1:
            raise ValueError("KVPP active page tensors must be one-dimensional.")
        if self.page_ids.numel() != self.valid_mask.numel():
            raise ValueError("KVPP active page tensor lengths must match.")
        if self.page_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("KVPP page_ids must use int32 or int64.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("KVPP valid_mask must use bool.")
        if not 0 <= self.count_upper_bound <= self.page_ids.numel():
            raise ValueError(
                "KVPP active-page count upper bound must be between zero "
                f"and the descriptor count, got {self.count_upper_bound} "
                f"for {self.page_ids.numel()} descriptors."
            )


def flatten_kvpp_cache(cache: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(cache, torch.Tensor):
        return (cache,)
    if isinstance(cache, (tuple, list)):
        tensors = tuple(cache)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("KVPP cache tuples may contain only tensors.")
        return tensors
    raise TypeError(f"Unsupported KVPP cache type: {type(cache)!r}.")


def build_kvpp_layer_metadata(kv_caches: dict[str, Any], num_blocks: int) -> dict[str, tuple[KVPPBufferMetadata, ...]]:
    """Describe logical pages once for the MTE data plane."""
    layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
    for layer_name, cache in kv_caches.items():
        buffers: list[KVPPBufferMetadata] = []
        for tensor in flatten_kvpp_cache(cache):
            if tensor.ndim == 0 or tensor.shape[0] % num_blocks != 0:
                raise RuntimeError(
                    f"KVPP layer {layer_name} cache shape {tuple(tensor.shape)} "
                    f"cannot be divided into {num_blocks} logical blocks."
                )
            block_size_scale = tensor.shape[0] // num_blocks
            block_stride_bytes = tensor.stride(0) * tensor.element_size() * block_size_scale
            logical_block = tensor[0:block_size_scale]
            if not logical_block.is_contiguous():
                raise RuntimeError(
                    f"KVPP layer {layer_name} logical cache block is not "
                    "contiguous and cannot be transferred by address."
                )
            block_bytes = logical_block.numel() * tensor.element_size()
            if block_bytes > block_stride_bytes:
                raise RuntimeError(
                    f"KVPP layer {layer_name} has overlapping logical blocks: "
                    f"payload={block_bytes}, stride={block_stride_bytes}."
                )
            buffers.append(
                KVPPBufferMetadata(
                    base_addr=tensor.data_ptr(),
                    block_stride_bytes=block_stride_bytes,
                    block_bytes=block_bytes,
                )
            )
        layers[layer_name] = tuple(buffers)
    return layers


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
    def record(cls, stream: Any, resources: tuple[Any, ...] = ()) -> MemFabricMTECompletion:
        event = torch.npu.Event()
        event.record(stream)
        return cls(event, resources)

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def wait_on_stream(self, stream: Any) -> None:
        if self.event is not None:
            stream.wait_event(self.event)


@dataclass(frozen=True)
class _MTEDeviceBufferMetadata:
    local_base_offsets: torch.Tensor
    block_strides: torch.Tensor
    block_bytes: torch.Tensor
    staging_bytes_per_slot: int


@dataclass(frozen=True)
class _MTEDeviceDescriptors:
    local_offsets: torch.Tensor
    staging_offsets: torch.Tensor
    lengths: torch.Tensor
    staging_base: int

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.local_offsets, self.staging_offsets, self.lengths)


class MemFabricMTEKVPPTransport:
    """Move active physical pages through bounded symmetric MTE staging."""

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
        self._anchors: dict[str, torch.Tensor] = {}
        self._device_layers: dict[str, _MTEDeviceBufferMetadata] = {}
        self._shm_id = _DEFAULT_SHM_ID

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
                raise ImportError("KVPP MTE requires memfabric_hybrid.shm.") from exc
            self._memfabric_module = memfabric_hybrid
            self._shm_module = memfabric_hybrid.shm
        if self._copy_op is None:
            # vllm-ascend loads its native extension lazily to avoid early RTS
            # initialization. KVPP needs the operator during cache setup, so
            # trigger that load before querying the dispatcher namespace.
            import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401

            namespace = getattr(torch.ops, "_C_ascend", None)
            self._copy_op = getattr(namespace, "kvpp_mte_copy", None)
            if self._copy_op is None:
                raise RuntimeError(
                    "KVPP MTE custom operator is unavailable. Rebuild "
                    "vllm-ascend after sourcing MemFabric Hybrid 1.2.0."
                )

        store_url = os.getenv("MF_CONFIG_STORE_URL") or os.getenv("ASCEND_MF_STORE_URL")
        if not store_url:
            raise RuntimeError(
                "KVPP MTE requires MF_CONFIG_STORE_URL (or the deprecated ASCEND_MF_STORE_URL compatibility variable)."
            )
        store_url = _store_url_for_kvpp_group(store_url, self.group)
        staging_bytes = int(os.getenv("ASCEND_KVPP_MTE_STAGING_BYTES", _DEFAULT_STAGING_BYTES))
        if staging_bytes <= 0 or staging_bytes % _SHM_ALIGNMENT:
            raise ValueError(
                f"ASCEND_KVPP_MTE_STAGING_BYTES must be a positive multiple "
                f"of {_SHM_ALIGNMENT} bytes, got {staging_bytes}."
            )
        shm_id = int(os.getenv("ASCEND_KVPP_MTE_SHM_ID", _DEFAULT_SHM_ID))
        if not 0 <= shm_id < _SHM_ID_LIMIT:
            raise ValueError(f"ASCEND_KVPP_MTE_SHM_ID must be in [0, {_SHM_ID_LIMIT}), got {shm_id}.")
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
                raise RuntimeError(f"KVPP MemFabric global initialization failed: error={ret}.")
            self._global_initialized = True
        ret = self._shm_module.initialize(
            store_url,
            self.group.world_size,
            self.group.rank_in_group,
            device_id,
            config,
        )
        if ret != 0:
            raise RuntimeError(f"KVPP MemFabric SHM initialization failed: error={ret}.")
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
            raise RuntimeError(f"KVPP MemFabric SHM does not support MTE: reported operation={operation}.")
        # ``create`` publishes the symmetric address before every rank's
        # device-side mapping is necessarily ready. The SHM barrier completes
        # that setup; a process-group barrier is not an equivalent substitute.
        self._memory.barrier()

        self._layers = build_kvpp_layer_metadata(kv_caches, self.num_blocks)
        self._anchors = {layer_name: flatten_kvpp_cache(cache)[0] for layer_name, cache in kv_caches.items()}
        self._device_layers = {}
        for layer_name, buffers in self._layers.items():
            device = self._anchors[layer_name].device
            anchor_base = self._anchors[layer_name].data_ptr()
            self._device_layers[layer_name] = _MTEDeviceBufferMetadata(
                local_base_offsets=torch.tensor(
                    [buffer.base_addr - anchor_base for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_strides=torch.tensor(
                    [buffer.block_stride_bytes for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_bytes=torch.tensor(
                    [buffer.block_bytes for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                staging_bytes_per_slot=sum(buffer.block_bytes for buffer in buffers),
            )
        # ``gva`` is the common symmetric base. MemFabric may align each
        # rank's segment to an internal symmetric size larger than the local
        # contribution. That size is intentionally queried inside the
        # AscendC kernel; the Python binding does not expose it.
        self._local_metadata = KVPPMTEPeerMetadata(
            staging_addr=int(self._memory.gva),
            staging_bytes=staging_bytes,
            rank=self.group.rank_in_group,
        )
        peers: list[KVPPMTEPeerMetadata | None] = [None] * self.group.world_size
        dist.all_gather_object(peers, self._local_metadata, group=self.group.cpu_group)
        if any(peer is None for peer in peers):
            raise RuntimeError("KVPP MTE did not receive every peer GVA.")
        self._peer_metadata = [peer for peer in peers if peer is not None]
        logger.info(
            "KVPP MemFabric MTE initialized: rank=%d, gva=%#x, staging_bytes=%d, shm_id=%d, store_url=%s",
            self.group.rank_in_group,
            self._local_metadata.staging_addr,
            staging_bytes,
            shm_id,
            store_url,
        )

    def _local_descriptors(self, layer_name: str, pages: KVPPActivePages) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self._device_layers[layer_name]
        page_ids = pages.page_ids.to(dtype=torch.int64)
        offsets = metadata.local_base_offsets[:, None] + page_ids[None, :] * metadata.block_strides[:, None]
        lengths = torch.where(
            pages.valid_mask[None, :],
            metadata.block_bytes[:, None],
            torch.zeros((), dtype=torch.int64, device=page_ids.device),
        )
        return offsets.flatten(), lengths.flatten()

    def _bundle_staging_offsets(
        self,
        layer_names: tuple[str, ...],
        pages: KVPPActivePages,
        staging_bytes: int,
    ) -> dict[str, torch.Tensor]:
        """Lay out all caches used by one transformer layer in one segment.

        Main SFA and Lightning Indexer caches are pushed before the owner
        publishes the completion token.  Giving each cache an independent
        zero-based layout would let a later push overwrite an earlier one.
        Reserve capacity for every buffer in bundle order instead.
        """
        if not layer_names:
            raise ValueError("KVPP MTE cache bundle cannot be empty.")

        bytes_per_slot = sum(self._device_layers[layer_name].staging_bytes_per_slot for layer_name in layer_names)
        max_active_pages = staging_bytes // bytes_per_slot
        if max_active_pages == 0:
            raise RuntimeError(
                "KVPP MTE staging segment cannot hold one page for the full cache bundle: "
                f"page_bytes={bytes_per_slot}, capacity={staging_bytes}."
            )
        if pages.count_upper_bound > max_active_pages:
            raise RuntimeError(
                "KVPP MTE active-page upper bound exceeds bundle staging "
                f"capacity: upper_bound={pages.count_upper_bound}, "
                f"capacity={max_active_pages}, staging_bytes={staging_bytes}. "
                "Increase ASCEND_KVPP_MTE_STAGING_BYTES."
            )

        active_ordinals = torch.cumsum(pages.valid_mask.to(dtype=torch.int64), dim=0) - 1
        result: dict[str, torch.Tensor] = {}
        bundle_base: int | torch.Tensor = 0
        for layer_name in layer_names:
            metadata = self._device_layers[layer_name]
            per_buffer_capacity = metadata.block_bytes * max_active_pages
            buffer_offsets = torch.cumsum(per_buffer_capacity, dim=0)
            buffer_offsets = buffer_offsets - per_buffer_capacity
            buffer_offsets = buffer_offsets + bundle_base
            result[layer_name] = (
                buffer_offsets[:, None] + active_ordinals[None, :] * metadata.block_bytes[:, None]
            ).flatten()
            bundle_base += per_buffer_capacity.sum()
        return result

    def _launch(
        self,
        layer_name: str,
        descriptors: _MTEDeviceDescriptors,
        *,
        source_rank: int = -1,
        destination_rank: int = -1,
    ) -> None:
        assert self._copy_op is not None
        self._copy_op(
            self._anchors[layer_name],
            descriptors.local_offsets,
            descriptors.staging_offsets,
            descriptors.lengths,
            descriptors.staging_base,
            source_rank,
            destination_rank,
            self._shm_id,
        )

    def push_active_bundle(
        self,
        layer_names: tuple[str, ...],
        pages: KVPPActivePages,
        stream: Any,
    ) -> MemFabricMTECompletion:
        if not layer_names:
            raise ValueError("KVPP MTE cache bundle cannot be empty.")
        owner_rank = self.layer_owners[layer_names[0]]
        if owner_rank != self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None or not self._peer_metadata:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        retained: list[torch.Tensor] = []
        for peer_rank, peer in enumerate(self._peer_metadata):
            if peer_rank == owner_rank:
                continue
            bundle_offsets = self._bundle_staging_offsets(
                layer_names,
                pages,
                peer.staging_bytes,
            )
            for layer_name in layer_names:
                local_offsets, lengths = self._local_descriptors(layer_name, pages)
                staging_offsets = bundle_offsets[layer_name]
                descriptors = _MTEDeviceDescriptors(
                    local_offsets,
                    staging_offsets,
                    lengths,
                    peer.staging_addr,
                )
                self._launch(layer_name, descriptors, destination_rank=peer.rank)
                retained.extend(descriptors.tensors())
        return MemFabricMTECompletion.record(stream, tuple(retained))

    def receive_active_bundle(
        self,
        layer_names: tuple[str, ...],
        pages: KVPPActivePages,
        stream: Any,
    ) -> MemFabricMTECompletion:
        if not layer_names:
            raise ValueError("KVPP MTE cache bundle cannot be empty.")
        owner_rank = self.layer_owners[layer_names[0]]
        if owner_rank == self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        bundle_offsets = self._bundle_staging_offsets(
            layer_names,
            pages,
            self._local_metadata.staging_bytes,
        )
        retained: list[torch.Tensor] = []
        for layer_name in layer_names:
            local_offsets, lengths = self._local_descriptors(layer_name, pages)
            descriptors = _MTEDeviceDescriptors(
                local_offsets=local_offsets,
                staging_offsets=bundle_offsets[layer_name],
                lengths=lengths,
                staging_base=self._local_metadata.staging_addr,
            )
            self._launch(
                layer_name,
                descriptors,
                source_rank=self._local_metadata.rank,
            )
            retained.extend(descriptors.tensors())
        return MemFabricMTECompletion.record(stream, tuple(retained))

    def close(self) -> None:
        if self._memory is not None:
            self._memory.destroy()
            self._memory = None
        if self._global_initialized:
            assert self._shm_module is not None
            self._shm_module.uninitialize()
            assert self._memfabric_module is not None
            self._memfabric_module.uninitialize()
            self._global_initialized = False
        self._peer_metadata.clear()
        self._local_metadata = None
        self._layers.clear()
        self._anchors.clear()
        self._device_layers.clear()
