# SPDX-License-Identifier: Apache-2.0
"""Transport boundary for KV layer parallelism.

KVPP scheduling owns page selection, layer ordering, scratch lifetime, and
completion synchronization. The MTE transport owns initialization and the
stream-ordered movement of selected physical pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class KVPPBufferMetadata:
    """Address and logical-page layout for one KV cache tensor."""

    base_addr: int
    block_stride_bytes: int
    block_bytes: int


@dataclass(frozen=True)
class KVPPActivePages:
    """Fixed-shape device representation of active physical KV pages.

    ``page_ids`` and ``valid_mask`` stay on the same device as vLLM's original
    block table.  Duplicate and invalid table entries remain as masked slots,
    so transports never need to read a dynamic page count back to the host.
    ``count_upper_bound`` is derived only from the host-resident sequence
    lengths and bounds the number of valid, unique pages without a device
    reduction.
    """

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


@runtime_checkable
class KVPPCompletion(Protocol):
    """Completion of one owner-to-peer active-page push.

    ``wait`` must not return until the destination pages are remotely visible.
    ``wait_on_stream`` expresses a local device dependency when the backend can
    do so without blocking the host.  Cross-rank notification remains the
    responsibility of the common KVPP execution layer.
    """

    def wait(self) -> None:
        """Block the caller until all remote destinations are visible."""

    def wait_on_stream(self, stream: Any) -> None:
        """Order a local device stream after this transfer."""


def flatten_kvpp_cache(cache: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(cache, torch.Tensor):
        return (cache,)
    if isinstance(cache, (tuple, list)):
        tensors = tuple(cache)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("KVPP cache tuples may contain only tensors.")
        return tensors
    raise TypeError(f"Unsupported KVPP cache type: {type(cache)!r}.")


def build_kvpp_layer_metadata(
    kv_caches: dict[str, Any], num_blocks: int
) -> dict[str, tuple[KVPPBufferMetadata, ...]]:
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
            block_stride_bytes = (
                tensor.stride(0) * tensor.element_size() * block_size_scale
            )
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


@runtime_checkable
class KVPPTransport(Protocol):
    """Backend-neutral data-plane contract consumed by the KVPP scheduler."""

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        """Prepare transport resources for all persistent and scratch caches."""

    def push_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """Enqueue selected pages and return remote-visible completion."""

    def receive_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """Unpack MTE staging into the original physical page IDs."""

    def close(self) -> None:
        """Release backend-owned sessions and memory metadata."""
