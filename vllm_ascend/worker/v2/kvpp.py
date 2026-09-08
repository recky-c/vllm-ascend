# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import build_kvpp_layer_layout, create_kvpp_cache_allocation_plan
from vllm_ascend.distributed.kv_transfer.kv_pool.broadcast_transport import BroadcastKVPPTransport
from vllm_ascend.distributed.parallel_state import get_kvpp_group
from vllm_ascend.worker.kvpp_cache import get_kvpp_cache_specs


@dataclass(frozen=True)
class KVPPCacheLayout:
    layer_caches: dict[str, Any]


class KVPPRuntime:
    """Bind contiguous cache storage to the shared layer prefetch scheduler."""

    def __init__(self, scheduler: KVPPScheduler | None = None) -> None:
        self.scheduler = scheduler

    @classmethod
    def create_from_kv_cache(
        cls, *, vllm_config: Any, kv_cache_config: Any, static_forward_context: dict[str, Any]
    ) -> KVPPRuntime:
        if KVPPConfig.from_vllm_config(vllm_config).size <= 1:
            return cls()
        caches = {
            name: static_forward_context[name].kv_cache
            for group in kv_cache_config.kv_cache_groups
            for name in group.layer_names
        }
        return cls.create_from_cache_layout(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            static_forward_context=static_forward_context,
            cache_layout=KVPPCacheLayout(caches),
        )

    @classmethod
    def create_from_cache_layout(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        cache_layout: KVPPCacheLayout,
    ) -> KVPPRuntime:
        config = KVPPConfig.from_vllm_config(vllm_config)
        if config.size <= 1:
            return cls()
        group = get_kvpp_group()
        plan = create_kvpp_cache_allocation_plan(
            vllm_config, get_kvpp_cache_specs(kv_cache_config), group.rank_in_group
        )
        if not plan.layer_owner_ranks:
            return cls()
        buffers = {}
        impls = {}
        signature = []
        for name, bundle in plan.layer_bundles.items():
            if name not in plan.layer_owner_ranks:
                continue
            layout, size = build_kvpp_layer_layout(bundle, plan.tensor_specs, kv_cache_config.num_blocks)
            first = cache_layout.layer_caches[name][0]
            storage = first.untyped_storage()
            base = first.storage_offset() * first.element_size()
            if base + size > storage.nbytes():
                raise ValueError(f"KVPP layer span exceeds its storage: {name}.")
            parts = []
            for cache_name, offsets in layout.items():
                tensors = cache_layout.layer_caches[cache_name]
                if len(tensors) != len(offsets):
                    raise ValueError(f"KVPP cache component count differs for {cache_name}.")
                for tensor, (offset, length) in zip(tensors, offsets):
                    if (
                        tensor.untyped_storage().data_ptr() != storage.data_ptr()
                        or tensor.storage_offset() * tensor.element_size() != base + offset
                        or tensor.numel() * tensor.element_size() != length
                    ):
                        raise ValueError(f"KVPP cache is not a contiguous layer bundle: {cache_name}.")
                    parts.append((offset, length))
            raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(storage, base, (size,), (1,))
            buffers[name] = (
                (raw,)
                if config.broadcast_granularity == "layer"
                else tuple(raw.narrow(0, offset, length) for offset, length in parts)
            )
            impl = static_forward_context[name].impl
            if not hasattr(impl, "layerwise_kv_cache_hook"):
                raise TypeError(f"KVPP requires an attention cache hook: {name}.")
            impls[name] = impl
            signature.append((name, plan.layer_owner_ranks[name], kv_cache_config.num_blocks, layout, size))
        signatures = [None] * group.world_size
        dist.all_gather_object(signatures, signature, group=group.cpu_group)
        if any(value != signature for value in signatures):
            raise ValueError("KVPP ranks have different layer layouts or block counts.")
        scheduler = KVPPScheduler(BroadcastKVPPTransport(group, plan.layer_owner_ranks, buffers), tuple(buffers))
        for impl in impls.values():
            impl.layerwise_kv_cache_hook = scheduler
        return cls(scheduler)

    def prepare_forward(self, has_history: bool) -> None:
        if self.scheduler is not None:
            self.scheduler.schedule_forward(has_history)

    def complete_forward(self) -> None:
        if self.scheduler is not None:
            self.scheduler.complete_forward()

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()


class KVPPScheduler:
    """Prefetch one layer ahead; Target execution ordinal selects scratch."""

    def __init__(self, transport: BroadcastKVPPTransport, attention_layer_names: tuple[str, ...]) -> None:
        self.transport = transport
        self.attention_layer_names = attention_layer_names
        self._has_history = False
        self._next_attention_layer_index = 0
        self._prefetch_future: Future[None] | None = None
        self._npu_device_id = torch.npu.current_device()
        self._kv_transfer_stream = torch.npu.Stream()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpp-prefetch")

    def schedule_forward(self, has_history: bool) -> None:
        if self._prefetch_future is not None:
            raise RuntimeError("KVPP previous prefetch must complete before the next forward.")
        self._has_history = has_history
        self._next_attention_layer_index = 0
        if has_history:
            self.start_layer_prefetch(self.attention_layer_names[0])

    def start_layer_prefetch(self, layer_name: str) -> None:
        cache_ready = torch.npu.Event()
        cache_ready.record(torch.npu.current_stream())
        self._prefetch_future = self._prefetch_executor.submit(self.run_layer_prefetch, layer_name, cache_ready)

    def run_layer_prefetch(self, layer_name: str, cache_ready: Any) -> None:
        torch.npu.set_device(self._npu_device_id)
        self.transport.prefetch(layer_name, cache_ready, self._kv_transfer_stream)

    def wait_for_layer(self, layer_name: str) -> None:
        if not self._has_history:
            return
        if self._prefetch_future is None:
            raise RuntimeError("KVPP has no pending layer prefetch.")
        self._prefetch_future.result()
        self._prefetch_future = None
        self._next_attention_layer_index += 1
        if self._next_attention_layer_index < len(self.attention_layer_names):
            self.start_layer_prefetch(self.attention_layer_names[self._next_attention_layer_index])

    def complete_forward(self) -> None:
        try:
            if self._prefetch_future is not None:
                self._prefetch_future.result()
        finally:
            self._prefetch_future = None
            self._has_history = False
            self._next_attention_layer_index = 0

    def close(self) -> None:
        try:
            self.complete_forward()
        finally:
            self._prefetch_executor.shutdown(wait=True)
