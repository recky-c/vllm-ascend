# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.worker.v2.kvpp import KVPPCacheLayout, KVPPRuntime


class KVPPV1Runtime:
    """Model Runner V1 adapter around the shared KVPP scheduler."""

    def __init__(self, runtime: KVPPRuntime | None = None) -> None:
        self._kvpp_runtime = runtime if runtime is not None else KVPPRuntime()

    @classmethod
    def create_from_kv_cache(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any],
    ) -> KVPPV1Runtime:
        if KVPPConfig.from_vllm_config(vllm_config).size <= 1:
            return cls()

        return cls(
            KVPPRuntime.create_from_cache_layout(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                static_forward_context=static_forward_context,
                cache_layout=KVPPCacheLayout(
                    layer_caches=kv_caches,
                ),
            ),
        )

    def prepare_forward(self, has_history: bool) -> None:
        self._kvpp_runtime.prepare_forward(has_history)

    def complete_forward(self) -> None:
        self._kvpp_runtime.complete_forward()

    def close(self) -> None:
        self._kvpp_runtime.close()
