# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm_ascend.core.kv_cache_placement import _get_replicated_mtp_layers
from vllm_ascend.worker.v2.kvpp import KVPPRuntime


def validate_v1_mtp_layers(
    vllm_config: Any,
    layer_names: tuple[str, ...],
    layer_owners: dict[str, int],
) -> None:
    """Reject MTP caches that this PP rank would treat as KVPP-owned.

    Presence is taken from the worker-local cache names, not from whether
    this rank is the last pipeline stage.
    """
    local_mtp_layers = _get_replicated_mtp_layers(vllm_config, layer_names)
    owned_mtp_layers = local_mtp_layers & set(layer_owners)
    if owned_mtp_layers:
        raise RuntimeError(
            "MTP attention layers must be replicated outside KVPP, "
            "but these layers have KVPP owners: "
            f"{sorted(owned_mtp_layers)}."
        )


class KVPPV1Runtime:
    """Model Runner V1 adapter around the shared KVPP scheduler."""

    def __init__(self, runtime: KVPPRuntime | None = None) -> None:
        self._runtime = runtime if runtime is not None else KVPPRuntime()

    @classmethod
    def try_build(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        kv_caches: dict[str, Any],
        block_tables: Any,
    ) -> KVPPV1Runtime:
        placement = KVPPRuntime._placement(vllm_config, kv_cache_config)
        if placement is None:
            return cls()
        layer_names, layer_owners, cache_group_index = placement
        validate_v1_mtp_layers(vllm_config, layer_names, layer_owners)
        block_table = block_tables[cache_group_index]
        return cls(
            KVPPRuntime._assemble(
                layer_owners=layer_owners,
                cache_group_index=cache_group_index,
                static_forward_context=static_forward_context,
                kv_caches={name: kv_caches[name] for name in layer_owners},
                num_kernel_blocks=(kv_cache_config.num_blocks * block_table.blocks_per_phys_block),
                block_size=block_table.logical_block_size,
            )
        )

    def begin_forward(
        self,
        input_batch: Any,
        num_reqs: int,
        seq_lens: Any,
    ) -> None:
        runtime = self._runtime
        if runtime.scheduler is None:
            return
        assert runtime.cache_group_index is not None
        block_table = input_batch.block_table[runtime.cache_group_index]
        runtime.scheduler.begin_forward(
            block_table.get_device_tensor(num_reqs),
            seq_lens,
        )

    def finish_forward(self) -> None:
        self._runtime.finish_forward()

    def close(self) -> None:
        self._runtime.close()
