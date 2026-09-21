# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_KVPP_COMPATIBLE_CONNECTORS = frozenset(
    {
        "MooncakeConnectorV2",
        "MooncakePullConnector",
    }
)


# Host -> owner is scheduled one layer ahead of owner -> peers.
KVPP_OFFLOAD_PREFETCH_DISTANCE = 2
KVPP_OFFLOAD_MIN_BUFFERS = KVPP_OFFLOAD_PREFETCH_DISTANCE + 1


def get_kvpp_offload_config(vllm_config: "VllmConfig") -> dict[str, Any] | None:
    """Return the direct Memcache layerwise configuration, if enabled."""
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None or transfer.kv_connector != "AscendStoreConnector":
        return None
    extra = transfer.kv_connector_extra_config or {}
    if str(extra.get("backend", "mooncake")).lower() != "memcache" or not extra.get("use_layerwise", False):
        return None
    return extra


@dataclass(frozen=True)
class KVPPConfig:
    """Ascend-owned configuration for KV layer parallelism."""

    size: int = 1

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "KVPPConfig":
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        size = additional_config.get("kvpp_size", 1)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"additional_config.kvpp_size must be a positive integer, got {size!r}.")
        return cls(size=size)

    def validate(self, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size != 1:
            raise ValueError("KVPP does not support PCP yet.")
        if parallel_config.decode_context_parallel_size != 1:
            raise ValueError("KVPP and DCP cannot be enabled at the same time.")
        if parallel_config.tensor_parallel_size % self.size != 0:
            raise ValueError(
                "tensor_parallel_size must be divisible by kvpp_size, got "
                f"TP={parallel_config.tensor_parallel_size}, KVPP={self.size}."
            )
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            connector = kv_transfer_config.kv_connector
            role = kv_transfer_config.kv_role
            offload = get_kvpp_offload_config(vllm_config)
            if (connector not in _KVPP_COMPATIBLE_CONNECTORS and offload is None) or role != "kv_producer":
                raise ValueError(
                    "KVPP supports KV transfer only with adapted Mooncake connectors or Memcache layerwise AscendStore "
                    f"on a kv_producer, got connector={connector!r}, role={role!r}."
                )

        if get_kvpp_offload_config(vllm_config) is not None:
            if vllm_config.use_v2_model_runner or not vllm_config.model_config.enforce_eager:
                raise ValueError("KVPP layerwise offload requires Model Runner V1 and eager execution.")
            if parallel_config.pipeline_parallel_size != 1 or vllm_config.speculative_config is not None:
                raise ValueError("KVPP layerwise offload currently requires PP=1 and no speculative decoding.")
            offload = get_kvpp_offload_config(vllm_config)
            assert offload is not None
            for name in ("layerwise_num_shared_buffers", "layerwise_prefetch_layers"):
                value = offload.get(name, KVPP_OFFLOAD_MIN_BUFFERS)
                if isinstance(value, bool) or not isinstance(value, int) or value < KVPP_OFFLOAD_MIN_BUFFERS:
                    raise ValueError(f"KVPP layerwise offload requires {name} >= {KVPP_OFFLOAD_MIN_BUFFERS}.")
            if offload.get("layerwise_prefetch_layers", KVPP_OFFLOAD_MIN_BUFFERS) != KVPP_OFFLOAD_MIN_BUFFERS:
                raise ValueError("KVPP layerwise offload currently requires layerwise_prefetch_layers=3.")
            if "layerwise_independent_layers" in offload:
                raise ValueError("KVPP layerwise offload manages its own owner and peer buffer layout.")

        model_config = vllm_config.model_config
        if not model_config.use_mla or model_config.is_hybrid:
            raise ValueError("KVPP currently supports only non-hybrid MLA models.")
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            if speculative_config.method != "mtp":
                raise ValueError("KVPP currently supports speculative decoding only with method='mtp'.")
            if getattr(speculative_config, "num_speculative_tokens_per_batch_size", None):
                raise ValueError("KVPP currently supports only a fixed number of MTP speculative tokens.")
