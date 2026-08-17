# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class KVPPConfig:
    """Ascend-owned configuration for KV layer parallelism."""

    size: int = 1

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "KVPPConfig":
        additional_config = vllm_config.additional_config or {}
        size = additional_config.get("kvpp_size", 1)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(
                "additional_config.kvpp_size must be a positive integer, "
                f"got {size!r}."
            )
        return cls(size=size)

    def validate(self, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        if not vllm_config.use_v2_model_runner:
            raise ValueError("KVPP is only supported by Ascend Model Runner V2.")
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError("KVPP does not support pipeline parallelism yet.")
        if parallel_config.prefill_context_parallel_size != 1:
            raise ValueError("KVPP does not support PCP yet.")
        if parallel_config.decode_context_parallel_size != 1:
            raise ValueError("KVPP and DCP cannot be enabled at the same time.")
        if parallel_config.tensor_parallel_size % self.size != 0:
            raise ValueError(
                "tensor_parallel_size must be divisible by kvpp_size, got "
                f"TP={parallel_config.tensor_parallel_size}, KVPP={self.size}."
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("KVPP does not support KV connectors or offload yet.")

        model_config = vllm_config.model_config
        if not model_config.use_mla or model_config.is_hybrid:
            raise ValueError("KVPP currently supports only non-hybrid MLA models.")
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            if speculative_config.method != "mtp":
                raise ValueError(
                    "KVPP currently supports speculative decoding only with "
                    "method='mtp'."
                )
            if getattr(
                speculative_config,
                "num_speculative_tokens_per_batch_size",
                None,
            ):
                raise ValueError(
                    "KVPP currently supports only a fixed number of MTP "
                    "speculative tokens."
                )
