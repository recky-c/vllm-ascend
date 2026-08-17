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
