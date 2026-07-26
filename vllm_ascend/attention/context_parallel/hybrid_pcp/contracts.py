# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch


class PCPInputLayout(str, Enum):
    """Token layout required by an attention/cache group."""

    DUAL_CHUNK_VIRTUAL = "dual_chunk_virtual"
    CONTIGUOUS_CAUSAL_STATE = "contiguous_causal_state"


class CacheInputDistribution(str, Enum):
    """Distribution of cache operands supplied to a backend."""

    LOCAL_FA = "local_fa"
    LOCAL_REPLICATED = "local_replicated"
    ALREADY_GLOBAL = "already_global"


class CacheReplication(str, Enum):
    """Physical cache ownership across the PCP group."""

    REPLICATED_PER_PCP_RANK = "replicated_per_pcp_rank"


@dataclass(frozen=True)
class PCPGroupCapability:
    """Static PCP strategy advertised by an attention metadata builder."""

    input_layout: PCPInputLayout
    cache_input_distributions: frozenset[CacheInputDistribution]
    cache_replication: CacheReplication = CacheReplication.REPLICATED_PER_PCP_RANK
    supports_piecewise: bool = True
    supports_chunked_prefill: bool = False
    supports_prefix_caching: bool = False
    supports_quantization: bool = False
    supports_sliding_window: bool = False

    def accepts(self, distribution: CacheInputDistribution) -> bool:
        return distribution in self.cache_input_distributions


def dual_chunk_pcp_capability() -> PCPGroupCapability:
    return PCPGroupCapability(
        input_layout=PCPInputLayout.DUAL_CHUNK_VIRTUAL,
        cache_input_distributions=frozenset(
            {
                CacheInputDistribution.LOCAL_FA,
                CacheInputDistribution.LOCAL_REPLICATED,
                CacheInputDistribution.ALREADY_GLOBAL,
            }
        ),
    )


def contiguous_state_pcp_capability() -> PCPGroupCapability:
    return PCPGroupCapability(
        input_layout=PCPInputLayout.CONTIGUOUS_CAUSAL_STATE,
        # GDN/Mamba owns request state rather than a token KV cache. Its PCP
        # state hand-off is implemented by the existing GDN seam, so no
        # CacheWritePlan distribution applies to this group.
        cache_input_distributions=frozenset(),
    )


class HybridPCPLayoutProtocol(Protocol):
    num_decode_tokens: int
    global_num_tokens: int
    linear_num_tokens_padded: int
    fa_num_tokens_padded: int
    hybrid_linear_ag_restore_idx: torch.Tensor
    hybrid_global_to_fa_idx: torch.Tensor
    hybrid_fa_to_linear_idx: torch.Tensor

    @property
    def has_prefill(self) -> bool: ...


class HybridPCPBridgeViewProtocol(Protocol):
    layout: HybridPCPLayoutProtocol


class HybridPCPForwardViewProtocol(Protocol):
    linear_num_tokens: int
    linear_num_tokens_padded: int
    linear_valid_mask: torch.Tensor
