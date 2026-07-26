# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Backend-facing Hybrid PCP contracts and adapters."""

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    CacheInputDistribution,
    CacheReplication,
    PCPGroupCapability,
    PCPInputLayout,
    contiguous_state_pcp_capability,
    dual_chunk_pcp_capability,
)

__all__ = [
    "CacheInputDistribution",
    "CacheReplication",
    "PCPGroupCapability",
    "PCPInputLayout",
    "contiguous_state_pcp_capability",
    "dual_chunk_pcp_capability",
]
