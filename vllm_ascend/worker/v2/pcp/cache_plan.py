# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    CacheInputDistribution,
    PCPGroupCapability,
)
from vllm_ascend.worker.v2.pcp.contracts import (
    CacheWritePlan,
    CacheWriteSegment,
    CacheWriteSegmentKind,
    HybridPCPLayout,
)


def build_fa_cache_write_plan(
    *,
    group_id: int,
    capability: PCPGroupCapability,
    layout: HybridPCPLayout,
    global_slot_mapping: torch.Tensor,
) -> CacheWritePlan:
    for distribution in (
        CacheInputDistribution.LOCAL_REPLICATED,
        CacheInputDistribution.ALREADY_GLOBAL,
    ):
        if not capability.accepts(distribution):
            raise NotImplementedError(
                "Hybrid FA group does not accept the required cache input "
                f"distribution {distribution.value}: group={group_id}."
            )

    segments: list[CacheWriteSegment] = []
    decode = layout.num_decode_tokens
    global_tokens = layout.global_num_tokens
    if decode:
        segments.append(
            CacheWriteSegment(
                kind=CacheWriteSegmentKind.DECODE,
                start=0,
                stop=decode,
                distribution=CacheInputDistribution.LOCAL_REPLICATED,
                slot_mapping=global_slot_mapping[:decode],
            )
        )
    if global_tokens > decode:
        segments.append(
            CacheWriteSegment(
                kind=CacheWriteSegmentKind.PREFILL,
                start=decode,
                stop=global_tokens,
                distribution=CacheInputDistribution.ALREADY_GLOBAL,
                slot_mapping=global_slot_mapping[decode:global_tokens],
            )
        )
    return CacheWritePlan(group_id=group_id, segments=tuple(segments))


def build_dummy_fa_cache_write_plan(
    *,
    group_id: int,
    slot_mapping: torch.Tensor,
    device: torch.device,
) -> CacheWritePlan:
    num_tokens = slot_mapping.shape[0]
    return CacheWritePlan(
        group_id=group_id,
        segments=(
            CacheWriteSegment(
                kind=CacheWriteSegmentKind.DECODE,
                start=0,
                stop=num_tokens,
                distribution=CacheInputDistribution.LOCAL_REPLICATED,
                slot_mapping=slot_mapping,
                valid_mask=torch.zeros(
                    num_tokens,
                    dtype=torch.bool,
                    device=device,
                ),
            ),
        ),
    )
