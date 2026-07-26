# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    PCPGroupCapability,
)


def resolve_group_capabilities(
    attn_groups: list[list[AttentionGroup]],
) -> tuple[PCPGroupCapability, ...]:
    """Resolve and validate one immutable capability per KV-cache group."""

    capabilities: list[PCPGroupCapability] = []
    for group_id, backend_groups in enumerate(attn_groups):
        group_capabilities: list[PCPGroupCapability] = []
        for backend_group in backend_groups:
            for builder in backend_group.metadata_builders:
                get_capability = getattr(
                    builder,
                    "get_pcp_group_capability",
                    None,
                )
                if get_capability is None:
                    raise NotImplementedError(
                        "Attention metadata builder does not advertise a PCP "
                        f"group capability: group={group_id}, "
                        f"builder={type(builder).__name__}."
                    )
                capability = get_capability()
                if not isinstance(capability, PCPGroupCapability):
                    raise TypeError(
                        f"get_pcp_group_capability() must return PCPGroupCapability, got {type(capability).__name__}."
                    )
                group_capabilities.append(capability)

        if not group_capabilities:
            raise RuntimeError(f"KV cache group {group_id} has no metadata builders.")
        capability = group_capabilities[0]
        if any(item != capability for item in group_capabilities[1:]):
            raise NotImplementedError(
                "All builders in one KV cache group must advertise the same "
                f"PCP capability; group {group_id} is heterogeneous."
            )
        capabilities.append(capability)
    return tuple(capabilities)
