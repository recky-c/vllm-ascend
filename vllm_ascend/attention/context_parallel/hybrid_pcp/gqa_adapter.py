# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp.bridge import (
    enter_hybrid_fa,
    exit_hybrid_fa,
)
from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    HybridPCPBridgeViewProtocol,
)
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator

if TYPE_CHECKING:
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl,
        AscendMetadata,
    )


def forward_hybrid_pcp_gqa(
    impl: AscendAttentionBackendImpl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
    bridge_view: HybridPCPBridgeViewProtocol,
) -> torch.Tensor:
    """Run GQA over the FA view while writing the complete replicated cache."""

    bridge_inputs = enter_hybrid_fa((query, key, value), bridge_view)
    fa_query, fa_key, fa_value = bridge_inputs.fa_operands
    layout = bridge_view.layout

    if impl.key_cache is None:
        impl.key_cache, impl.value_cache = kv_cache[0], kv_cache[1]
    if impl.kv_sharing_target_layer_name is None:
        if layout.has_prefill:
            _, global_key, global_value = bridge_inputs.global_prefill_operands
            if layout.num_decode_tokens:
                cache_key = torch.cat(
                    (key[: layout.num_decode_tokens], global_key),
                    dim=0,
                )
                cache_value = torch.cat(
                    (value[: layout.num_decode_tokens], global_value),
                    dim=0,
                )
            else:
                cache_key = global_key
                cache_value = global_value
        else:
            cache_key = key[: layout.num_decode_tokens]
            cache_value = value[: layout.num_decode_tokens]

        cache_write_plan = getattr(
            attn_metadata,
            "hybrid_cache_write_plan",
            None,
        )
        if cache_write_plan is None:
            raise RuntimeError("Hybrid GQA attention requires a CacheWritePlan.")
        cache_slots = torch.cat(
            tuple(segment.effective_slot_mapping() for segment in cache_write_plan.segments),
            dim=0,
        )
        if cache_key.shape[0] != cache_slots.shape[0]:
            raise RuntimeError(
                f"Hybrid cache operands and slot mappings disagree: {cache_key.shape[0]} != {cache_slots.shape[0]}."
            )
        DeviceOperator.reshape_and_cache(
            key=cache_key,
            value=cache_value,
            key_cache=impl.key_cache,
            value_cache=impl.value_cache,
            slot_mapping=cache_slots,
        )
        notify_kv_cache_written()

    fa_output = output.new_empty((layout.fa_num_tokens_padded, *output.shape[1:]))
    fa_output = impl.forward_impl(
        fa_query,
        fa_key,
        fa_value,
        kv_cache,
        attn_metadata,
        fa_output,
    )
    linear_output = exit_hybrid_fa(fa_output, bridge_view)
    output[: linear_output.shape[0]].copy_(linear_output)
    return output
