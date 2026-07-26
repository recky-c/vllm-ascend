# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

from vllm_ascend.attention.context_parallel.hybrid_pcp.bridge import (
    enter_hybrid_fa,
    exit_hybrid_fa,
)
from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    HybridPCPBridgeViewProtocol,
)
from vllm_ascend.attention.utils import (
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if TYPE_CHECKING:
    from vllm_ascend.attention.mla_v1 import (
        AscendMLAImpl,
        AscendMLAMetadata,
        DecodeMLAPreprocessResult,
        PrefillMLAPreprocessResult,
    )


def _project_hybrid_mla_inputs(
    impl: AscendMLAImpl,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl.fused_qkv_a_proj is not None:
        qkv_lora = impl.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [
                impl.q_lora_rank,
                impl.kv_lora_rank + impl.qk_rope_head_dim,
            ],
            dim=-1,
        )
        q_c = impl.q_a_layernorm(q_c)
        return q_c, kv_no_split.contiguous()
    q_c = hidden_states
    kv_no_split = impl.kv_a_proj_with_mqa(hidden_states)[0]
    return q_c, kv_no_split


def preprocess_hybrid_pcp_mla(
    impl: AscendMLAImpl,
    layer_name: str,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: AscendMLAMetadata,
    bridge_view: HybridPCPBridgeViewProtocol,
) -> tuple[
    DecodeMLAPreprocessResult | None,
    PrefillMLAPreprocessResult | None,
]:
    """Project MLA inputs, enter the FA view, and write global cache rows."""

    # Imported lazily to keep this adapter below the shared MLA module without
    # creating an import cycle during backend registration.
    from vllm_ascend.attention.mla_v1 import (
        DecodeMLAPreprocessResult,
        PrefillMLAPreprocessResult,
    )

    layout = bridge_view.layout
    q_c, kv_no_split = _project_hybrid_mla_inputs(impl, hidden_states)
    cos, sin = get_cos_and_sin_mla(positions[: layout.linear_num_tokens_padded].long())
    bridge_inputs = enter_hybrid_fa(
        (q_c, kv_no_split, cos, sin),
        bridge_view,
    )
    fa_q_c, fa_kv, fa_cos, fa_sin = bridge_inputs.fa_operands

    cache_write_plan = getattr(
        attn_metadata,
        "hybrid_cache_write_plan",
        None,
    )
    if cache_write_plan is None:
        raise RuntimeError("Hybrid MLA attention requires a CacheWritePlan.")
    cache_slots = torch.cat(
        tuple(segment.effective_slot_mapping() for segment in cache_write_plan.segments),
        dim=0,
    )
    if cache_slots.shape[0] != layout.global_num_tokens:
        raise RuntimeError(
            "Hybrid MLA cache slots do not match the global token count: "
            f"{cache_slots.shape[0]} != {layout.global_num_tokens}."
        )
    decode = layout.num_decode_tokens
    if layout.has_prefill:
        (
            global_prefill_q,
            global_prefill_kv,
            global_cos,
            global_sin,
        ) = bridge_inputs.global_prefill_operands
        del global_prefill_q
    else:
        global_prefill_kv = fa_kv.new_empty((0, *fa_kv.shape[1:]))
        global_cos = fa_cos.new_empty((0, *fa_cos.shape[1:]))
        global_sin = fa_sin.new_empty((0, *fa_sin.shape[1:]))

    decode_preprocess_res = None
    if decode:
        decode_q_c = fa_q_c[:decode]
        decode_ql_nope, decode_q_pe = impl._q_proj_and_k_up_proj(decode_q_c)
        decode_ql_nope, decode_q_pe = impl.reorg_decode_q(
            decode_ql_nope,
            decode_q_pe,
        )
        decode_q_pe = impl.rope_single(
            decode_q_pe,
            fa_cos[:decode],
            fa_sin[:decode],
        )
        dequant_scale_q_nope = None
        if impl.fa_quant_layer and get_ascend_device_type() == AscendDeviceType.A5:
            (
                decode_ql_nope,
                dequant_scale_q_nope,
            ) = torch_npu.npu_dynamic_quant(
                decode_ql_nope,
                dst_type=torch.float8_e4m3fn,
            )
            decode_q_pe = (decode_q_pe / dequant_scale_q_nope.unsqueeze(-1) / impl.fak_descale_float).to(torch.bfloat16)
        decode_k_pe, decode_k_nope = impl.exec_kv_decode(
            fa_kv[:decode],
            fa_cos[:decode],
            fa_sin[:decode],
            kv_cache,
            cache_slots[:decode],
        )
        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_ql_nope,
            decode_q_pe,
            decode_k_nope,
            decode_k_pe,
            dequant_scale_q_nope=dequant_scale_q_nope,
        )

    prefill_preprocess_res = None
    num_prefill_tokens = attn_metadata.num_actual_tokens - decode
    if num_prefill_tokens:
        wait_for_kv_layer_from_connector(layer_name)
        global_k_pe, global_k_c_normed = impl.exec_kv_prefill(
            global_prefill_kv,
            global_cos,
            global_sin,
            kv_cache,
            cache_slots[decode:],
        )
        fa_indices = layout.hybrid_global_to_fa_idx[:num_prefill_tokens]
        prefill_k_pe = global_k_pe[fa_indices]
        prefill_k_c_normed = global_k_c_normed[fa_indices]
        prefill_q_c = fa_q_c[decode : attn_metadata.num_actual_tokens]
        prefill_q = impl.q_proj(prefill_q_c)[0].view(
            -1,
            impl.num_heads,
            impl.qk_head_dim,
        )
        prefill_q_nope, prefill_q_pe = prefill_q.split(
            [impl.qk_nope_head_dim, impl.qk_rope_head_dim],
            dim=-1,
        )
        prefill_q_pe = impl.rope_single(
            prefill_q_pe,
            fa_cos[decode : attn_metadata.num_actual_tokens],
            fa_sin[decode : attn_metadata.num_actual_tokens],
        )
        prefill_k_nope, prefill_value = (
            impl.kv_b_proj(prefill_k_c_normed)[0]
            .view(
                -1,
                impl.num_heads,
                impl.qk_nope_head_dim + impl.v_head_dim,
            )
            .split(
                [impl.qk_nope_head_dim, impl.v_head_dim],
                dim=-1,
            )
        )
        prefill_k_pe = prefill_k_pe.view(
            prefill_q_c.shape[0],
            impl.num_kv_heads,
            -1,
        ).expand((*prefill_k_nope.shape[:-1], -1))
        prefill_preprocess_res = PrefillMLAPreprocessResult(
            prefill_q_nope,
            prefill_q_pe,
            prefill_k_nope,
            prefill_k_pe,
            prefill_value,
        )

    notify_kv_cache_written(layer_name)
    return decode_preprocess_res, prefill_preprocess_res


def restore_hybrid_pcp_mla_output(
    output: torch.Tensor,
    linear_output: torch.Tensor,
    bridge_view: HybridPCPBridgeViewProtocol,
) -> torch.Tensor:
    linear_result = exit_hybrid_fa(output, bridge_view)
    linear_output[: linear_result.shape[0]].copy_(linear_result)
    return linear_output
