from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch_npu
from torch import nn
from vllm.distributed import get_tp_group

from vllm_ascend.attention.sfa.metadata import DSACPContext
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.utils import _round_up

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa.impl import AscendSFAImpl
    from vllm_ascend.attention.sfa.metadata import M


class DSACPKVGatherHandle(NamedTuple):
    """In-flight DSA-CP KV all-gather state (wait + write happen later)."""

    fused_kv_no_split: torch.Tensor
    kv_ag_handle: torch.distributed.Work | None
    k_li: torch.Tensor | None
    k_li_scale: torch.Tensor | None


class FinishDSACPKVResult(NamedTuple):
    k_pe: torch.Tensor | None
    k_nope: torch.Tensor | None
    k_li: torch.Tensor | None
    k_li_scale: torch.Tensor | None
    o_proj_full_handle: torch.distributed.Work | None
    o_proj_full_param_handles: list[torch.distributed.Work | None] | None


def build_dsa_cp_context(
    num_input_tokens: int,
    num_actual_tokens: int,
    num_reqs: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    draft_index: int | None,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    spec_actual_seq_lengths_query: list[torch.Tensor] | None,
    spec_actual_seq_lengths_key: list[torch.Tensor] | None,
) -> tuple[DSACPContext, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_tp_size = get_tp_group().world_size
    num_tokens = num_input_tokens
    num_tokens_pad = _round_up(num_tokens, global_tp_size)
    num_tokens_per_device = num_tokens_pad // global_tp_size
    local_start = get_tp_group().rank_in_group * num_tokens_per_device
    local_end_with_pad = local_start + num_tokens_per_device
    local_end = min(local_end_with_pad, num_actual_tokens)

    pad_size = num_tokens_pad - cos.shape[0]
    assert cos.shape == sin.shape, f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

    if pad_size > 0:
        cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
        sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

    pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
    if pad_size_slot > 0:
        slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
    else:
        slot_mapping = slot_mapping[:num_tokens_pad]
    slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

    cos = cos[local_start:local_end_with_pad]
    sin = sin[local_start:local_end_with_pad]

    assert cos.shape[0] == num_tokens_per_device, (
        f"cos.shape[0] must be equal to num_tokens_per_device, \
            got {cos.shape[0]} and {num_tokens_per_device}"
    )
    assert slot_mapping_cp.shape[0] == num_tokens_per_device, (
        f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
            got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
    )
    assert slot_mapping.shape[0] == num_tokens_pad, (
        f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
            got {slot_mapping.shape[0]} and {num_tokens_pad}"
    )

    if draft_index is not None:
        assert spec_actual_seq_lengths_query is not None
        assert spec_actual_seq_lengths_key is not None
        # Per-draft-step buffers: independent, graph-stable storage so
        # later draft steps don't clobber earlier ones' metadata.
        actual_seq_lengths_query = spec_actual_seq_lengths_query[draft_index - 1]
        actual_seq_lengths_key = spec_actual_seq_lengths_key[draft_index - 1]

    num_segs = cum_query_lens.shape[0]

    # Vectorized per-request local query/key lengths for this rank's
    # [local_start, local_end_with_pad) slice. Replaces a Python loop
    # that did 2 .item() NPU->CPU syncs per request (2 * num_reqs
    # syncs/step); now fully on-device with zero syncs.
    # global_start[i] = 0 for i==0, else cum_query_lens[i-1]
    global_start = query_start_loc[:num_segs]
    global_end = cum_query_lens

    # Clip each request's [global_start, global_end) to the local range.
    # num_local_tokens may be < 0 when the request falls entirely
    # outside [local_start, local_end_with_pad); clamp before cumsum.
    req_local_start = global_start.clamp(min=local_start)
    req_local_end = global_end.clamp(max=local_end_with_pad)
    num_local_tokens = req_local_end - req_local_start

    local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
    offset = global_end - req_local_end  # request tokens on later ranks
    local_key_lens = torch.where(num_local_tokens > 0, seq_lens - offset, 0)

    actual_seq_lengths_query[:num_segs] = local_query_lens
    actual_seq_lengths_key[:num_segs] = local_key_lens
    actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
    actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

    dsa_cp_context = DSACPContext(
        num_tokens=num_tokens,
        num_tokens_pad=num_tokens_pad,
        local_start=local_start,
        local_end=local_end,
        local_end_with_pad=local_end_with_pad,
        slot_mapping_cp=slot_mapping_cp,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
    )
    return dsa_cp_context, cos, sin, slot_mapping


def start_o_proj_full_gather(
    host: AscendSFAImpl,
) -> tuple[torch.distributed.Work | None, list[torch.distributed.Work | None]]:
    _, o_proj_full_handle = all_gather_async(
        host.o_proj_tp_weight_gather_input,
        get_tp_group(),
        output=host.o_proj_full_gather_pool,
    )
    o_proj_full_param_handles: list[torch.distributed.Work | None] = []
    for param_name, param in host.o_proj_tp_input_sharded_quant_params.items():
        _, param_handle = all_gather_async(
            param,
            get_tp_group(),
            output=host.o_proj_full_input_sharded_quant_params[param_name],
        )
        o_proj_full_param_handles.append(param_handle)
    return o_proj_full_handle, o_proj_full_param_handles


def start_dsa_cp_kv_gather(
    host: AscendSFAImpl,
    k_pe: torch.Tensor,
    k_nope: torch.Tensor,
    knope_scale: torch.Tensor | None,
    k_li: torch.Tensor | None,
    k_li_scale: torch.Tensor | None,
    async_op: bool,
) -> DSACPKVGatherHandle:
    """Kick off fused KV (+ optional C8 indexer) all-gather; do not wait yet.

    Callers should run q_proj / rope between this and ``finish_dsa_cp_kv_gather``
    so communication can overlap with compute.
    """
    assert k_pe is not None
    assert k_nope is not None

    if host.use_sparse_c8_sfa:
        assert knope_scale is not None
        fused_kv_parts = [
            k_nope.view(-1, k_nope.shape[-1]),
            k_pe.view(-1, k_pe.shape[-1]),
            knope_scale.view(-1, knope_scale.shape[-1]),
        ]
    else:
        fused_kv_parts = [
            k_pe.view(-1, k_pe.shape[-1]),
            k_nope.view(-1, k_nope.shape[-1]),
        ]
        if host.has_indexer and not host.use_sparse_c8_indexer:
            assert k_li is not None
            fused_kv_parts.append(k_li.view(-1, k_li.shape[-1]))

    fused_kv_input = torch.cat(fused_kv_parts, dim=1)
    fused_kv_no_split, kv_ag_handle = all_gather_async(
        fused_kv_input,
        get_tp_group(),
        async_op=async_op,
    )

    if host.has_indexer and host.use_sparse_c8_indexer:
        assert k_li is not None
        assert k_li_scale is not None
        k_li, kv_ag_handle = all_gather_async(
            k_li,
            get_tp_group(),
            async_op=async_op,
        )
        k_li_scale, kv_ag_handle = all_gather_async(
            k_li_scale,
            get_tp_group(),
            async_op=async_op,
        )

    return DSACPKVGatherHandle(
        fused_kv_no_split=fused_kv_no_split,
        kv_ag_handle=kv_ag_handle,
        k_li=k_li,
        k_li_scale=k_li_scale,
    )


def finish_dsa_cp_kv_gather(
    host: AscendSFAImpl,
    gather_handle: DSACPKVGatherHandle,
    kv_cache: tuple[torch.Tensor, ...] | None,
    slot_mapping_sfa: torch.Tensor,
    attn_metadata: M,
    full_gather_o_proj_enabled: bool,
) -> FinishDSACPKVResult:
    """Wait for KV all-gather, optionally start o_proj gather, write KV cache."""
    fused_kv_no_split = gather_handle.fused_kv_no_split
    kv_ag_handle = gather_handle.kv_ag_handle
    k_li = gather_handle.k_li
    k_li_scale = gather_handle.k_li_scale
    k_pe: torch.Tensor | None = None
    k_nope: torch.Tensor | None = None
    o_proj_full_handle: torch.distributed.Work | None = None
    o_proj_full_param_handles: list[torch.distributed.Work | None] | None = None

    if kv_ag_handle is not None:
        kv_ag_handle.wait()

    if full_gather_o_proj_enabled:
        o_proj_full_handle, o_proj_full_param_handles = start_o_proj_full_gather(host)

    if kv_cache is not None:
        assert fused_kv_no_split is not None
        if host.use_sparse_c8_sfa:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(-1, fused_kv_no_split.shape[-1]),
                slot_mapping_sfa[: attn_metadata.num_actual_tokens].view(-1, 1),
                fused_kv_no_split[: attn_metadata.num_actual_tokens],
            )
            k_pe = None
            k_nope = None
        elif not host.has_indexer:
            k_pe, k_nope = fused_kv_no_split.split(
                [host.qk_rope_head_dim, host.kv_lora_rank],
                dim=-1,
            )
        elif not host.use_sparse_c8_indexer:
            k_pe, k_nope, k_li = fused_kv_no_split.split(
                [host.qk_rope_head_dim, host.kv_lora_rank, host.head_dim],
                dim=-1,
            )
        else:
            k_pe, k_nope = fused_kv_no_split.split(
                [host.qk_rope_head_dim, host.kv_lora_rank],
                dim=-1,
            )
        if not host.use_sparse_c8_sfa:
            assert k_pe is not None
            assert k_nope is not None
            k_nope = k_nope.view(k_nope.shape[0], 1, -1)
            k_pe = k_pe.view(k_pe.shape[0], 1, -1)
            DeviceOperator.reshape_and_cache(
                key=k_nope[: attn_metadata.num_actual_tokens],
                value=k_pe[: attn_metadata.num_actual_tokens],
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_mapping=slot_mapping_sfa[: attn_metadata.num_actual_tokens],
            )

    return FinishDSACPKVResult(
        k_pe=k_pe,
        k_nope=k_nope,
        k_li=k_li,
        k_li_scale=k_li_scale,
        o_proj_full_handle=o_proj_full_handle,
        o_proj_full_param_handles=o_proj_full_param_handles,
    )
