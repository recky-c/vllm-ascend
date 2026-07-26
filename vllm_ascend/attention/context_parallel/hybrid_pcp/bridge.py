# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass

import torch
from vllm.distributed.parallel_state import get_pcp_group

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    HybridPCPBridgeViewProtocol,
)


@dataclass(frozen=True)
class HybridFAInputs:
    """FA-local operands plus global cache operands from one packed AG."""

    fa_operands: tuple[torch.Tensor, ...]
    global_prefill_operands: tuple[torch.Tensor, ...]


def _pack_operands(
    operands: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Size, ...]]:
    if not operands:
        raise ValueError("At least one bridge operand is required.")
    rows = operands[0].shape[0]
    dtype = operands[0].dtype
    device = operands[0].device
    if any(operand.shape[0] != rows or operand.dtype != dtype or operand.device != device for operand in operands):
        raise ValueError("Hybrid bridge operands must share row count, dtype, and device.")
    shapes = tuple(operand.shape[1:] for operand in operands)
    packed = torch.cat(
        tuple(operand.reshape(rows, -1) for operand in operands),
        dim=-1,
    )
    return packed, shapes


def _unpack_operands(
    packed: torch.Tensor,
    shapes: tuple[torch.Size, ...],
) -> tuple[torch.Tensor, ...]:
    widths = tuple(shape.numel() for shape in shapes)
    return tuple(tensor.reshape(packed.shape[0], *shape) for tensor, shape in zip(packed.split(widths, dim=-1), shapes))


@torch.compiler.disable
def enter_hybrid_fa(
    operands: tuple[torch.Tensor, ...],
    bridge_view: HybridPCPBridgeViewProtocol | None,
) -> HybridFAInputs:
    """Convert linear operands to the local DualChunk FA view.

    Decode rows are replicated and bypass communication. All prefill operands
    are packed into one payload, so each FA layer performs one entry AG.
    """

    if bridge_view is None:
        return HybridFAInputs(operands, ())
    layout = bridge_view.layout
    decode = layout.num_decode_tokens
    if not layout.has_prefill:
        return HybridFAInputs(operands, ())
    if operands[0].shape[0] != layout.linear_num_tokens_padded:
        raise ValueError(
            "Hybrid FA entry expected linear padded rows "
            f"{layout.linear_num_tokens_padded}, got {operands[0].shape[0]}."
        )

    packed, shapes = _pack_operands(operands)
    gathered_prefill = get_pcp_group().all_gather(
        packed[decode:].contiguous(),
        dim=0,
    )
    global_prefill = gathered_prefill[layout.hybrid_linear_ag_restore_idx]
    fa_prefill = global_prefill[layout.hybrid_global_to_fa_idx]

    if decode:
        fa_packed = torch.cat((packed[:decode], fa_prefill), dim=0)
    else:
        fa_packed = fa_prefill
    return HybridFAInputs(
        fa_operands=_unpack_operands(fa_packed, shapes),
        global_prefill_operands=_unpack_operands(global_prefill, shapes),
    )


@torch.compiler.disable
def exit_hybrid_fa(
    output: torch.Tensor,
    bridge_view: HybridPCPBridgeViewProtocol | None,
) -> torch.Tensor:
    """Convert a local DualChunk FA output back to the linear main layout."""

    if bridge_view is None:
        return output
    layout = bridge_view.layout
    decode = layout.num_decode_tokens
    if not layout.has_prefill:
        return output
    if output.shape[0] != layout.fa_num_tokens_padded:
        raise ValueError(
            f"Hybrid FA exit expected FA padded rows {layout.fa_num_tokens_padded}, got {output.shape[0]}."
        )

    gathered_prefill = get_pcp_group().all_gather(
        output[decode:].contiguous(),
        dim=0,
    )
    linear_prefill = gathered_prefill[layout.hybrid_fa_to_linear_idx]
    if decode:
        return torch.cat((output[:decode], linear_prefill), dim=0)
    return linear_prefill
