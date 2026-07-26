# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    HybridPCPForwardViewProtocol,
)


class HybridPCPTokenCompactor:
    """Own the compact/restore lifecycle for padded linear PCP inputs."""

    def __init__(self) -> None:
        self._valid_mask: torch.Tensor | None = None
        self._num_valid_tokens: int | None = None

    def compact_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        forward_view: HybridPCPForwardViewProtocol | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forward_view is None:
            self._valid_mask = None
            self._num_valid_tokens = None
            return hidden_states, router_logits

        valid_mask = forward_view.linear_valid_mask
        if hidden_states.shape[0] != valid_mask.shape[0]:
            raise RuntimeError(
                "Hybrid PCP MoE inputs do not match the linear forward "
                f"view: {hidden_states.shape[0]} != {valid_mask.shape[0]}."
            )
        if router_logits.shape[0] != valid_mask.shape[0]:
            raise RuntimeError(
                "Hybrid PCP router logits do not match the linear forward "
                f"view: {router_logits.shape[0]} != {valid_mask.shape[0]}."
            )
        self._valid_mask = valid_mask
        self._num_valid_tokens = forward_view.linear_num_tokens
        return hidden_states[valid_mask], router_logits[valid_mask]

    def compact_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        valid_mask = self._valid_mask
        if valid_mask is None:
            return tensor
        if tensor.shape[0] != valid_mask.shape[0]:
            raise RuntimeError(
                "Hybrid PCP auxiliary token tensor does not match the "
                f"linear forward view: {tensor.shape[0]} != "
                f"{valid_mask.shape[0]}."
            )
        return tensor[valid_mask]

    def restore_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        valid_mask = self._valid_mask
        num_valid_tokens = self._num_valid_tokens
        self._valid_mask = None
        self._num_valid_tokens = None
        if valid_mask is None:
            return hidden_states
        assert num_valid_tokens is not None
        if hidden_states.shape[0] != num_valid_tokens:
            raise RuntimeError(
                "Hybrid PCP MoE output does not match the compact token "
                f"count: {hidden_states.shape[0]} != {num_valid_tokens}."
            )
        restored = hidden_states.new_zeros((valid_mask.shape[0], *hidden_states.shape[1:]))
        restored[valid_mask] = hidden_states
        return restored


def mask_shared_expert_output(
    shared_output: torch.Tensor,
    forward_view: HybridPCPForwardViewProtocol | None,
) -> torch.Tensor:
    if forward_view is None:
        return shared_output
    valid_mask = forward_view.linear_valid_mask
    if shared_output.shape[0] != valid_mask.shape[0]:
        raise RuntimeError(
            "Hybrid PCP shared-expert output does not match the linear "
            f"forward view: {shared_output.shape[0]} != "
            f"{valid_mask.shape[0]}."
        )
    return shared_output.masked_fill(
        ~valid_mask.reshape(
            valid_mask.shape[0],
            *([1] * (shared_output.ndim - 1)),
        ),
        0,
    )
