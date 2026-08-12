# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_ascend.utils import enable_sp, is_moe_model

FLASHCOMM_DENSE_TOKEN_THRESHOLD = 1000


def flashcomm_enabled(vllm_config: VllmConfig, num_tokens: int) -> bool:
    """Return whether FlashComm1 is active for this execution shape."""
    return enable_sp(vllm_config) and (is_moe_model(vllm_config) or num_tokens > FLASHCOMM_DENSE_TOKEN_THRESHOLD)


def all_gather_hidden_states(
    hidden_states: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Gather sequence-parallel output and remove transport padding."""
    gathered = tensor_model_parallel_all_gather(hidden_states, 0)
    return gathered[:num_tokens]


def all_gather_hidden_states_and_aux(
    hidden_states: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    num_tokens: int,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    if isinstance(hidden_states, tuple):
        return (
            all_gather_hidden_states(hidden_states[0], num_tokens),
            [all_gather_hidden_states(aux_hidden_state, num_tokens) for aux_hidden_state in hidden_states[1]],
        )
    return all_gather_hidden_states(hidden_states, num_tokens)
