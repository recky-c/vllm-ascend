# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Hand a PCP-sharded target step to a replicated, PCP-unaware draft."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState


class TargetOutputRestorer(Protocol):
    """The target sharding primitives the handoff relies on."""

    @property
    def global_batch(self) -> InputBatch: ...

    @property
    def is_last_pp_rank(self) -> bool: ...

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def restore_hidden_state_buffer(self, buffer: torch.Tensor) -> None: ...


@runtime_checkable
class ProvidesMTPTargetHiddenStates(Protocol):
    """A target whose draft reads a model-owned buffer (DeepSeek V4 pre-hc_head residual)."""

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None: ...


class ReplicatedDraftHandoff:
    """Turns the PCP-local target step into the global step the draft consumes.

    The draft runs on every PCP rank with the full batch, so nothing that
    leaves the target step may stay in the rank-local layout.
    """

    def __init__(self, restorer: TargetOutputRestorer) -> None:
        self._restorer = restorer
        self._restored_outputs: tuple[torch.Tensor, list[torch.Tensor] | None] | None = None

    def to_draft_step(
        self,
        state: "ExecuteModelState",
        model_hidden_buffer: torch.Tensor | None,
    ) -> "ExecuteModelState":
        # Rank-local attention metadata and DP shape do not describe the
        # global batch; the draft builds and synchronizes its own.
        state = state._replace(attn_metadata=None, slot_mappings_by_layer=None, dp_sync=None)
        if state.input_batch.is_dummy:
            return state
        if not self._restorer.is_last_pp_rank:
            # PP receive/postprocess must use the global request layout.
            return state._replace(input_batch=self._restorer.global_batch)
        if model_hidden_buffer is not None:
            # In place: the draft graph reads this buffer at a fixed address.
            self._restorer.restore_hidden_state_buffer(model_hidden_buffer)
        hidden_states = self._restorer.restore_hidden_states(state.hidden_states)
        aux_hidden_states = state.aux_hidden_states
        if aux_hidden_states is not None:
            aux_hidden_states = [self._restorer.restore_hidden_states(aux) for aux in aux_hidden_states]
        self._restored_outputs = (hidden_states, aux_hidden_states)
        return state._replace(hidden_states=hidden_states, aux_hidden_states=aux_hidden_states)

    def take_restored_outputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None] | None:
        """Return this step's restored outputs if ``hidden_states`` is one of them.

        A second all-gather of already global rows would reorder them.
        """
        restored_outputs, self._restored_outputs = self._restored_outputs, None
        if restored_outputs is None or hidden_states is not restored_outputs[0]:
            return None
        return restored_outputs
