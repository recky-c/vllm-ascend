# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu import model_runner as vllm_model_runner

from vllm_ascend.worker.v2.pcp.draft_handoff import ProvidesMTPTargetHiddenStates, ReplicatedDraftHandoff


class _FakeRestorer:
    def __init__(self, is_last_pp_rank: bool = True) -> None:
        self.global_batch = SimpleNamespace(name="global")
        self.is_last_pp_rank = is_last_pp_rank
        self.restored: list[torch.Tensor] = []
        self.restored_buffers: list[torch.Tensor] = []

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.restored.append(hidden_states)
        return hidden_states.repeat(2, 1)

    def restore_hidden_state_buffer(self, buffer: torch.Tensor) -> None:
        self.restored_buffers.append(buffer)


def _state(is_dummy: bool = False, aux: bool = True):
    return vllm_model_runner.ExecuteModelState(
        input_batch=SimpleNamespace(is_dummy=is_dummy),
        attn_metadata={"layer": object()},
        slot_mappings_by_layer={"layer": torch.zeros(1)},
        hidden_states=torch.zeros(2, 4),
        aux_hidden_states=[torch.ones(2, 4)] if aux else None,
        dp_sync=object(),
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
    )


def _assert_rank_local_views_dropped(state) -> None:
    assert state.attn_metadata is None
    assert state.slot_mappings_by_layer is None
    assert state.dp_sync is None


def test_last_pp_rank_restores_outputs_and_model_buffer() -> None:
    restorer = _FakeRestorer()
    handoff = ReplicatedDraftHandoff(restorer)
    state = _state()
    buffer = torch.zeros(8, 4)

    draft_state = handoff.to_draft_step(state, buffer)

    _assert_rank_local_views_dropped(draft_state)
    assert draft_state.input_batch is state.input_batch
    assert draft_state.hidden_states.shape == (4, 4)
    assert draft_state.aux_hidden_states[0].shape == (4, 4)
    assert restorer.restored_buffers == [buffer]
    assert handoff.take_restored_outputs(draft_state.hidden_states) == (
        draft_state.hidden_states,
        draft_state.aux_hidden_states,
    )


def test_restored_outputs_are_taken_once_and_only_by_identity() -> None:
    handoff = ReplicatedDraftHandoff(_FakeRestorer())
    draft_state = handoff.to_draft_step(_state(aux=False), None)

    assert handoff.take_restored_outputs(torch.zeros(4, 4)) is None
    # A mismatching caller also consumes the step, so it never leaks forward.
    assert handoff.take_restored_outputs(draft_state.hidden_states) is None


def test_non_last_pp_rank_gets_global_batch_without_restoring() -> None:
    restorer = _FakeRestorer(is_last_pp_rank=False)
    handoff = ReplicatedDraftHandoff(restorer)

    draft_state = handoff.to_draft_step(_state(), torch.zeros(8, 4))

    _assert_rank_local_views_dropped(draft_state)
    assert draft_state.input_batch is restorer.global_batch
    assert restorer.restored == restorer.restored_buffers == []


@pytest.mark.parametrize("is_last_pp_rank", [False, True])
def test_dummy_step_keeps_layout_but_drops_rank_local_views(is_last_pp_rank: bool) -> None:
    restorer = _FakeRestorer(is_last_pp_rank)
    handoff = ReplicatedDraftHandoff(restorer)
    state = _state(is_dummy=True)

    draft_state = handoff.to_draft_step(state, torch.zeros(8, 4))

    _assert_rank_local_views_dropped(draft_state)
    assert draft_state.input_batch is state.input_batch
    assert draft_state.hidden_states is state.hidden_states
    assert restorer.restored == restorer.restored_buffers == []
    assert handoff.take_restored_outputs(state.hidden_states) is None


def test_mtp_hidden_state_provider_is_structural() -> None:
    class _Provider:
        def get_mtp_target_hidden_states(self):
            return None

    assert isinstance(_Provider(), ProvidesMTPTargetHiddenStates)
    assert not isinstance(object(), ProvidesMTPTargetHiddenStates)
