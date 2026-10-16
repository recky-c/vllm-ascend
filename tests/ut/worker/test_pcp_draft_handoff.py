# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.pcp.draft_handoff import ProvidesMTPTargetHiddenStates, ReplicatedDraftHandoff


def _padded_global_batch(is_dummy: bool = False) -> InputBatch:
    buffers = AscendInputBuffers(max_num_reqs=4, max_num_tokens=8, device=torch.device("cpu"))
    batch = InputBatch.make_dummy(num_reqs=2, num_tokens=4, input_buffers=buffers)
    batch.num_tokens = 3
    batch.query_start_loc.copy_(torch.tensor([0, 1, 3, 4], dtype=torch.int32)[: batch.query_start_loc.shape[0]])
    batch.query_start_loc_np = np.array([0, 1, 3, 4], dtype=np.int32)
    batch.is_dummy = is_dummy
    return batch


class _FakeRestorer:
    def __init__(self, is_last_pp_rank: bool = True, sharded_spec_decode: bool = False) -> None:
        self.global_batch = _padded_global_batch()
        self.is_last_pp_rank = is_last_pp_rank
        self.vllm_config = SimpleNamespace(speculative_config=object() if sharded_spec_decode else None)
        self.is_decode_sharded = sharded_spec_decode
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


def test_sharded_spec_decode_hides_target_graph_padding() -> None:
    handoff = ReplicatedDraftHandoff(_FakeRestorer(sharded_spec_decode=True))
    global_batch = _padded_global_batch()
    hidden, aux = torch.arange(8).reshape(4, 2), [torch.ones(4, 2)]

    scheduled_hidden, scheduled_aux, scheduled_batch = handoff.to_scheduled_tokens(hidden, aux, global_batch)

    assert scheduled_hidden.shape == scheduled_aux[0].shape == (3, 2)
    assert scheduled_hidden.data_ptr() == hidden.data_ptr()
    assert scheduled_batch.num_tokens_after_padding == 3
    assert scheduled_batch.query_start_loc_np.tolist() == [0, 1, 3]
    assert scheduled_batch.query_start_loc.tolist() == [0, 1, 3]
    # The target keeps its padded batch.
    assert global_batch.num_tokens_after_padding == 4


@pytest.mark.parametrize(("sharded_spec_decode", "is_dummy"), [(False, False), (True, True)])
def test_other_layouts_keep_global_padding(sharded_spec_decode: bool, is_dummy: bool) -> None:
    handoff = ReplicatedDraftHandoff(_FakeRestorer(sharded_spec_decode=sharded_spec_decode))
    global_batch = _padded_global_batch(is_dummy)
    hidden = torch.zeros(4, 2)

    assert handoff.to_scheduled_tokens(hidden, None, global_batch) == (hidden, None, global_batch)


def test_draft_step_caches_scheduled_outputs() -> None:
    handoff = ReplicatedDraftHandoff(_FakeRestorer(sharded_spec_decode=True))
    draft_state = handoff.to_draft_step(_state(), None)

    # _FakeRestorer doubles 2 local rows into the padded 4-row global layout.
    assert draft_state.hidden_states.shape == draft_state.aux_hidden_states[0].shape == (3, 4)
    assert handoff.take_restored_outputs(draft_state.hidden_states)[0] is draft_state.hidden_states
