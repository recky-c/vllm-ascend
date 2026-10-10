# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as dflash_aclgraph
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


def _speculator(speculator_cls=AscendDFlashSpeculator):
    speculator = object.__new__(speculator_cls)
    speculator.num_query_per_req = 4
    speculator._group_causal = {0: False, 1: True}
    speculator.attn_backends = {"draft.layer.0": object()}
    speculator.input_batch = SimpleNamespace(
        num_reqs=2,
        seq_lens_cpu_upper_bound=torch.tensor([17, 31], dtype=torch.int32),
    )
    return speculator


def _metadata():
    first = SimpleNamespace(
        actual_seq_lengths_q=[4, 8, 8, 8], block_tables=object(), slot_mapping=object(), causal=False
    )
    second = SimpleNamespace(
        actual_seq_lengths_q=[4, 8, 8, 8], block_tables=object(), slot_mapping=object(), causal=True
    )
    return {"draft.layer.0": first, "draft.layer.1": first, "draft.layer.2": second}


def _build(speculator, desc):
    return speculator._build_uniform_attn_metadata(
        batch_desc=desc,
        num_reqs=speculator.input_batch.num_reqs,
        num_query_per_req=speculator.num_query_per_req,
        seq_lens_cpu_upper_bound=speculator.input_batch.seq_lens_cpu_upper_bound,
        step=speculator.num_query_per_req,
        causal=speculator._group_causal,
    )


def _propose(speculator):
    return speculator.propose(
        speculator.input_batch,
        {},
        {},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("speculator_cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("updatable", [False, True])
def test_full_graph_reuses_current_draft_metadata(monkeypatch, speculator_cls, updatable):
    speculator = _speculator(speculator_cls)
    manager = object.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager._graph_replay = MagicMock()
    manager._updatable_graph_replay = MagicMock()
    monkeypatch.setattr(dflash_aclgraph, "use_updatable_graph", lambda backend: updatable)
    built = [_metadata(), _metadata()]
    parent_build = MagicMock(side_effect=built)
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)
    # Runtime descriptors include uniform_token_count, while the Ascend
    # fallback descriptor only specifies the shape. Shape matching must still
    # reuse the upstream build, including its group-local cache addresses.
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4, uniform_token_count=4)

    def parent_propose(self, *args, **kwargs):
        metadata = _build(self, desc)
        manager.run_fullgraph(desc)
        return metadata

    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    for expected in built:
        result = _propose(speculator)
        assert result is expected
        for metadata in result.values():
            assert metadata.actual_seq_lengths_q == [4, 8, 12, 16]
        assert result["draft.layer.0"].block_tables is not result["draft.layer.2"].block_tables
        assert result["draft.layer.0"].slot_mapping is not result["draft.layer.2"].slot_mapping
        assert result["draft.layer.0"].causal is False
        assert result["draft.layer.2"].causal is True
        assert speculator._draft_attn_metadata_for_graph is None
        assert speculator._reuse_draft_attn_metadata is False
    assert parent_build.call_count == 2
    replay = manager._updatable_graph_replay if updatable else manager._graph_replay
    metadata_arg = 1 if updatable else 3
    assert replay.call_args_list[0].args[metadata_arg][0] is built[0]
    assert replay.call_args_list[1].args[metadata_arg][0] is built[1]


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_eager_propose_does_not_handoff_metadata(monkeypatch, mode):
    speculator = _speculator()
    expected = _metadata()
    parent_build = MagicMock(return_value=expected)
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)

    def parent_propose(self, *args, **kwargs):
        metadata = _build(self, BatchExecutionDescriptor(mode, 8, 2))
        assert self._draft_attn_metadata_for_graph is None
        return metadata

    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    assert _propose(speculator) is expected
    assert expected["draft.layer.0"].actual_seq_lengths_q == [4, 8, 8, 8]
    parent_build.assert_called_once()


@pytest.mark.parametrize("num_tokens,num_reqs", [(12, 3), (12, 4)])
def test_full_graph_shape_mismatch_rebuilds(monkeypatch, num_tokens, num_reqs):
    speculator = _speculator()
    initial, rebuilt = _metadata(), _metadata()
    parent_build = MagicMock(side_effect=[initial, rebuilt])
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)

    def parent_propose(self, *args, **kwargs):
        _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens, num_reqs))
        return self.build_draft_attn_metadatas(4, self.input_batch.seq_lens_cpu_upper_bound)[0]

    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    assert _propose(speculator) is rebuilt
    assert parent_build.call_count == 2
    assert initial["draft.layer.0"].actual_seq_lengths_q == [4, 8, 8, 8]
    assert rebuilt["draft.layer.0"].actual_seq_lengths_q == [4, 8, 12, 16]


def test_full_graph_handoff_is_consumed_once_per_proposal(monkeypatch):
    speculator = _speculator()
    built = [_metadata(), _metadata(), _metadata()]
    parent_build = MagicMock(side_effect=built)
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)

    def parent_propose(self, *args, **kwargs):
        _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4))
        for expected in built:
            result = self.build_draft_attn_metadatas(4, self.input_batch.seq_lens_cpu_upper_bound)
            assert result[0] is expected
            assert result[0]["draft.layer.0"].actual_seq_lengths_q == [4, 8, 12, 16]
            assert self._draft_attn_metadata_for_graph is None
        return built[-1]

    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    assert _propose(speculator) is built[-1]
    assert parent_build.call_count == 3
    assert speculator._reuse_draft_attn_metadata is False


def test_failed_propose_drops_metadata_before_standalone_graph_update(monkeypatch):
    speculator = _speculator()
    initial, rebuilt = _metadata(), _metadata()
    parent_build = MagicMock(side_effect=[initial, rebuilt])
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)

    def parent_propose(self, *args, **kwargs):
        _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4))
        raise RuntimeError("graph replay failed")

    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    with pytest.raises(RuntimeError, match="graph replay failed"):
        _propose(speculator)
    assert speculator._draft_attn_metadata_for_graph is None
    assert speculator._reuse_draft_attn_metadata is False
    result = speculator.build_draft_attn_metadatas(4, speculator.input_batch.seq_lens_cpu_upper_bound)
    assert result[0] is rebuilt
    assert parent_build.call_count == 2
    assert speculator._draft_attn_metadata_for_graph is None


def test_uniform_build_forwards_dcp_metadata(monkeypatch):
    speculator = _speculator()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    local_seq_lens = torch.tensor([11, 19], dtype=torch.int32)
    parent_build = MagicMock(return_value=_metadata())
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)
    result = speculator._build_uniform_attn_metadata(
        desc,
        2,
        4,
        speculator.input_batch.seq_lens_cpu_upper_bound,
        4,
        speculator._group_causal,
        dcp_local_seq_lens=local_seq_lens,
    )
    assert result is parent_build.return_value
    assert parent_build.call_args.kwargs["dcp_local_seq_lens"] is local_seq_lens
    assert parent_build.call_args.kwargs["causal"] is speculator._group_causal
