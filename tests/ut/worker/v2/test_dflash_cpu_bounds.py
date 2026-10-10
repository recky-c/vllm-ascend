# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from tests.ut.attention.test_batch_metadata_reuse import make_fia_builder, make_gdn_common
from tests.ut.worker.v2.test_dflash_speculator import _build, _propose, _speculator
from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.attention.attention_v1 import AscendAttentionState, FIAParamProvider
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.attn_utils import (
    _get_dflash_draft_fia_seq_lens_cpu,
    dflash_draft_kv_optimistic_bound,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _supports_dflash_draft_kv_optimistic_bound,
)


def _config(**changes):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        speculative_config=SimpleNamespace(method="dflash", enable_adaptive_verification=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    for name, value in changes.items():
        owner = config.model_config.hf_text_config if name == "model_type" else config.speculative_config
        if name.endswith("parallel_size"):
            owner = config.parallel_config
        setattr(owner, name, value)
    return config


def _eligible_speculator():
    speculator = _speculator()
    speculator.max_model_len = 256
    speculator._dflash_draft_kv_optimistic_bound_enabled = True
    speculator._dflash_draft_kv_optimistic_bound_active = True
    speculator._dflash_draft_kv_zeroing_ready = True
    speculator.input_batch.has_prefill = False
    speculator.input_batch.idx_mapping_np = np.array([2, 0])
    speculator.draft_kv_cache_group_ids = [0, 1]
    speculator.block_tables = SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.array([[2, 0, 1], [2, 0, 1]])),
        kernel_block_sizes=[32, 32],
    )
    return speculator


def test_config_is_default_off_and_validates_boolean():
    kwargs = dict(sparse_kv_offload_config=SparseKVOffloadConfig())
    assert AscendConfig(**kwargs).enable_dflash_draft_kv_optimistic_bound is False
    assert AscendConfig(**kwargs, enable_dflash_draft_kv_optimistic_bound=True).enable_dflash_draft_kv_optimistic_bound
    assert not AscendConfig(
        **kwargs, enable_dflash_draft_kv_optimistic_bound="false"
    ).enable_dflash_draft_kv_optimistic_bound
    with pytest.raises(ValueError):
        AscendConfig(**kwargs, enable_dflash_draft_kv_optimistic_bound="invalid")


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "glm5_next"},
        {"model_type": "qwen3"},
        {"method": "dflash2"},
        {"method": "dspark"},
        {"enable_adaptive_verification": True},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_unsupported_config_retains_exact_lengths(changes):
    assert _supports_dflash_draft_kv_optimistic_bound(_config())
    assert not _supports_dflash_draft_kv_optimistic_bound(_config(**changes))


@pytest.mark.parametrize("target_bounds", [[28, 60], [29, 60], [28, 61]])
def test_allocation_ledger_checks_each_request_and_group(target_bounds):
    speculator = _eligible_speculator()
    bounds = torch.tensor(target_bounds, dtype=torch.int32)
    assert speculator._can_use_dflash_draft_cpu_bound(2, bounds, 4) is (target_bounds == [28, 60])
    # A short SWA allocation cannot borrow Full attention's longer allocation.
    speculator.block_tables.num_blocks.np[1, 0] = 1
    assert not speculator._can_use_dflash_draft_cpu_bound(2, bounds, 4)


def test_allocation_guard_clamps_max_model_len_and_fails_closed():
    speculator = _eligible_speculator()
    speculator.max_model_len = 64
    speculator.block_tables.num_blocks.np[:, 2] = 2
    bounds = torch.tensor([63, 64], dtype=torch.int32)
    assert speculator._can_use_dflash_draft_cpu_bound(2, bounds, 4)
    assert not speculator._can_use_dflash_draft_cpu_bound(2, bounds, 3)
    speculator.input_batch.idx_mapping_np = np.array([-1, 0])
    assert not speculator._can_use_dflash_draft_cpu_bound(2, bounds, 4)


def test_optimistic_bounds_require_complete_recycled_block_zeroing():
    speculator = _eligible_speculator()
    speculator._dflash_draft_kv_zeroing_ready = False
    assert not speculator._can_use_dflash_draft_cpu_bound(2, torch.tensor([17, 31]), 4)


def test_mixed_target_gdn_group_does_not_disable_draft_bounds():
    speculator = _eligible_speculator()
    # Group 0 is target GDN state storage, not token-addressed draft KV.
    speculator.draft_kv_cache_group_ids = [1, 2]
    speculator.block_tables.num_blocks.np = np.array([[1, 1, 1], [2, 0, 1], [2, 0, 1]])
    speculator.block_tables.kernel_block_sizes = [1, 32, 32]
    assert speculator._can_use_dflash_draft_cpu_bound(2, torch.tensor([28, 60]), 4)
    del speculator.block_tables.num_blocks
    assert not speculator._can_use_dflash_draft_cpu_bound(2, torch.tensor([28, 60]), 4)


@pytest.mark.parametrize(
    "bounds",
    [None, torch.tensor([40]), torch.tensor([40.0, 60.0, 0.0]), torch.tensor([-1, 60, 0]), torch.tensor([65, 60, 0])],
)
def test_invalid_cpu_bound_falls_back_to_exact(bounds):
    assert _get_dflash_draft_fia_seq_lens_cpu(bounds, torch.tensor([0, 4, 8, 8]), 3, 64) is None


def test_per_request_bounds_keep_padding_separate():
    actual = _get_dflash_draft_fia_seq_lens_cpu(torch.tensor([40, 60, 0]), torch.tensor([0, 4, 8, 8]), 3, 64)
    assert actual.tolist() == [40, 60, 1]


def test_context_is_nested_exception_safe_and_thread_local():
    flag = attn_utils._DFLASH_DRAFT_KV_OPTIMISTIC_BOUND
    assert flag.get() is False
    with dflash_draft_kv_optimistic_bound(True):
        assert flag.get() is True
        with pytest.raises(RuntimeError), dflash_draft_kv_optimistic_bound(False):
            assert flag.get() is False
            raise RuntimeError("test nested failure")
        assert flag.get() is True
        assert Context().run(flag.get) is False
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(flag.get).result() is False
    assert flag.get() is False


@pytest.mark.parametrize(
    "dummy_run,is_profile,has_prefill",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
)
def test_only_real_decode_proposal_enables_scope(monkeypatch, dummy_run, is_profile, has_prefill):
    speculator = _eligible_speculator()
    speculator.input_batch.has_prefill = has_prefill
    observed = []

    def parent_build(self, *args, **kwargs):
        observed.append(attn_utils._DFLASH_DRAFT_KV_OPTIMISTIC_BOUND.get())
        return {}

    def parent_propose(self, *args, **kwargs):
        return _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4))

    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)
    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    speculator.propose(
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
        dummy_run=dummy_run,
        is_profile=is_profile,
    )
    assert observed == [not (dummy_run or is_profile or has_prefill)]
    assert speculator._dflash_draft_kv_optimistic_bound_active is False
    assert attn_utils._DFLASH_DRAFT_KV_OPTIMISTIC_BOUND.get() is False


def test_failed_proposal_and_standalone_update_use_exact_scope(monkeypatch):
    speculator = _eligible_speculator()
    observed = []

    def parent_build(self, *args, **kwargs):
        observed.append(attn_utils._DFLASH_DRAFT_KV_OPTIMISTIC_BOUND.get())
        return {}

    def parent_propose(self, *args, **kwargs):
        _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4))
        raise RuntimeError("proposal failed")

    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)
    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    with pytest.raises(RuntimeError, match="proposal failed"):
        _propose(speculator)
    speculator.build_draft_attn_metadatas(4, speculator.input_batch.seq_lens_cpu_upper_bound)
    assert observed == [True, False]


@pytest.mark.parametrize("sliding", [False, True])
def test_fia_host_bounds_preserve_device_lengths_and_cache_scope(monkeypatch, sliding):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    builder = make_fia_builder(torch.device("cpu"), sliding=sliding)
    common = make_gdn_common(torch.device("cpu"), [4, 4, 0], [38, 58, 8], [3, 3, -1])
    common.attn_state = AscendAttentionState.SpecDecoding
    common.causal = sliding
    exact = common.seq_lens
    bound = torch.tensor([40, 60, 1], dtype=torch.int32)
    caches = ({}, {})
    exact_metadata = builder.build(0, common, batch_metadata_cache=caches[0], common_fia_metadata=caches[1])
    assert exact_metadata.seq_lens_list == [38, 58, 8]
    common.dflash_draft_seq_lens_cpu_upper_bound = bound
    tolist = torch.Tensor.tolist

    def no_exact_readback(tensor):
        assert tensor.data_ptr() != exact.data_ptr(), "unexpected exact device length readback"
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", no_exact_readback)
    optimistic_metadata = builder.build(0, common, batch_metadata_cache=caches[0], common_fia_metadata=caches[1])
    assert optimistic_metadata.seq_lens_list == [40, 60, 1]
    assert optimistic_metadata.seq_lens is exact
    assert optimistic_metadata.seq_lens_gpu is exact
    assert len(caches[0]) == len(caches[1]) == 2
    params = FIAParamProvider("draft", 16 if sliding else None, is_draft_model=True).resolve(
        {"draft": optimistic_metadata}
    )
    assert params["actual_seq_lengths_kv"] == [40, 60, 1]
    assert params["block_table"] is optimistic_metadata.block_tables


@pytest.mark.parametrize("capture", [False, True])
def test_runner_sets_bound_only_in_draft_scope(monkeypatch, capture):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    builder = make_fia_builder(torch.device("cpu"))
    group = SimpleNamespace(
        layer_names=["layer"], kv_cache_spec=builder.kv_cache_spec, get_metadata_builder=lambda _: builder
    )
    common = make_gdn_common(torch.device("cpu"), [4, 4, 0], [38, 58, 8], [3, 3, -1])
    kwargs = dict(
        attn_groups=[[group]],
        num_reqs=3,
        num_tokens=8,
        query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        max_query_len=4,
        seq_lens=common.seq_lens,
        max_seq_len=64,
        block_tables=[common.block_table_tensor],
        slot_mappings=common.slot_mapping.unsqueeze(0),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=builder.kv_cache_spec)]),
        seq_lens_cpu_upper_bound=torch.tensor([40, 60, 0], dtype=torch.int32),
        for_cudagraph_capture=capture,
    )
    # Target build is outside the draft scope even with the same CPU bound.
    target = attn_utils.build_attn_metadata(**kwargs)["layer"]
    assert target.seq_lens_list == [38, 58, 8]
    with dflash_draft_kv_optimistic_bound(True):
        draft = attn_utils.build_attn_metadata(**kwargs)["layer"]
    assert draft.seq_lens_list == ([38, 58, 8] if capture else [40, 60, 1])
    assert torch.equal(draft.seq_lens_gpu, common.seq_lens)
    kwargs.pop("seq_lens_cpu_upper_bound")
    with dflash_draft_kv_optimistic_bound(True):
        fallback = attn_utils.build_attn_metadata(**kwargs)["layer"]
    assert fallback.seq_lens_list == [38, 58, 8]


def test_real_mixed_dflash_producer_full_handoff_and_shape_fallback(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    speculator = _eligible_speculator()
    speculator.draft_max_seq_len = 35
    speculator.arange_np = np.arange(5, dtype=np.int32)
    speculator.draft_is_prefilling = torch.zeros(2, dtype=torch.bool)
    speculator._group_causal = {1: False, 2: True}
    speculator.draft_kv_cache_group_ids = [1, 2]
    builders = [make_fia_builder(torch.device("cpu"), sliding=sliding) for sliding in (False, True)]
    groups = [
        SimpleNamespace(
            layer_names=[f"draft.{idx}"],
            kv_cache_spec=builder.kv_cache_spec,
            get_metadata_builder=lambda _, builder=builder: builder,
        )
        for idx, builder in enumerate(builders)
    ]
    specs = [MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,))]
    specs.extend(builder.kv_cache_spec for builder in builders)
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    )
    speculator.attn_groups = [[], [groups[0]], [groups[1]]]
    speculator.block_tables = SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.array([[1, 1, 1], [2, 0, 1], [2, 0, 1]])),
        kernel_block_sizes=[1, 32, 32],
        cp_size=1,
        input_block_tables=[torch.zeros(4, 4, dtype=torch.int32) for _ in specs],
        slot_mappings=torch.full((3, 16), -1, dtype=torch.int32),
    )
    exact = torch.tensor([19, 32, 8, 8], dtype=torch.int32)
    speculator.input_buffers = SimpleNamespace(
        seq_lens=exact, query_start_loc=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32)
    )

    def parent_propose(self, *args, **kwargs):
        initial = _build(self, BatchExecutionDescriptor(CUDAGraphMode.FULL, 12, 3))
        for metadata in initial.values():
            assert metadata.seq_lens_list == [21, 35, 1]
        reused = self.build_draft_attn_metadatas(3, self.input_batch.seq_lens_cpu_upper_bound)[0]
        assert reused is initial
        assert reused["draft.0"].actual_seq_lengths_q == [4, 8, 12]
        fresh = self.build_draft_attn_metadatas(4, self.input_batch.seq_lens_cpu_upper_bound)[0]
        assert fresh is not initial
        for metadata in fresh.values():
            assert metadata.seq_lens_list == [21, 35, 1, 1]
            assert metadata.actual_seq_lengths_q == [4, 8, 12, 16]
            torch.testing.assert_close(metadata.seq_lens_gpu, exact)
        return fresh

    tolist = torch.Tensor.tolist

    def no_exact_readback(tensor):
        assert tensor.data_ptr() != exact.data_ptr(), "mixed DFlash producer read device lengths"
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", no_exact_readback)
    monkeypatch.setattr(DFlashSpeculator, "propose", parent_propose)
    assert set(_propose(speculator)) == {"draft.0", "draft.1"}
    assert speculator._dflash_draft_kv_optimistic_bound_active is False
