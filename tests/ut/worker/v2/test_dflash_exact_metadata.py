# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import Context
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from tests.ut.attention.test_batch_metadata_reuse import make_fia_builder, make_gdn_common
from tests.ut.worker.v2.test_dflash_speculator import _build, _speculator
from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.attention.attention_v1 import FIAParamProvider
from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, UpdatableGraph
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.attn_utils import dflash_draft_seq_lens_cpu
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as aclgraph_module
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager, DFlashCudaGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator, dflash_inputs_prepared


@pytest.mark.parametrize("field", ["enable_dflash_exact_metadata_optimizations", "enable_gdn_graph_state_batching"])
def test_default_true_and_typed_control(field):
    kwargs = dict(sparse_kv_offload_config=SparseKVOffloadConfig())
    assert getattr(AscendConfig(**kwargs), field) is True
    assert getattr(AscendConfig(**kwargs, **{field: "false"}), field) is False
    with pytest.raises(ValueError):
        AscendConfig(**kwargs, **{field: "invalid"})


def test_deferred_control_defaults_false_and_is_typed():
    kwargs = dict(sparse_kv_offload_config=SparseKVOffloadConfig())
    assert AscendConfig(**kwargs).enable_dflash_deferred_metadata is False
    assert AscendConfig(**kwargs, enable_dflash_deferred_metadata="true").enable_dflash_deferred_metadata is True
    with pytest.raises(ValueError):
        AscendConfig(**kwargs, enable_dflash_deferred_metadata="invalid")


def test_input_callback_is_execution_local_and_resets_after_exception(monkeypatch):
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", lambda *args, **kwargs: None)
    factory = dflash_module.prepare_dflash_inputs_factory(128)
    seen = []
    with dflash_inputs_prepared(lambda: seen.append("outer")):
        factory()
        with pytest.raises(RuntimeError), dflash_inputs_prepared(lambda: seen.append("inner")):
            factory()
            raise RuntimeError("test")
        factory()
        Context().run(factory)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(factory).result()
    factory()
    assert seen == ["outer", "inner", "outer"]


def test_snapshot_waits_only_copy_event_and_reuses_pinned_allocation(monkeypatch):
    spec = _speculator()
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor([73, 81, 97, 999], dtype=torch.int32))
    stream, event = MagicMock(), MagicMock()
    monkeypatch.setattr(torch.npu, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.npu, "current_stream", lambda: "producer")
    empty = torch.empty
    allocations = []

    def host_empty(*args, **kwargs):
        allocations.append(kwargs.pop("pin_memory", False))
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    spec._start_draft_seq_lens_copy(3)
    first = spec._prepare_draft_seq_lens_cpu(3, 4)
    assert first.tolist() == [73, 81, 97, 0]
    address = first.data_ptr()
    spec.input_buffers.seq_lens.copy_(torch.tensor([8, 94, 123, 999]))
    spec._start_draft_seq_lens_copy(1)
    second = spec._prepare_draft_seq_lens_cpu(1, 4)
    assert second.tolist() == [8, 0, 0, 0]
    assert second.data_ptr() == address
    spec._start_draft_seq_lens_copy(3)
    assert spec._prepare_draft_seq_lens_cpu(3, 4).tolist() == [8, 94, 123, 0]
    assert allocations == [True]
    assert event.synchronize.call_count == 3
    assert stream.wait_stream.call_args.args == ("producer",)
    assert spec._draft_seq_lens_copy_count is None


@pytest.mark.parametrize("sliding", [False, True])
def test_exact_host_list_keeps_device_source_and_padding(monkeypatch, sliding):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    builder = make_fia_builder(torch.device("cpu"), sliding=sliding)
    common = make_gdn_common(torch.device("cpu"), [4, 4, 0], [38, 58, 99], [3, 3, -1])
    device = common.seq_lens
    common.seq_lens_cpu = torch.tensor([36, 55, 0], dtype=torch.int32)
    common.seq_lens_cpu_is_exact = True
    metadata = builder.build(0, common)
    assert metadata.seq_lens_list == [36, 55, 1]
    assert metadata.seq_lens is device and metadata.seq_lens_gpu is device
    common.seq_lens_cpu_is_exact = False
    assert builder.build(0, common).seq_lens_list == [38, 58, 99]


def _real_producer(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    spec = _speculator()
    spec.max_model_len = 8192
    spec.draft_max_seq_len = 8192
    spec.arange_np = np.arange(5, dtype=np.int32)
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec._group_causal = {1: False, 2: True}
    spec.draft_kv_cache_group_ids = [1, 2]
    spec._reuse_draft_layout = True
    spec._use_cpu_seq_lens = True
    spec._exact_draft_metadata_active = True
    spec._reuse_draft_attn_metadata = True
    spec._draft_metadata_template = None
    builders = [make_fia_builder(torch.device("cpu"), sliding=sliding) for sliding in (False, True)]
    for builder in builders:
        builder.supports_update_block_table = True
    groups = [
        SimpleNamespace(layer_names=[f"draft{idx}"], get_metadata_builder=lambda _, b=b: b)
        for idx, b in enumerate(builders)
    ]
    spec.attn_groups = [[], [groups[0]], [groups[1]]]
    spec.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=b.kv_cache_spec) for b in [builders[0], *builders]]
    )
    spec.block_tables = SimpleNamespace(
        cp_size=1,
        input_block_tables=[torch.zeros(4, 64, dtype=torch.int32) for _ in range(3)],
        slot_mappings=torch.full((3, 16), -1, dtype=torch.int32),
    )
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([73, 81, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
    )
    mirror = torch.tensor([73, 81, 0, 0], dtype=torch.int32)
    monkeypatch.setattr(spec, "_can_use_exact_draft_metadata", lambda *args: True)
    monkeypatch.setattr(spec, "_can_defer_draft_metadata", lambda desc: False)
    monkeypatch.setattr(spec, "_prepare_draft_seq_lens_cpu", lambda real, padded: mirror[:padded])
    return spec, mirror


def test_real_upstream_template_refresh_keeps_group_tables_and_scratch(monkeypatch):
    spec, mirror = _real_producer(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    first = _build(spec, desc)
    assert first["draft0"].seq_lens_list == [73, 81, 1, 1]
    first["draft0"].qfa_metadata_cache["old"] = True
    first["draft0"].reshape_cache_event = object()
    spec.block_tables.input_block_tables[1].fill_(3)
    spec.block_tables.input_block_tables[2].fill_(7)
    mirror.copy_(torch.tensor([74, 60, 0, 0]))
    parent_build = MagicMock(side_effect=AssertionError("template rebuilt"))
    monkeypatch.setattr(DFlashSpeculator, "_build_uniform_attn_metadata", parent_build)
    refreshed = _build(spec, desc)
    assert refreshed["draft0"].seq_lens_list == [74, 60, 1, 1]
    assert refreshed["draft0"].block_tables[0, 0] == 3
    assert refreshed["draft1"].block_tables[0, 0] == 7
    assert refreshed["draft0"].qfa_metadata_cache == {}
    assert refreshed["draft0"].reshape_cache_event is None
    assert refreshed["draft0"].seq_lens_gpu.data_ptr() == spec.input_buffers.seq_lens.data_ptr()
    assert refreshed["draft0"].causal is False and refreshed["draft1"].causal is True
    assert spec.build_draft_attn_metadatas(4, spec.input_batch.seq_lens_cpu_upper_bound)[0] is refreshed
    parent_build.assert_not_called()


@pytest.mark.parametrize("change", ["batch", "causal", "table", "builder"])
def test_template_key_invalidates_on_layout_change(monkeypatch, change):
    spec, _ = _real_producer(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    key = spec._draft_layout_key(desc, 2, 4, 4, spec._group_causal)
    if change == "batch":
        spec.input_batch.num_reqs = 1
    elif change == "causal":
        spec._group_causal[1] = True
    elif change == "table":
        spec.block_tables.input_block_tables[1] = torch.zeros(4, 65, dtype=torch.int32)
    else:
        builder = make_fia_builder(torch.device("cpu"))
        spec.attn_groups[1][0].get_metadata_builder = lambda _: builder
    assert key != spec._draft_layout_key(desc, spec.input_batch.num_reqs, 4, 4, spec._group_causal)


@pytest.mark.parametrize(
    "dummy,prefill,profile", [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
)
def test_proposal_exact_scope_cleans_on_failure(monkeypatch, dummy, prefill, profile):
    spec = _speculator()
    spec._use_cpu_seq_lens = True
    spec.input_batch.has_prefill = prefill
    active = []

    def proposal(self, *args, **kwargs):
        active.append(self._exact_draft_metadata_active)
        raise ValueError("failed")

    monkeypatch.setattr(DFlashSpeculator, "propose", proposal)
    with pytest.raises(ValueError, match="failed"):
        spec.propose(
            spec.input_batch,
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
            dummy_run=dummy,
            is_profile=profile,
        )
    assert active == [not (dummy or prefill or profile)]
    assert not spec._exact_draft_metadata_active
    assert spec._deferred_draft_attn_metadata is None
    assert dflash_module._DFLASH_INPUTS_PREPARED.get() is None


def test_nested_exact_mirror_scope_does_not_leak_to_target():
    outer, inner = torch.tensor([8]), torch.tensor([16])
    with dflash_draft_seq_lens_cpu(outer):
        with pytest.raises(ValueError), dflash_draft_seq_lens_cpu(inner):
            assert attn_utils._DFLASH_DRAFT_SEQ_LENS_CPU.get() is inner
            raise ValueError("test")
        assert attn_utils._DFLASH_DRAFT_SEQ_LENS_CPU.get() is outer
        assert Context().run(attn_utils._DFLASH_DRAFT_SEQ_LENS_CPU.get) is None
    assert attn_utils._DFLASH_DRAFT_SEQ_LENS_CPU.get() is None


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize(
    "optimistic,exact,cp,method,adaptive,enabled",
    [
        (False, True, 1, "dflash", False, True),
        (True, True, 1, "dflash", False, False),
        (False, False, 1, "dflash", False, False),
        (False, True, 2, "dflash", False, False),
        (False, True, 1, "dflash2", False, False),
        (False, True, 1, "dspark", False, False),
        (False, True, 1, "dflash", True, False),
    ],
)
def test_exact_static_gate_and_explicit_optimistic_precedence(
    monkeypatch, optimistic, exact, cp, method, adaptive, enabled, deferred
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method=method, enable_adaptive_verification=adaptive),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=cp, decode_context_parallel_size=1
        ),
    )
    monkeypatch.setattr(DFlashSpeculator, "__init__", lambda *args: None)
    monkeypatch.setattr(AscendDFlashSpeculator, "_lmhead_tp_validate_draft_sampling", lambda self: None)
    monkeypatch.setattr(
        dflash_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_dflash_draft_kv_optimistic_bound=optimistic,
            enable_dflash_exact_metadata_optimizations=exact,
            enable_dflash_deferred_metadata=deferred,
        ),
    )
    spec = AscendDFlashSpeculator(config, torch.device("cpu"))
    assert spec._exact_metadata_enabled is enabled
    assert spec._deferred_metadata_enabled is (deferred and enabled)
    assert spec._draft_seq_lens_copy_count is None
    assert spec._draft_metadata_template is None
    assert not spec._exact_draft_metadata_active


@pytest.mark.parametrize("failure", ["materialize", "resolve", "update"])
def test_deferred_replay_orders_input_dependency_and_rescues_all_tasks(monkeypatch, failure):
    order = []
    spec = _speculator()
    spec._deferred_metadata_enabled = True
    spec._draft_metadata_template = ("key", {})
    spec._draft_attn_metadata_for_graph = object()
    spec._deferred_draft_attn_metadata = {"pending": True}
    graph = MagicMock(spec=UpdatableGraph)
    neutral = (object(), object())
    graph._dflash_neutral_tasks = neutral
    manager = object.__new__(DFlashAclGraphManager)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    manager.graphs, manager.speculator = {desc: graph}, spec
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: order.append("input_wait"))
    compute = SimpleNamespace(synchronize=lambda: order.append("drain"))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: compute)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", lambda *args: order.append("replay"))

    def materialize(*args):
        order.append("materialize")
        if failure == "materialize":
            raise ValueError("original failure")
        return [{}]

    def resolve(*args):
        order.append("resolve")
        if failure == "resolve":
            raise ValueError("original failure")
        return ("runtime",)

    def update(stream, tasks):
        if tasks is neutral:
            order.append("rescue_all")
        else:
            order.append("update")
            raise ValueError("original failure")

    spec.build_draft_attn_metadatas = materialize
    graph.resolve_tasks.side_effect = resolve
    graph.update.side_effect = update
    with pytest.raises(ValueError, match="original failure"):
        manager.run_fullgraph(desc)
    assert order[:3] == ["input_wait", "replay", "materialize"]
    assert order[-2:] == ["rescue_all", "drain"]
    assert spec._deferred_draft_attn_metadata is None
    assert spec._draft_metadata_template is None
    assert spec._draft_attn_metadata_for_graph is None


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("padding", [0, 64])
def test_neutral_page_preserves_dense_inner_layout_and_independent_storage(monkeypatch, rank, padding):
    monkeypatch.setattr(aclgraph_module.torch_npu, "get_npu_format", lambda _: 2, raising=False)
    shape = (3, 128, 1024) if rank == 3 else (3, 128, 8, 128)
    inner = (1024, 1) if rank == 3 else (1024, 128, 1)
    payload, stride = 131072, 262144 + padding
    backing = torch.full((3 * stride,), 7, dtype=torch.bfloat16)
    cache = backing.as_strided(shape, (stride, *inner), storage_offset=payload)
    before = backing.clone()
    neutral = DFlashAclGraphManager._make_neutral_cache(cache)
    assert neutral.shape == (1, *shape[1:])
    assert neutral.stride() == cache.stride()
    assert neutral.storage_offset() == 0
    assert neutral.untyped_storage().nbytes() == stride * cache.element_size()
    assert neutral.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
    assert not torch.count_nonzero(neutral)
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "geometry",
    ["inner_transpose", "outer_overlap", "outer_zero", "empty", "rank", "integer", "nz"],
)
def test_neutral_unknown_or_overlapping_layout_keeps_metadata_first(monkeypatch, geometry):
    monkeypatch.setattr(
        aclgraph_module.torch_npu, "get_npu_format", lambda _: 29 if geometry == "nz" else 2, raising=False
    )
    backing = torch.ones(1024)
    if geometry == "inner_transpose":
        cache = backing.as_strided((3, 8, 4), (64, 1, 8))
    elif geometry == "outer_overlap":
        cache = backing.as_strided((3, 8, 4), (31, 4, 1))
    elif geometry == "outer_zero":
        cache = backing.as_strided((3, 8, 4), (0, 4, 1))
    elif geometry == "empty":
        cache = torch.empty(3, 0, 4)
    elif geometry == "rank":
        cache = torch.ones(8, 4)
    elif geometry == "integer":
        cache = torch.ones(3, 8, 4, dtype=torch.int8)
    else:
        cache = backing.as_strided((3, 8, 4), (64, 4, 1))
    assert DFlashAclGraphManager._make_neutral_cache(cache) is None


def test_neutral_preparation_is_complete_before_deferral(monkeypatch):
    monkeypatch.setattr(aclgraph_module.torch_npu, "get_npu_format", lambda _: 2, raising=False)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 2)
    cache = torch.ones(3, 8, 4)
    kwargs = {
        "key": cache,
        "value": cache,
        "block_table": torch.ones(2, 4, dtype=torch.int32),
        "block_size": 8,
        "input_layout": "TND",
    }
    task = GraphUpdateTask(
        aclgraph_module.torch_npu.npu_fused_infer_attention_score.out,
        kwargs,
        FIAParamProvider("draft0", None, is_draft_model=True),
        0,
        object(),
        object(),
    )
    graph = MagicMock(spec=UpdatableGraph)
    graph._dflash_neutral_tasks = None
    graph.tasks = [task, task.bind({"value": cache.transpose(1, 2)})]
    spec = _speculator()
    spec._deferred_metadata_enabled = True
    spec._reuse_draft_layout = True
    spec._exact_draft_metadata_active = True
    spec._draft_seq_lens_copy_count = 2
    manager = object.__new__(DFlashAclGraphManager)
    manager.graphs, manager.speculator = {desc: graph}, spec
    spec.query_cudagraph_manager = manager
    manager._prepare_neutral_tasks(desc)
    assert graph._dflash_neutral_tasks is None
    assert not spec._can_defer_draft_metadata(desc)
    graph.tasks[1] = task
    manager._prepare_neutral_tasks(desc)
    assert len(graph._dflash_neutral_tasks) == 2
    assert spec._can_defer_draft_metadata(desc)
    for neutral in graph._dflash_neutral_tasks:
        assert neutral.kwargs["actual_seq_lengths_kv"] == [4, 4]
        assert not torch.count_nonzero(neutral.kwargs["block_table"])


def test_default_off_builds_exact_template_before_replay_without_neutral_allocation(monkeypatch):
    spec, mirror = _real_producer(monkeypatch)
    spec._deferred_metadata_enabled = AscendConfig(
        sparse_kv_offload_config=SparseKVOffloadConfig()
    ).enable_dflash_deferred_metadata
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    first = _build(spec, desc)
    graph = MagicMock(spec=UpdatableGraph)
    graph.tasks = [object()]
    graph._dflash_neutral_tasks = None
    manager = object.__new__(DFlashAclGraphManager)
    manager.graphs, manager.speculator = {desc: graph}, spec
    spec.query_cudagraph_manager = manager
    monkeypatch.setattr(
        manager, "_make_neutral_cache", MagicMock(side_effect=AssertionError("unused neutral allocation"))
    )
    manager._prepare_neutral_tasks(desc)
    manager._make_neutral_cache.assert_not_called()
    assert graph._dflash_neutral_tasks is None
    # Exercise the real gate even if a graph from an earlier enabled mode
    # happened to retain legal neutral parameters.
    graph._dflash_neutral_tasks = (object(),)
    spec._draft_seq_lens_copy_count = 2
    monkeypatch.setattr(
        spec, "_can_defer_draft_metadata", AscendDFlashSpeculator._can_defer_draft_metadata.__get__(spec)
    )
    assert not spec._can_defer_draft_metadata(desc)
    mirror.copy_(torch.tensor([75, 80, 0, 0]))
    refreshed = _build(spec, desc)
    assert refreshed is not None and refreshed is not first
    assert refreshed["draft0"].seq_lens_list == [75, 80, 1, 1]
    assert getattr(spec, "_deferred_draft_attn_metadata", None) is None


def test_exact_unsupported_buffer_capacity_uses_original_build():
    spec = _speculator()
    spec._exact_draft_metadata_active = spec._use_cpu_seq_lens = True
    spec.input_buffers = SimpleNamespace(
        seq_lens=SimpleNamespace(
            device=SimpleNamespace(type="npu"),
            ndim=1,
            dtype=torch.int32,
            is_contiguous=lambda: True,
            numel=lambda: 4,
        )
    )
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    assert spec._can_use_exact_draft_metadata(desc, 4, 4, None)
    too_large = BatchExecutionDescriptor(CUDAGraphMode.FULL, 20, 5)
    assert not spec._can_use_exact_draft_metadata(too_large, 4, 4, None)
    assert not spec._can_use_exact_draft_metadata(desc, 4, 4, object())
    assert not spec._can_use_exact_draft_metadata(desc, 4, 1, None)


def test_failed_copy_cleanup_preserves_primary_proposal_exception(monkeypatch):
    spec = _speculator()
    spec._draft_metadata_template = ("old", {})
    spec._draft_seq_lens_copy_count = 2
    spec._draft_seq_lens_copy_event = SimpleNamespace(synchronize=MagicMock(side_effect=RuntimeError("copy cleanup")))
    monkeypatch.setattr(DFlashSpeculator, "propose", MagicMock(side_effect=ValueError("original proposal")))
    with pytest.raises(ValueError, match="original proposal") as caught:
        spec.propose(spec.input_batch, {}, {}, None, None, None, None, None, None, None, None)
    assert spec._draft_metadata_template is None
    assert not spec._exact_draft_metadata_active
    assert "copy cleanup" in caught.value.__notes__[0]
