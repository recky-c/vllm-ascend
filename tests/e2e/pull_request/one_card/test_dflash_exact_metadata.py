# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.test_prepare_dflash_inputs import (
    _allocate_outputs,
    _build_inputs,
    _impl_args,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder, FIAParamProvider
from vllm_ascend.compilation.updatable_graph import UpdatableGraph, register_task
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.attn_utils import _reshape_combined_attention_kv_cache
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
    dflash_inputs_prepared,
    prepare_dflash_inputs_factory,
)


def _builder(device, sliding, num_kv_heads=2, head_size=64):
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.device, builder.pcp_enabled, builder.decode_threshold = device, False, 4
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.vllm_config = SimpleNamespace(model_config=builder.model_config)
    builder.speculative_config = SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False)
    builder.supports_update_block_table = True
    builder.kv_cache_spec = (
        SlidingWindowSpec(
            block_size=128, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.bfloat16, sliding_window=4096
        )
        if sliding
        else FullAttentionSpec(block_size=128, num_kv_heads=num_kv_heads, head_size=head_size, dtype=torch.bfloat16)
    )
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1) if sliding else None
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda causal, config: mask)
    return builder, mask


@pytest.mark.parametrize(
    "cache_layout,deferred_metadata",
    [("contiguous", True), ("mixed", True), ("padded", True), ("contiguous", False)],
    ids=["contiguous", "mixed", "padded", "contiguous_metadata_first"],
)
def test_exact_producer_template_and_deferred_full_graph_four_rounds(monkeypatch, cache_layout, deferred_metadata):
    """Actual rejection producer, exact snapshot and production event ordering.

    No model weights are loaded. The graph prefix/FIA use real NPU operations,
    the real metadata producer and manager; only upstream graph dispatch is a
    single graph.replay call. This is component correctness, not throughput.
    """
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(18139)
    device = torch.device("npu")
    width, padded, max_tokens = 4, 4, 16
    kv_heads, head_size, query_heads = (2, 64, 4) if cache_layout == "contiguous" else (8, 128, 32)
    outputs = _allocate_outputs(padded, max_tokens, 3, device)
    builders, masks = zip(*[_builder(device, sliding, kv_heads, head_size) for sliding in (False, True)])
    spec = object.__new__(AscendDFlashSpeculator)
    spec.num_query_per_req = width
    spec.num_speculative_steps = 3
    spec.max_model_len = spec.draft_max_seq_len = 8192
    spec.arange_np = np.arange(padded + 1, dtype=np.int32)
    spec.input_buffers = outputs.input_buffers
    spec.draft_is_prefilling = torch.zeros(padded, dtype=torch.bool)
    spec.draft_kv_cache_group_ids = [1, 2]
    spec._group_causal = {1: False, 2: True}
    spec._use_cpu_seq_lens = spec._reuse_draft_layout = True
    spec._exact_draft_metadata_active = spec._reuse_draft_attn_metadata = True
    spec._deferred_metadata_enabled = deferred_metadata
    spec._draft_metadata_template = None
    spec.attn_backends = {"draft0": object(), "draft1": object()}
    groups = [
        SimpleNamespace(layer_names=[f"draft{i}"], get_metadata_builder=lambda _, b=b: b)
        for i, b in enumerate(builders)
    ]
    spec.attn_groups = [[], [groups[0]], [groups[1]]]
    specs = [
        MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,)),
        builders[0].kv_cache_spec,
        builders[1].kv_cache_spec,
    ]
    spec.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs])
    spec.block_tables = SimpleNamespace(
        cp_size=1,
        input_block_tables=[torch.zeros(padded, 64, dtype=torch.int32, device=device) for _ in specs],
        slot_mappings=torch.full((3, max_tokens), -1, dtype=torch.int32, device=device),
    )
    backing = []
    caches = []
    for _ in range(2):
        if cache_layout == "contiguous":
            pair = tuple(torch.randn(67, 128, 128, dtype=torch.bfloat16, device=device) for _ in range(2))
            backing.extend(pair)
        else:
            # Actual mixed-model page geometry, scaled only in page count:
            # BF16 [blocks,128,1024], stride[262144,1024,1], K offset0,
            # V offset131072, physical640 split into five kernel128 blocks.
            page_elements = 262144 + (2048 if cache_layout == "padded" else 0)
            raw = torch.randn(70 * page_elements, dtype=torch.bfloat16, device=device)
            views = _reshape_combined_attention_kv_cache(
                raw.view(torch.uint8), (2, 70, 128, kv_heads, head_size), torch.bfloat16, page_elements * 2 * 5, 5
            )
            pair = tuple(view.view(70, 128, -1) for view in views)
            assert pair[0].stride() == pair[1].stride() == (page_elements, 1024, 1)
            assert pair[0].storage_offset() == 0 and pair[1].storage_offset() == 131072
            assert torch_npu.get_npu_format(pair[0]) == torch_npu.get_npu_format(pair[1]) == 2
            if cache_layout == "padded":
                raw.view(70, page_elements)[:, 262144:].fill_(19)
            backing.append(raw)
        caches.append(pair)
    for key, value in caches:
        key[0].zero_()
        value[0].zero_()
    raw_query = torch.randn(max_tokens, query_heads, head_size, dtype=torch.bfloat16, device=device)
    query = torch.empty_like(raw_query)
    torch.mul(raw_query, 1.01, out=query)
    prefix_expected_cpu = query.cpu()
    result = [torch.empty_like(query) for _ in range(2)]
    kwargs = []
    for index, (key, value) in enumerate(caches):
        sliding = bool(index)
        kwargs.append(
            dict(
                query=query,
                key=key,
                value=value,
                atten_mask=masks[index],
                block_table=spec.block_tables.input_block_tables[index + 1],
                input_layout="TND",
                block_size=128,
                actual_seq_lengths=[4, 8, 12, 16],
                actual_seq_lengths_kv=[128, 8185, 1, 1],
                num_key_value_heads=kv_heads,
                num_heads=query_heads,
                pre_tokens=4096 if sliding else 2147483647,
                next_tokens=0 if sliding else 2147483647,
                scale=head_size**-0.5,
                sparse_mode=4 if sliding else 0,
            )
        )
    graph = UpdatableGraph()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    prefix_ready = torch.npu.Event()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, max_tokens, padded)
    manager = object.__new__(DFlashAclGraphManager)
    manager.graphs, manager.speculator, manager.update_stream = {desc: graph}, spec, update_stream
    spec.query_cudagraph_manager = manager
    monkeypatch.setattr(
        DFlashCudaGraphManager, "run_fullgraph", lambda self, descriptor: self.graphs[descriptor].replay()
    )
    # The fallback selector only decides legacy/updatable dispatch, not event ordering.
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.dflash.aclgraph.use_updatable_graph", lambda _: True)
    rounds = [([120, 8180], [0, 3]), ([125, 4095], [1, 0]), ([126], [2]), ([118, 8188, 64], [0, 2, 1])]
    factory = prepare_dflash_inputs_factory(128)
    copy_addresses = []
    deferred_rounds = []
    copies = []
    start_copy = spec._start_draft_seq_lens_copy

    def track_copy(count):
        copies.append(count)
        start_copy(count)

    monkeypatch.setattr(spec, "_start_draft_seq_lens_copy", track_copy)
    tolist = torch.Tensor.tolist

    def no_device_length_readback(tensor):
        assert tensor.data_ptr() != spec.input_buffers.seq_lens.data_ptr(), "exact draft lengths were read back again"
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", no_device_length_readback)

    def produce(starts, rejected, phase):
        count = len(starts)
        case = dict(
            req_lens=[4] * count,
            position_starts=starts,
            idx_mapping=list(range(count)),
            num_rejected=rejected,
            max_num_reqs=padded,
            max_num_tokens=max_tokens,
            max_model_len=8192,
            block_size=128,
            num_query_per_req=4,
            num_speculative_steps=3,
            parallel_drafting_token_id=151669,
        )
        data = _build_inputs(case, device)
        data.outputs = outputs
        spec.input_batch = data.input_batch
        spec.input_batch.seq_lens_cpu_upper_bound = torch.tensor([start + 4 for start in starts], dtype=torch.int32)
        spec._draft_input_groups_prepared = 0
        spec._draft_attn_metadata_for_graph = None
        for index in (1, 2):
            table = spec.block_tables.input_block_tables[index]
            table.zero_()
            table[0, :2].copy_(torch.tensor([1, 2] if phase % 2 == 0 else [2, 1], dtype=torch.int32, device=device))
            if count > 1:
                table[1].copy_(torch.roll(torch.arange(3, 67, dtype=torch.int32, device=device), phase))
                if index == 2 and starts[1] > 4096:
                    table[1, :31].zero_()
            if count > 2:
                table[2, :2].copy_(torch.tensor([1, 2], dtype=torch.int32, device=device))
        with dflash_inputs_prepared(spec._on_draft_inputs_prepared):
            for index in (1, 2):
                data.block_table = spec.block_tables.input_block_tables[index]
                args = _impl_args(data, case)
                args[1] = spec.block_tables.slot_mappings[index]
                factory(*args)
                if index == 1:
                    assert getattr(spec, "_draft_seq_lens_copy_count", None) is None
        assert spec._draft_seq_lens_copy_count == count
        expected = [min(start + 8 - reject, 8192) for start, reject in zip(starts, rejected)]
        return expected + [1] * (padded - count)

    for phase, (starts, rejected) in enumerate(rounds):
        expected = produce(starts, rejected, phase)
        metadata = spec._build_uniform_attn_metadata(
            desc, len(starts), width, spec.input_batch.seq_lens_cpu_upper_bound, width, spec._group_causal
        )
        if phase == 0:
            assert metadata is not None
            spec._update_draft_attn_metadata(metadata, padded)
            for index in range(2):
                kwargs[index]["actual_seq_lengths_kv"] = expected
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs[index])
                kwargs[index].update(
                    workspace=workspace, out=[result[index], torch.empty(1, dtype=query.dtype, device=device)]
                )
            with torch.npu.stream(capture_stream):
                capture_stream.wait_stream(torch.npu.default_stream())
                torch.mul(raw_query, 1.01, out=query)
                for arguments in kwargs:
                    torch_npu.npu_fused_infer_attention_score.out(**arguments)
                capture_stream.synchronize()
                with torch.npu.graph(graph, stream=capture_stream):
                    torch.mul(raw_query, 1.01, out=query)
                    prefix_ready.record()
                    for index, arguments in enumerate(kwargs):
                        register_task(
                            torch_npu.npu_fused_infer_attention_score.out,
                            arguments,
                            FIAParamProvider(f"draft{index}", 4096 if index else None, is_draft_model=True),
                        )
            torch.npu.synchronize()
        pending = spec._deferred_draft_attn_metadata if hasattr(spec, "_deferred_draft_attn_metadata") else None
        deferred_rounds.append(pending is not None)
        original = spec.build_draft_attn_metadatas

        def verify_prefix(*args, pending=pending, original=original, expected=expected):
            if pending is not None:
                # This regular event is only a readiness hint: capture may
                # leave it completed. The fresh sentinel check below proves
                # this replay's prefix progress before any FIA update.
                prefix_ready.synchronize()
                # A fresh sentinel prevents an already-completed event or old
                # prefix output from making this ordering assertion vacuous.
                deadline = time.monotonic() + 5
                observed = query.cpu()
                while not torch.equal(observed, prefix_expected_cpu) and time.monotonic() < deadline:
                    observed = query.cpu()
                torch.testing.assert_close(observed, prefix_expected_cpu, rtol=0, atol=0)
            current = original(*args)
            for item in current[0].values():
                assert item.seq_lens_list == expected
                assert item.seq_lens_gpu.data_ptr() == spec.input_buffers.seq_lens.data_ptr()
            return current

        query.fill_(-99)
        monkeypatch.setattr(spec, "build_draft_attn_metadatas", verify_prefix)
        manager.run_fullgraph(desc)
        monkeypatch.setattr(spec, "build_draft_attn_metadatas", original)
        torch.npu.synchronize()
        copy_addresses.append(spec._draft_seq_lens_cpu.data_ptr())
        for index, arguments in enumerate(kwargs):
            eager = arguments.copy()
            eager.pop("workspace")
            eager.pop("out")
            eager["actual_seq_lengths_kv"] = expected
            eager["block_table"] = spec.block_tables.input_block_tables[index + 1]
            reference, _ = torch_npu.npu_fused_infer_attention_score(**eager)
            assert torch.isfinite(result[index]).all()
            torch.testing.assert_close(result[index], reference, rtol=0, atol=0)
    assert deferred_rounds == [False, deferred_metadata, False, False]
    assert len(set(copy_addresses)) == 1
    if not deferred_metadata:
        assert getattr(spec, "_deferred_draft_attn_metadata", None) is None
        assert getattr(graph, "_dflash_neutral_tasks", None) is None
        return
    neutral_tasks = graph._dflash_neutral_tasks
    assert len(neutral_tasks) == 2
    for index, (task, (key, value)) in enumerate(zip(neutral_tasks, caches)):
        assert task.kwargs["workspace"].data_ptr() == kwargs[index]["workspace"].data_ptr()
        for name, live in (("key", key), ("value", value)):
            neutral = task.kwargs[name]
            assert neutral.stride() == live.stride()
            assert neutral.storage_offset() == 0
            assert neutral.untyped_storage().nbytes() == live.stride(0) * live.element_size()
            assert neutral.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
            assert torch_npu.get_npu_format(neutral) == 2
    # One subsequent same-layout proposal is deferred; host preparation fails
    # after replay. Independent neutral KV releases all events without reading
    # old lengths from rebound tables, then the original error propagates.
    old_cache = [(key.clone(), value.clone()) for key, value in caches]
    old_backing = [tensor.clone() for tensor in backing]
    produce(*rounds[-1], phase=4)
    prepared_tables = [tensor.clone() for tensor in spec.block_tables.input_block_tables]
    assert (
        spec._build_uniform_attn_metadata(
            desc, 3, width, spec.input_batch.seq_lens_cpu_upper_bound, width, spec._group_causal
        )
        is None
    )

    def fail(*args):
        raise ValueError("injected metadata failure")

    monkeypatch.setattr(spec, "build_draft_attn_metadatas", fail)
    with pytest.raises(ValueError, match="injected metadata failure"):
        manager.run_fullgraph(desc)
    monkeypatch.setattr(spec, "build_draft_attn_metadatas", original)
    assert spec._draft_metadata_template is None
    for output in result:
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output) == 0
    for current, previous in zip(caches, old_cache):
        for tensor, old in zip(current, previous):
            torch.testing.assert_close(tensor, old, rtol=0, atol=0)
    for tensor, previous in zip(backing, old_backing):
        torch.testing.assert_close(tensor, previous, rtol=0, atol=0)
    for tensor, previous in zip(spec.block_tables.input_block_tables, prepared_tables):
        torch.testing.assert_close(tensor, previous, rtol=0, atol=0)
    spec._finish_draft_seq_lens_copy()
    # Layout invalidation forces metadata-first order and the next graph can resume.
    expected = produce(*rounds[0], phase=5)
    assert (
        spec._build_uniform_attn_metadata(
            desc, 2, width, spec.input_batch.seq_lens_cpu_upper_bound, width, spec._group_causal
        )
        is not None
    )
    manager.run_fullgraph(desc)
    torch.npu.synchronize()
    for index, arguments in enumerate(kwargs):
        eager = arguments.copy()
        eager.pop("workspace")
        eager.pop("out")
        eager["actual_seq_lengths_kv"] = expected
        eager["block_table"] = spec.block_tables.input_block_tables[index + 1]
        reference, _ = torch_npu.npu_fused_infer_attention_score(**eager)
        assert torch.isfinite(result[index]).all()
        torch.testing.assert_close(result[index], reference, rtol=0, atol=0)
    assert copies == [2, 2, 1, 3, 3, 2]
