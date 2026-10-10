# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops import gdn_attn_builder
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops.triton.v2.mamba.graph_state import GDNGraphStateUpdater
from vllm_ascend.worker.v2.model_states.mamba_hybrid import _compute_aligned_state_indices


@pytest.fixture(autouse=True)
def _graph_state_component_config(monkeypatch):
    # Component tests create lightweight runner configs without a model worker.
    monkeypatch.setattr(
        gdn_attn_builder, "get_ascend_config", lambda: SimpleNamespace(enable_gdn_graph_state_batching=True)
    )


@pytest.mark.parametrize("num_groups", [1, 24])
@pytest.mark.parametrize("num_reqs", [0, 1, 17, 33])
@pytest.mark.parametrize("num_state_slots", [1, 3, 4, 7, 8, 16, 17])
@pytest.mark.parametrize("max_reqs", [33, 64])
def test_aligned_state_indices_match_physical_tables(num_groups, num_reqs, num_state_slots, max_reqs):
    torch.npu.set_device(0)
    columns = 32
    # Retain a non-contiguous row stride, as runner block-table views do.
    tables = torch.arange(num_groups * max_reqs * columns * 2, dtype=torch.int32, device="npu").view(
        num_groups, max_reqs, columns * 2
    )[:, :, :columns]
    lengths_cpu = (torch.arange(max_reqs, dtype=torch.int64) * 7 % (columns - num_state_slots + 1)) * 16 + 1
    lengths_cpu[::5] = 0
    seq_lens = torch.zeros(max_reqs * 2, dtype=torch.int64, device="npu")
    seq_lens[::2].copy_(lengths_cpu)
    output = torch.full((num_groups, max_reqs, num_state_slots), -99, dtype=torch.int32, device="npu")
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=tables.stride(1),
        block_size=16,
        num_groups=num_groups,
        aligned_state_indices=output,
    )
    actual = _compute_aligned_state_indices(ctx, seq_lens[::2], num_reqs, columns)
    slots = ((lengths_cpu[:num_reqs] - 1) // 16).clamp_min(0)[:, None] + torch.arange(num_state_slots)
    expected = torch.stack([table.cpu().gather(1, slots.long()) for table in tables])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:, num_reqs:].cpu() == -99)


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("max_reqs,slots", [(64, 3), (64, 16), (33, 4), (33, 1), (33, 3), (33, 8)])
def test_aligned_state_indices_preserve_int32_bits(max_reqs, slots, seq_dtype):
    torch.npu.set_device(0)
    num_groups, num_reqs, columns = 24, 17, 32
    # Include IDs beyond exact fp32 integer precision, subnormal and NaN bits.
    values = torch.tensor([0, 1, -1, 2**24 + 1, 2**31 - 1, -(2**31), 0x7F800001, -0x7FFFFF], dtype=torch.int32)
    tables = values.repeat(num_groups * max_reqs * columns // values.numel()).view(num_groups, max_reqs, columns)
    tables = tables.to("npu")
    lengths = (torch.arange(max_reqs, dtype=seq_dtype) % (columns - slots + 1)) * 16 + 1
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=columns,
        block_size=16,
        num_groups=num_groups,
        aligned_state_indices=torch.full((num_groups, max_reqs, slots), -99, dtype=torch.int32, device="npu"),
    )
    actual = _compute_aligned_state_indices(ctx, lengths.to("npu"), num_reqs, columns)
    cols = (lengths[:num_reqs, None] - 1) // 16 + torch.arange(slots)
    expected = torch.stack([table.cpu().gather(1, cols.long()) for table in tables])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.all(ctx.aligned_state_indices[:, num_reqs:].cpu() == -99)


@pytest.mark.parametrize("num_groups", [1, 24])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("seq_length", [0, 1, 113, 129])
def test_aligned_state_indices_single_row_window(num_groups, seq_dtype, seq_length):
    torch.npu.set_device(0)
    max_reqs, columns, slots = 33, 32, 8
    tables = torch.arange(num_groups * max_reqs * columns * 2, dtype=torch.int32, device="npu").view(
        num_groups, max_reqs, columns * 2
    )[:, :, :columns]
    seq_lens = torch.full((max_reqs * 2,), seq_length, dtype=seq_dtype, device="npu")[::2]
    output = torch.full((num_groups, max_reqs, slots), -99, dtype=torch.int32, device="npu")
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=tables.stride(1),
        block_size=16,
        num_groups=num_groups,
        aligned_state_indices=output,
    )
    actual = _compute_aligned_state_indices(ctx, seq_lens, 1, columns)
    first_slot = max((seq_length - 1) // 16, 0)
    expected = tables[:, :1, first_slot : first_slot + slots]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.all(output[:, 1:].cpu() == -99)


@pytest.mark.parametrize("num_reqs", [1, 17])
@pytest.mark.parametrize("max_reqs,slots", [(64, 3), (64, 16), (33, 4), (33, 8)])
def test_aligned_state_indices_aclgraph_replay(max_reqs, slots, num_reqs):
    torch.npu.set_device(0)
    num_groups, columns = 24, 32
    tables = torch.arange(num_groups * max_reqs * columns, dtype=torch.int32, device="npu").view(
        num_groups, max_reqs, columns
    )
    seq_lens = torch.ones(max_reqs, dtype=torch.int32, device="npu")
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=columns,
        block_size=16,
        num_groups=num_groups,
        aligned_state_indices=torch.full((num_groups, max_reqs, slots), -99, dtype=torch.int32, device="npu"),
    )
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    snapshots = []
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        _compute_aligned_state_indices(ctx, seq_lens, num_reqs, columns)
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            _compute_aligned_state_indices(ctx, seq_lens, num_reqs, columns)
        for step in range(1, 9):
            lengths = (torch.arange(max_reqs, dtype=torch.int32) * step % (columns - slots + 1)) * 16 + 1
            seq_lens.copy_(lengths)
            tables.add_(100)
            graph.replay()
            snapshots.append((step, lengths, ctx.aligned_state_indices.clone()))
    torch.npu.synchronize()
    original = torch.arange(num_groups * max_reqs * columns, dtype=torch.int32).view(num_groups, max_reqs, columns)
    for step, lengths, output in snapshots:
        cols = (lengths[:num_reqs, None] - 1) // 16 + torch.arange(slots)
        expected = torch.stack([(table + step * 100).gather(1, cols.long()) for table in original])
        torch.testing.assert_close(output[:, :num_reqs].cpu(), expected, rtol=0, atol=0)
        assert torch.all(output[:, num_reqs:].cpu() == -99)


@pytest.mark.parametrize("num_groups", [5, 24])
def test_gdn_shared_metadata_aclgraph_reads_updated_group_states(num_groups):
    torch.npu.set_device(0)
    graph_reqs, width = 16, 4
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,), num_speculative_blocks=3)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        additional_config=None,
        model_config=SimpleNamespace(max_model_len=1024),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=1024),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=3, parallel_drafting=False),
    )
    builders = [
        AscendGDNAttentionMetadataBuilder(spec, [f"layer{i}"], config, torch.device("npu")) for i in range(num_groups)
    ]
    table = torch.arange(graph_reqs * width, dtype=torch.int32, device="npu").view(graph_reqs, width)
    group_tables = [table + i * 1000 for i in range(num_groups)]

    def prepare(count, step):
        query_cpu = torch.tensor([min(i, count) * width for i in range(graph_reqs + 1)], dtype=torch.int32)
        common = AscendCommonAttentionMetadata(
            query_start_loc=query_cpu.to("npu"),
            query_start_loc_cpu=query_cpu,
            seq_lens=torch.full((graph_reqs,), 32, dtype=torch.int32, device="npu"),
            seq_lens_cpu_upper_bound=torch.full((graph_reqs,), 32, dtype=torch.int32),
            num_reqs=graph_reqs,
            num_actual_tokens=count * width,
            max_query_len=width,
            max_seq_len=1024,
            block_table_tensor=group_tables[0],
            slot_mapping=torch.empty(graph_reqs * width, dtype=torch.int64, device="npu"),
            is_prefilling=torch.zeros(graph_reqs, dtype=torch.bool),
        )
        accepted = torch.full((graph_reqs,), step, dtype=torch.int32, device="npu")
        drafts = torch.tensor([3] * count + [-1] * (graph_reqs - count), dtype=torch.int32)
        first = builders[0].build(0, common, accepted, drafts, num_actual_reqs=count)
        updates = []
        metadata = [first] + [
            builder.update_block_table(first, bt, graph_state_updates=updates)
            for builder, bt in zip(builders[1:], group_tables[1:])
        ]
        builders[0].graph_state_updater.apply(updates)
        assert builders[0].graph_state_updater._pointers is not None
        return metadata

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    state_outputs = torch.empty((num_groups, graph_reqs, width), dtype=torch.int32, device="npu")
    accepted_output = torch.empty(graph_reqs, dtype=torch.int32, device="npu")
    query_output = torch.empty(graph_reqs + 1, dtype=torch.int32, device="npu")
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        captured = prepare(graph_reqs, 1)

        def consume():
            for i, metadata in enumerate(captured):
                state_outputs[i].copy_(metadata.spec_state_indices_tensor)
            accepted_output.copy_(captured[0].num_accepted_tokens)
            query_output.copy_(captured[0].spec_query_start_loc)

        consume()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            consume()
        snapshots = []
        for step, count in enumerate([1, 4, 16, 0, 16], start=2):
            for bt in group_tables:
                bt.add_(100)
            current = prepare(count, step)
            for before, after in zip(captured, current):
                if count:
                    assert before.spec_state_indices_tensor.data_ptr() == after.spec_state_indices_tensor.data_ptr()
                    assert before.num_accepted_tokens.data_ptr() == after.num_accepted_tokens.data_ptr()
            graph.replay()
            snapshots.append((count, step, state_outputs.clone(), accepted_output.clone(), query_output.clone()))
    torch.npu.synchronize()
    for count, step, states, accepted, query in snapshots:
        expected = torch.stack([table.cpu() + i * 1000 + (step - 1) * 100 for i in range(num_groups)])
        if count:
            expected[:, count:].zero_()
        else:
            expected.fill_(-1)
        torch.testing.assert_close(states.cpu(), expected, rtol=0, atol=0)
        assert accepted.cpu().tolist() == ([step] * count + [1] * (graph_reqs - count) if count else [0] * graph_reqs)
        assert query.cpu().tolist() == [min(i, count) * width for i in range(graph_reqs + 1)]


@pytest.mark.parametrize("groups", [4, 23])
@pytest.mark.parametrize("width", [1, 3, 4, 17])
@pytest.mark.parametrize("reset_spec", [False, True])
def test_batched_graph_state_native_replay_preserves_int32_and_padding(groups, width, reset_spec):
    torch.npu.set_device(0)
    capacity = 17  # Exercise partial store sectors for odd widths.
    values = torch.tensor([0, 1, -1, 2**24 + 1, 2**31 - 1, -(2**31), 0x7F800001, -0x7FFFFF], dtype=torch.int32)
    count = groups * capacity * (width + 2)
    tables = (
        values.repeat((count + values.numel() - 1) // values.numel())[:count]
        .view(groups, capacity, width + 2)
        .to("npu")
    )
    outputs = [torch.full((capacity, width), -99, dtype=torch.int32, device="npu") for _ in range(groups)]
    resets = [torch.full((capacity, 4), 99, dtype=torch.int32, device="npu") if reset_spec else None for _ in outputs]
    updater = GDNGraphStateUpdater()
    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
    snapshots = torch.empty((groups, capacity, width), dtype=torch.int32, device="npu")
    reset_snapshots = torch.empty((groups, capacity, 4), dtype=torch.int32, device="npu")
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        updater.apply(
            [(src[:, :width], dst, 16, capacity, 0, clear) for src, dst, clear in zip(tables, outputs, resets)]
        )
        assert updater._pointers is not None
        original_pointers = updater._pointers

        def consume():
            for index, (dst, clear) in enumerate(zip(outputs, resets)):
                snapshots[index].copy_(dst)
                if clear is not None:
                    reset_snapshots[index].copy_(clear)

        consume()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            consume()
        for step, actual in enumerate([1, 4, 16, 0, 16]):
            tables.copy_(torch.roll(tables, shifts=1, dims=2))
            updater.apply(
                [(src[:, :width], dst, actual, capacity, 0, clear) for src, dst, clear in zip(tables, outputs, resets)]
            )
            assert updater._pointers is original_pointers
            graph.replay()
            torch.npu.synchronize()
            expected = tables[:, :, :width].cpu()
            expected[:, actual:].zero_()
            torch.testing.assert_close(snapshots.cpu(), expected, rtol=0, atol=0)
            if reset_spec:
                assert torch.all(reset_snapshots.cpu() == -1)


def test_native_graph_state_full_reset_span_and_alias_fallback():
    torch.npu.set_device(0)
    src = torch.arange(16, dtype=torch.int32, device="npu")
    dst = torch.full((8,), -99, dtype=torch.int32, device="npu")
    reset = torch.full((16, 4), 99, dtype=torch.int32, device="npu")
    updater = GDNGraphStateUpdater()
    updater.apply([(src, dst, 1, 8, 0, reset)])
    assert updater._pointers is not None
    torch.testing.assert_close(dst.cpu(), torch.tensor([0] * 8, dtype=torch.int32), rtol=0, atol=0)
    assert torch.all(reset.cpu() == -1)
    # Adjacent subviews share a partial 32-byte store sector, so use Torch.
    shared = torch.full((8,), -99, dtype=torch.int32, device="npu")
    fallback = GDNGraphStateUpdater()
    fallback.apply([(src, shared[:3], 2, 3, 0, None), (src + 100, shared[3:6], 2, 3, 0, None)])
    assert fallback._pointers is None
    torch.testing.assert_close(shared.cpu(), torch.tensor([0, 1, 0, 100, 101, 0, -99, -99], dtype=torch.int32))


@pytest.mark.parametrize("num_groups", [5, 24])
@pytest.mark.parametrize("draft_count", [None, 0])
def test_gdn_non_spec_and_zero_draft_native_chain_matches_per_group_reference(num_groups, draft_count):
    torch.npu.set_device(0)
    capacity = 16
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,), num_speculative_blocks=3)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        additional_config=None,
        model_config=SimpleNamespace(max_model_len=1024),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=1024),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=3, parallel_drafting=False),
    )
    builders = [
        AscendGDNAttentionMetadataBuilder(spec, [f"layer{i}"], config, torch.device("npu"))
        for i in range(num_groups + 1)
    ]
    owner, reference, *others = builders
    tables = torch.arange(num_groups * capacity * 6, dtype=torch.int32, device="npu").view(num_groups, capacity, 6)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    output_shape = (num_groups - 1, capacity) if draft_count is None else (num_groups - 1, capacity, 4)
    captured_outputs = torch.empty(output_shape, dtype=torch.int32, device="npu")
    reset_outputs = torch.empty((num_groups - 1, capacity, 4), dtype=torch.int32, device="npu")

    def consume():
        for index, builder in enumerate(others):
            state = builder.non_spec_state_indices_tensor if draft_count is None else builder.spec_state_indices_tensor
            captured_outputs[index].copy_(state[:capacity])
            reset_outputs[index].copy_(builder.spec_state_indices_tensor[:capacity])

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        for step, actual in enumerate([16, 1, 4, 16, 0, 16]):
            tables.add_(100)
            query = torch.tensor([min(index, actual) for index in range(capacity + 1)], dtype=torch.int32)
            lengths = torch.tensor([32] * actual + [0] * (capacity - actual), dtype=torch.int32)
            common = AscendCommonAttentionMetadata(
                query_start_loc=query.to("npu"),
                query_start_loc_cpu=query,
                seq_lens=lengths.to("npu"),
                seq_lens_cpu_upper_bound=lengths,
                num_reqs=capacity,
                num_actual_tokens=actual,
                max_query_len=1,
                max_seq_len=1024,
                block_table_tensor=tables[0],
                slot_mapping=torch.empty(capacity, dtype=torch.int64, device="npu"),
                is_prefilling=torch.zeros(capacity, dtype=torch.bool),
            )
            drafts = (
                None
                if draft_count is None
                else torch.tensor([0] * actual + [-1] * (capacity - actual), dtype=torch.int32)
            )
            accepted = None if drafts is None else torch.full((capacity,), 2, dtype=torch.int32, device="npu")
            first = owner.build(0, common, accepted, drafts, num_actual_reqs=actual)
            updates = []
            for builder, table in zip(others, tables[1:]):
                builder.spec_state_indices_tensor[:capacity].fill_(99)
                builder.update_block_table(first, table, graph_state_updates=updates)
            owner.graph_state_updater.apply(updates)
            assert owner.graph_state_updater._pointers is not None
            if step == 0:
                consume()
                torch.npu.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    consume()
            graph.replay()
            torch.npu.synchronize()
            for index, table in enumerate(tables[1:]):
                expected = reference.update_block_table(first, table)
                expected_state = (
                    expected.non_spec_state_indices_tensor
                    if draft_count is None
                    else reference.spec_state_indices_tensor[:capacity]
                )
                torch.testing.assert_close(captured_outputs[index].cpu(), expected_state.cpu(), rtol=0, atol=0)
            if draft_count is None or actual == 0:
                assert torch.all(reset_outputs.cpu() == -1)
            else:
                # Zero-draft MRV2 rows retain accepted-token offsets and remain
                # speculative; they must keep their current physical state IDs.
                assert first.num_spec_decodes == actual
                assert torch.all(first.num_accepted_tokens[:actual].cpu() == 2)


@pytest.mark.parametrize("groups", [4, 23])
def test_graph_state_native_pointer_refresh_across_streams_and_source_strides(groups):
    torch.npu.set_device(0)
    updater = GDNGraphStateUpdater()
    capacity, width = 16, 4
    outputs = [torch.full((capacity, width), -99, dtype=torch.int32, device="npu") for _ in range(groups)]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    previous_stream = torch.npu.default_stream()
    for step in range(6):
        # Separate allocations exercise pointer-cache replacement and positive
        # column stride; sources are allocated on a different stream.
        values = (
            torch.arange(groups * capacity * width * 2, dtype=torch.int32, device="npu").view(
                groups, capacity, width * 2
            )
            + 2**24
            + step * 10000
        )
        sources = [table[:, ::2] for table in values]
        stream = streams[step % 2]
        with torch.npu.stream(stream):
            stream.wait_stream(torch.npu.default_stream())
            stream.wait_stream(previous_stream)
            updater.apply(
                [(source, destination, 4, capacity, 0, None) for source, destination in zip(sources, outputs)]
            )
            assert updater._recorded_stream == stream
        previous_stream = stream
        torch.npu.synchronize()
        for source, destination in zip(sources, outputs):
            expected = source.cpu()
            expected[4:].zero_()
            torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    # Strided destinations keep adjacent untouched columns and fall back.
    storage = torch.full((capacity, width * 2), -99, dtype=torch.int32, device="npu")
    destination = storage[:, ::2]
    fallback = GDNGraphStateUpdater()
    fallback.apply([(sources[0], destination, 4, capacity, 0, None)])
    assert fallback._pointers is None
    assert torch.all(storage[:, 1::2].cpu() == -99)


def test_graph_state_pointer_table_owns_cross_stream_h2d_readiness():
    torch.npu.set_device(0)
    updater = GDNGraphStateUpdater()
    capacity, width, groups = 16, 4, 4
    tables = torch.arange(groups * capacity * width, dtype=torch.int32, device="npu").view(groups, capacity, width)
    outputs = [torch.full((capacity, width), -99, dtype=torch.int32, device="npu") for _ in range(groups)]
    sources = list(tables)
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(creator):
        creator.wait_stream(torch.npu.default_stream())
        # Initialize pointer metadata with no state writes, then immediately
        # use it on another stream without caller waiting for the creator.
        updater.apply([(source, destination, 0, 0, 0, None) for source, destination in zip(sources, outputs)])
    event = updater._pointer_ready_event
    pointers = updater._pointers
    with torch.npu.stream(consumer):
        consumer.wait_stream(torch.npu.default_stream())  # Source readiness is caller-owned.
        updater.apply([(source, destination, 4, capacity, 0, None) for source, destination in zip(sources, outputs)])
    assert updater._pointer_ready_event is event
    assert updater._pointers is pointers
    assert updater._pointer_ready_streams == {creator.npu_stream, consumer.npu_stream}
    torch.npu.synchronize()
    for source, destination in zip(sources, outputs):
        expected = source.cpu()
        expected[4:].zero_()
        torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)


def test_graph_state_first_use_during_capture_replays_after_external_metadata_update():
    torch.npu.set_device(0)
    updater = GDNGraphStateUpdater()
    capacity, width = 16, 4
    source = torch.arange(capacity * width, dtype=torch.int32, device="npu").view(capacity, width)
    destination = torch.full((capacity, width), -99, dtype=torch.int32, device="npu")
    reset = torch.full((capacity, width), 99, dtype=torch.int32, device="npu")
    snapshot = torch.empty_like(destination)
    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
    updates = [(source, destination, 4, capacity, 0, reset)]
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())  # Caller-owned source readiness.
        # Warm the original Torch ops without creating updater pointer metadata.
        GDNGraphStateUpdater._apply_reference(updates)
        snapshot.copy_(destination)
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            updater.apply(updates)
            snapshot.copy_(destination)
        assert updater._pointers is None
        assert updater._pointer_ready_event is None
        source.add_(1000)
        # This is before the first replay. Its event is recorded outside capture,
        # so metadata preparation cannot wait on a future graph event.
        updater.apply(updates)
        assert updater._pointers is not None
        assert updater._pointer_ready_event is not None
        graph.replay()
    torch.npu.synchronize()
    expected = source.cpu()
    expected[4:].zero_()
    torch.testing.assert_close(snapshot.cpu(), expected, rtol=0, atol=0)
    assert torch.all(reset.cpu() == -1)
