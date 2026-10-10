# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.v2.mamba import graph_state


@pytest.mark.parametrize("width", [1, 3, 8, 17])
@pytest.mark.parametrize("reset_spec", [False, True])
def test_graph_state_writes_group_ids_and_clears_padding(width, reset_spec):
    updater = graph_state.GDNGraphStateUpdater()
    sources = torch.arange(24 * 8 * (width + 2), dtype=torch.int32).view(24, 8, width + 2)
    destinations = [torch.full((8, width), -99, dtype=torch.int32) for _ in range(24)]
    clears = [torch.full((8, 4), 99, dtype=torch.int32) if reset_spec else None for _ in range(24)]
    addresses = [dst.data_ptr() for dst in destinations]
    for actual_rows, graph_rows in [(5, 8), (1, 4), (0, 4), (8, 8)]:
        sources.add_(1000)
        before = [dst.clone() for dst in destinations]
        updater.apply(
            [
                (src[:, :width], dst, actual_rows, graph_rows, 0, clear)
                for src, dst, clear in zip(sources, destinations, clears)
            ]
        )
        for src, dst, previous, clear in zip(sources, destinations, before, clears):
            torch.testing.assert_close(dst[:actual_rows], src[:actual_rows, :width], rtol=0, atol=0)
            assert torch.all(dst[actual_rows:graph_rows] == 0)
            torch.testing.assert_close(dst[graph_rows:], previous[graph_rows:])
            if clear is not None:
                assert torch.all(clear[:graph_rows] == -1)
        assert [dst.data_ptr() for dst in destinations] == addresses


def test_graph_state_validates_all_groups_before_writing():
    source, destination = torch.ones(2, dtype=torch.int32), torch.full((4,), -99, dtype=torch.int32)
    with pytest.raises(ValueError):
        graph_state.GDNGraphStateUpdater().apply(
            [(source, destination, 2, 4, 0, None), (source, torch.empty(1, dtype=torch.int32), 2, 4, 0, None)]
        )
    assert torch.all(destination == -99)


def test_graph_state_only_reads_destination_columns():
    source = torch.arange(24, dtype=torch.int32).view(4, 6)
    destination = torch.full((4, 3), -99, dtype=torch.int32)
    graph_state.GDNGraphStateUpdater().apply([(source, destination, 2, 4, 0, None)])
    torch.testing.assert_close(destination[:2], source[:2, :3])
    assert torch.all(destination[2:] == 0)


def test_graph_state_launches_once_and_reuses_pointer_table(monkeypatch):
    updater = graph_state.GDNGraphStateUpdater()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    source, destination = MagicMock(), MagicMock()
    source.dtype = destination.dtype = torch.int32
    source.device = destination.device = SimpleNamespace(type="npu")
    source.ndim = destination.ndim = 2
    source.size.side_effect = destination.size.side_effect = lambda dim: (8, 4)[dim]
    source.stride.side_effect = lambda dim: (12, 1)[dim]
    source.data_ptr.return_value, destination.data_ptr.return_value = 1024, 2048
    destination.is_contiguous.return_value = True
    monkeypatch.setattr(updater, "_can_batch", lambda _: True)
    host, pointers, kernel = MagicMock(), MagicMock(), MagicMock()
    host.to.return_value = pointers
    tensor = MagicMock(return_value=host)
    monkeypatch.setattr(graph_state.torch, "tensor", tensor)
    monkeypatch.setattr(graph_state, "_graph_state_kernel", kernel)
    streams = [SimpleNamespace(npu_stream=index, wait_event=MagicMock()) for index in (1, 2)]
    event = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    for index, count in enumerate((5, 1, 0)):
        monkeypatch.setattr(torch.npu, "current_stream", lambda _, index=index: streams[index // 2])
        updater.apply([(source, destination, count, 8, 0, None)])
    event.record.assert_called_once_with(streams[0])
    streams[0].wait_event.assert_not_called()
    streams[1].wait_event.assert_called_once_with(event)
    assert (
        source.record_stream.call_count
        == destination.record_stream.call_count
        == pointers.record_stream.call_count
        == 2
    )
    tensor.assert_called_once()
    host.to.assert_called_once_with(destination.device, non_blocking=True)
    assert kernel.__getitem__.return_value.call_count == 3
    for call in kernel.__getitem__.return_value.call_args_list:
        assert call.args[0] is pointers
        assert call.kwargs["SOURCE_ROW_STRIDE"] == 12
        assert call.kwargs["multibuffer"] is False and call.kwargs["unit_flag"] is False


@pytest.mark.parametrize("case", ["adjacent", "alias", "stride", "different_geometry"])
def test_unsupported_graph_state_layout_uses_torch_reference(case, monkeypatch):
    src = torch.arange(16, dtype=torch.int32).view(4, 4)
    dst = torch.full((4, 4), -99, dtype=torch.int32)
    updates = [(src, dst, 2, 4, 0, None)]
    if case == "adjacent":
        storage = torch.full((8,), -99, dtype=torch.int32)
        updates = [(src[:, 0], storage[:3], 2, 3, 0, None), (src[:, 1], storage[3:6], 2, 3, 0, None)]
    elif case == "alias":
        updates = [(src, src, 2, 4, 0, None)]
    elif case == "stride":
        dst = torch.full((4, 8), -99, dtype=torch.int32)[:, ::2]
        updates = [(src, dst, 2, 4, 0, None)]
    else:
        updates.append((src[:, 0], torch.full((4,), -99, dtype=torch.int32), 1, 4, 0, None))
    updater = graph_state.GDNGraphStateUpdater()
    assert not updater._can_batch(updates)
    kernel = MagicMock(side_effect=AssertionError("Unsupported geometry must not launch Triton"))
    monkeypatch.setattr(graph_state, "_graph_state_kernel", kernel)
    updater.apply(updates)
    for source, destination, actual, graph, padding, _ in updates:
        torch.testing.assert_close(destination[:actual], source[:actual])
        assert torch.all(destination[actual:graph] == padding)
    assert updater._pointers is None


def test_reset_clears_full_capture_span_beyond_non_spec_graph_rows():
    source = torch.arange(4, dtype=torch.int32)
    output = torch.full((8,), -99, dtype=torch.int32)
    reset = torch.full((16, 4), 99, dtype=torch.int32)
    graph_state.GDNGraphStateUpdater().apply([(source, output, 1, 8, 0, reset)])
    assert torch.all(reset == -1)


@pytest.mark.parametrize("case", ["rank", "dtype", "columns", "reset"])
def test_invalid_graph_state_inputs_are_atomic(case):
    src = torch.ones((2, 4), dtype=torch.int32)
    dst = torch.full((4, 4), -99, dtype=torch.int32)
    reset = None
    bad_src = src
    if case == "rank":
        bad_src = src[:, 0]
    elif case == "dtype":
        bad_src = src.to(torch.int64)
    elif case == "columns":
        bad_src = src[:, :2]
    else:
        reset = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError):
        graph_state.GDNGraphStateUpdater().apply(
            [(src, dst, 2, 4, 0, None), (bad_src, torch.empty_like(dst), 2, 4, 0, reset)]
        )
    assert torch.all(dst == -99)
