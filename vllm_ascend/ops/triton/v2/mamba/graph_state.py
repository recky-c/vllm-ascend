# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Batch group-local GDN graph input writes without changing captured storage."""

from bisect import bisect_left

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

_BLOCK_SIZE = 256
_STORE_ALIGNMENT_BYTES = 32


@triton.jit(do_not_specialize=["actual_rows", "graph_rows", "clear_rows"])
def _graph_state_kernel(
    pointers,
    actual_rows,
    graph_rows,
    clear_rows,
    WIDTH: tl.constexpr,
    SOURCE_ROW_STRIDE: tl.constexpr,
    SOURCE_COL_STRIDE: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    CLEAR_WIDTH: tl.constexpr,
    CLEAR_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    source = tl.load(pointers + group * 3).to(tl.pointer_type(tl.int32))
    destination = tl.load(pointers + group * 3 + 1).to(tl.pointer_type(tl.int32))
    for tile in range(tl.cdiv(graph_rows * WIDTH, BLOCK)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        rows = offsets // WIDTH
        columns = offsets % WIDTH
        values = tl.load(
            source + rows.to(tl.int64) * SOURCE_ROW_STRIDE + columns.to(tl.int64) * SOURCE_COL_STRIDE,
            (offsets < graph_rows * WIDTH) & (rows < actual_rows),
            other=PAD_VALUE,
        )
        # One program owns each allocation, including its partial 32-byte tail.
        tl.store(destination + offsets, values, offsets < graph_rows * WIDTH)
    if CLEAR_WIDTH:
        clear = tl.load(pointers + group * 3 + 2).to(tl.pointer_type(tl.int32))
        for tile in range(tl.cdiv(clear_rows * CLEAR_WIDTH, BLOCK)):
            offsets = tile * BLOCK + tl.arange(0, BLOCK)
            tl.store(clear + offsets, CLEAR_VALUE, offsets < clear_rows * CLEAR_WIDTH)


class GDNGraphStateUpdater:
    """Write stable group buffers, retaining the allocations behind pointers.

    Each reset tensor is already sliced to its full captured request range;
    that range can exceed the current non-spec graph view. Unsupported valid
    layouts use the original Torch copy/fill operations before returning.
    """

    def __init__(self):
        self._signature = None
        self._pointers = None
        self._host_pointers = None
        self._buffers = None
        self._recorded_stream = None
        self._pointer_ready_event = None
        self._pointer_ready_streams = set()

    @staticmethod
    def _validate(updates):
        for source, destination, actual, graph, _, reset in updates:
            if source.ndim not in (1, 2) or source.ndim != destination.ndim:
                raise ValueError("GDN graph state source and destination ranks must match")
            if source.dtype != torch.int32 or destination.dtype != torch.int32:
                raise ValueError("GDN graph state indices must use int32")
            if source.device != destination.device:
                raise ValueError("GDN graph state source and destination devices must match")
            width = destination.size(1) if destination.ndim == 2 else 1
            if not 0 <= actual <= graph <= destination.size(0) or actual > source.size(0):
                raise ValueError("GDN graph state request counts exceed their storage")
            if source.ndim == 2 and source.size(1) < width:
                raise ValueError("GDN graph state source has insufficient columns")
            if reset is not None and (
                reset.device != destination.device or reset.dtype != torch.int32 or reset.ndim != 2
            ):
                raise ValueError("GDN graph state reset must be an int32 matrix on the destination device")

    @staticmethod
    def _can_batch(updates):
        source, destination, actual_rows, graph_rows, padding, clear = updates[0]
        width = destination.size(1) if destination.ndim == 2 else 1
        geometry = (
            source.ndim,
            width,
            source.stride(0),
            source.stride(1) if source.ndim == 2 else 1,
            actual_rows,
            graph_rows,
            padding,
            0 if clear is None else clear.size(0),
            0 if clear is None else clear.size(1),
            destination.device,
        )
        write_spans = []
        read_spans = []
        for src, dst, actual, graph, pad, reset in updates:
            candidate = (
                src.ndim,
                dst.size(1) if dst.ndim == 2 else 1,
                src.stride(0),
                src.stride(1) if src.ndim == 2 else 1,
                actual,
                graph,
                pad,
                0 if reset is None else reset.size(0),
                0 if reset is None else reset.size(1),
                dst.device,
            )
            if candidate != geometry or not dst.is_contiguous() or any(stride <= 0 for stride in src.stride()):
                return False
            for output, rows in ((dst, graph), (reset, 0 if reset is None else reset.size(0))):
                if output is None or not rows:
                    continue
                if not output.is_contiguous():
                    return False
                begin = output.data_ptr()
                end = begin + rows * (output.size(1) if output.ndim == 2 else 1) * output.element_size()
                # A vector store may touch a partial 32-byte sector. Distinct
                # programs must not share that sector, including adjacent views.
                write_spans.append(
                    (begin // _STORE_ALIGNMENT_BYTES, (end + _STORE_ALIGNMENT_BYTES - 1) // _STORE_ALIGNMENT_BYTES)
                )
            if actual and width:
                begin = src.data_ptr()
                end = begin + ((actual - 1) * src.stride(0) + (width - 1) * geometry[3] + 1) * src.element_size()
                read_spans.append(
                    (begin // _STORE_ALIGNMENT_BYTES, (end + _STORE_ALIGNMENT_BYTES - 1) // _STORE_ALIGNMENT_BYTES)
                )
        write_spans.sort()
        if any(left[1] > right[0] for left, right in zip(write_spans, write_spans[1:])):
            return False
        # Captured state buffers normally have separate storage. Alias inputs
        # are rare and retain the sequential Torch semantics on the fallback.
        for read_start, read_end in read_spans:
            index = bisect_left(write_spans, (read_end,))
            if index and write_spans[index - 1][1] > read_start:
                return False
        return True

    @staticmethod
    def _apply_reference(updates):
        for src, dst, actual, graph, pad, reset in updates:
            width = dst.size(1) if dst.ndim == 2 else 1
            values = src[:actual, :width] if src.ndim == 2 else src[:actual]
            dst[:actual].copy_(values)
            dst[actual:graph].fill_(pad)
            if reset is not None:
                reset.fill_(PAD_SLOT_ID)

    def apply(self, updates):
        if not updates:
            return
        # Do not leave any group partially updated on malformed input.
        self._validate(updates)
        source, destination, actual_rows, graph_rows, padding, clear = updates[0]
        device = destination.device
        if device.type == "cpu" or not self._can_batch(updates):
            self._apply_reference(updates)
            return
        if torch.npu.is_current_stream_capturing():
            # A first-use ready event recorded only inside a not-yet-replayed
            # graph cannot protect graph-external metadata preparation. Keep
            # capture on the original stable Torch writes, with no new cache.
            self._apply_reference(updates)
            return
        width = destination.size(1) if destination.ndim == 2 else 1
        clear_rows = 0 if clear is None else clear.size(0)
        clear_width = 0 if clear is None else clear.size(1)
        signature = tuple(
            (src.data_ptr(), dst.data_ptr(), 0 if reset is None else reset.data_ptr())
            for src, dst, _, _, _, reset in updates
        )
        stream = torch.npu.current_stream(device)
        if self._signature != signature:
            self._host_pointers = torch.tensor(signature, dtype=torch.int64, pin_memory=True)
            self._pointers = self._host_pointers.to(device, non_blocking=True)
            self._buffers = tuple((src, dst, reset) for src, dst, _, _, _, reset in updates)
            self._signature = signature
            self._recorded_stream = None
            # This H2D copy belongs to the updater, so it also owns readiness
            # when the same pointer table is later consumed on a new stream.
            self._pointer_ready_event = torch.npu.Event()
            self._pointer_ready_event.record(stream)
            self._pointer_ready_streams = {stream.npu_stream}
        if stream.npu_stream not in self._pointer_ready_streams:
            stream.wait_event(self._pointer_ready_event)
            self._pointer_ready_streams.add(stream.npu_stream)
        # Compare numeric stream IDs: this torch_npu version reports both
        # `None == stream` and `None != stream` as False.
        if self._recorded_stream is None or self._recorded_stream.npu_stream != stream.npu_stream:
            # Pointer-cache replacement can release the previous allocations
            # before kernels on this side stream complete. Record their use
            # once per storage/stream change so allocator reuse remains ordered.
            self._pointers.record_stream(stream)
            for src, dst, reset in self._buffers:
                src.record_stream(stream)
                dst.record_stream(stream)
                if reset is not None:
                    reset.record_stream(stream)
            self._recorded_stream = stream
        if graph_rows or clear_rows:
            _graph_state_kernel[(len(updates),)](
                self._pointers,
                actual_rows,
                graph_rows,
                clear_rows,
                WIDTH=width,
                SOURCE_ROW_STRIDE=source.stride(0),
                SOURCE_COL_STRIDE=source.stride(1) if source.ndim == 2 else 1,
                PAD_VALUE=padding,
                CLEAR_WIDTH=clear_width,
                CLEAR_VALUE=PAD_SLOT_ID,
                BLOCK=_BLOCK_SIZE,
                multibuffer=False,
                unit_flag=False,
            )
