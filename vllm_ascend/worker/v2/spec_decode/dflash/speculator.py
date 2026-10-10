# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder, AscendMetadata
from vllm_ascend.attention.metadata_reuse import tensor_view_key
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    dflash_draft_kv_optimistic_bound,
    dflash_draft_seq_lens_cpu,
)
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)

# Callback is execution-local: the global upstream dispatch function must not
# capture one speculator and send another worker's lengths to its buffer.
_DFLASH_INPUTS_PREPARED: ContextVar[Callable[[], None] | None] = ContextVar("dflash_inputs_prepared", default=None)


@contextmanager
def dflash_inputs_prepared(callback: Callable[[], None]):
    token = _DFLASH_INPUTS_PREPARED.set(callback)
    try:
        yield
    finally:
        _DFLASH_INPUTS_PREPARED.reset(token)


def prepare_dflash_inputs_factory(kv_cache_block_size: int) -> Callable[..., None]:
    # Upstream uses the attention kernel block size for DCP ownership, which is
    # incorrect when physical KV blocks are larger than kernel blocks. Bind the
    # physical size so ownership uses KV cache blocks while slot lookup uses the
    # kernel-sized block table supplied by the upstream caller.
    def prepare_with_block_size(*args: Any, **kwargs: Any) -> None:
        prepare_dflash_inputs(*args, **kwargs, kv_cache_block_size=kv_cache_block_size)
        callback = _DFLASH_INPUTS_PREPARED.get()
        if callback is not None:
            callback()

    return prepare_with_block_size


def _supports_dflash_draft_kv_optimistic_bound(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    parallel_config = vllm_config.parallel_config
    return (
        speculative_config is not None
        and speculative_config.method == "dflash"
        and vllm_config.model_config.hf_text_config.model_type in ("qwen3_5", "qwen3_5_text")
        and not speculative_config.enable_adaptive_verification
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
    )


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
    def _can_use_dflash_draft_cpu_bound(self, num_reqs: int, upper_bound: torch.Tensor, step: int) -> bool:
        """Require an upstream CPU bound covered by allocated kernel blocks.

        DFlash reserves K+1 lookahead tokens. The upper bound adds the same
        query width; rejection can only shorten the exact device length.
        Check the host allocation ledger rather than preallocated table width.
        """
        if (
            not getattr(self, "_dflash_draft_kv_optimistic_bound_active", False)
            or not getattr(self, "_dflash_draft_kv_zeroing_ready", False)
            or step != self.num_query_per_req
            or num_reqs <= 0
            or upper_bound is None
            or upper_bound.device.type != "cpu"
            or upper_bound.ndim != 1
            or upper_bound.dtype not in (torch.int32, torch.int64)
            or upper_bound.numel() < num_reqs
        ):
            return False
        block_tables = getattr(self, "block_tables", None)
        num_blocks = getattr(getattr(block_tables, "num_blocks", None), "np", None)
        idx_mapping = getattr(self.input_batch, "idx_mapping_np", None)
        group_ids = getattr(self, "draft_kv_cache_group_ids", None)
        kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
        if (
            num_blocks is None
            or idx_mapping is None
            or group_ids is None
            or not group_ids
            or kernel_block_sizes is None
            or len(idx_mapping) < num_reqs
            or num_blocks.ndim != 2
        ):
            return False
        bounds = upper_bound[:num_reqs].to(torch.int64).numpy()
        for req in range(num_reqs):
            state_idx = int(idx_mapping[req])
            if state_idx < 0 or state_idx >= num_blocks.shape[1] or bounds[req] < 0:
                return False
            draft_bound = min(int(bounds[req]) + step, self.max_model_len)
            for group_id in group_ids:
                if group_id < 0 or group_id >= num_blocks.shape[0] or group_id >= len(kernel_block_sizes):
                    return False
                block_size = kernel_block_sizes[group_id]
                if block_size <= 0 or draft_bound > int(num_blocks[group_id, state_idx]) * block_size:
                    return False
        return True

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            return super().load_draft_model(target_model, target_attn_layer_names)

    def _on_draft_inputs_prepared(self) -> None:
        if not getattr(self, "_exact_draft_metadata_active", False):
            return
        source = self.input_buffers.seq_lens
        if (
            source.device.type != "npu"
            or source.ndim != 1
            or not source.is_contiguous()
            or source.dtype not in (torch.int32, torch.int64)
            or self.input_batch.num_reqs > source.numel()
        ):
            return
        self._draft_input_groups_prepared += 1
        # Each global draft group writes the shared length buffer. Snapshot
        # after its last writer, before context KV work is submitted.
        if self._draft_input_groups_prepared == len(self.draft_kv_cache_group_ids):
            self._start_draft_seq_lens_copy(self.input_batch.num_reqs)

    def _start_draft_seq_lens_copy(self, num_reqs: int) -> None:
        source = self.input_buffers.seq_lens
        key = tensor_view_key(source)
        if getattr(self, "_draft_seq_lens_source_key", None) != key:
            self._finish_draft_seq_lens_copy()
            self._draft_seq_lens_cpu = torch.empty(source.numel(), dtype=source.dtype, device="cpu", pin_memory=True)
            self._draft_seq_lens_copy_stream = torch.npu.Stream(device=source.device)
            self._draft_seq_lens_copy_event = torch.npu.Event()
            self._draft_seq_lens_source_key = key
        stream = self._draft_seq_lens_copy_stream
        event = self._draft_seq_lens_copy_event
        assert stream is not None and event is not None
        assert self._draft_seq_lens_cpu is not None
        current = torch.npu.current_stream()
        with torch.npu.stream(stream):
            stream.wait_stream(current)
            self._draft_seq_lens_cpu[:num_reqs].copy_(source[:num_reqs], non_blocking=True)
            event.record(stream)
        self._draft_seq_lens_copy_count = num_reqs

    def _finish_draft_seq_lens_copy(self) -> None:
        if getattr(self, "_draft_seq_lens_copy_count", None) is not None:
            assert self._draft_seq_lens_copy_event is not None
            # Wait only for the copy stream, including exception cleanup. The
            # compute stream may already be waiting on FIA update events.
            self._draft_seq_lens_copy_event.synchronize()
            self._draft_seq_lens_copy_count = None

    def _prepare_draft_seq_lens_cpu(self, num_reqs: int, num_reqs_padded: int) -> torch.Tensor:
        if getattr(self, "_draft_seq_lens_copy_count", None) != num_reqs:
            self._start_draft_seq_lens_copy(num_reqs)
        self._finish_draft_seq_lens_copy()
        assert self._draft_seq_lens_cpu is not None
        if num_reqs_padded > self._draft_seq_lens_cpu.numel():
            raise ValueError("Draft graph request count exceeds the persistent sequence-length buffer")
        mirror = self._draft_seq_lens_cpu[:num_reqs_padded]
        mirror[num_reqs:].zero_()
        return mirror

    def _can_use_exact_draft_metadata(self, batch_desc, num_query_per_req, step, dcp_local_seq_lens) -> bool:
        source = self.input_buffers.seq_lens if getattr(self, "_exact_draft_metadata_active", False) else None
        return (
            source is not None
            and source.device.type == "npu"
            and source.ndim == 1
            and source.is_contiguous()
            and source.dtype in (torch.int32, torch.int64)
            and source.numel() >= (batch_desc.num_reqs or self.input_batch.num_reqs)
            and getattr(self, "_exact_draft_metadata_active", False)
            and getattr(self, "_use_cpu_seq_lens", False)
            and num_query_per_req == self.num_query_per_req
            and step == self.num_query_per_req
            and dcp_local_seq_lens is None
            and batch_desc.cg_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
        )

    def _can_defer_draft_metadata(self, batch_desc) -> bool:
        if (
            not getattr(self, "_deferred_metadata_enabled", False)
            or not getattr(self, "_exact_draft_metadata_active", False)
            or not getattr(self, "_reuse_draft_layout", False)
            or getattr(self, "_materializing_draft_metadata", False)
            or getattr(self, "_draft_seq_lens_copy_count", None) is None
            or batch_desc.cg_mode != CUDAGraphMode.FULL
        ):
            return False
        graph = self.query_cudagraph_manager.graphs.get(batch_desc)
        return (
            isinstance(graph, UpdatableGraph)
            and bool(graph.tasks)
            and getattr(graph, "_dflash_neutral_tasks", None) is not None
        )

    def _draft_layout_key(self, batch_desc, num_reqs, num_query_per_req, step, causal):
        # Static builder configuration is covered by its identity and KV spec;
        # group IDs are global, including empty target-only group entries.
        groups_key = tuple(
            (
                gid,
                tuple(group.layer_names),
                id(group.get_metadata_builder(0)),
                repr(group.get_metadata_builder(0).kv_cache_spec),
            )
            for gid, groups in enumerate(self.attn_groups)
            for group in groups
        )
        return (
            num_reqs,
            batch_desc.num_reqs,
            batch_desc.num_tokens,
            num_query_per_req,
            step,
            causal if isinstance(causal, bool) else tuple(sorted(causal.items())),
            self.max_model_len,
            groups_key,
            tensor_view_key(self.input_buffers.seq_lens),
            tensor_view_key(self.input_buffers.query_start_loc),
            tuple(tensor_view_key(self.block_tables.input_block_tables[gid]) for gid in self.draft_kv_cache_group_ids),
            tensor_view_key(self.block_tables.slot_mappings),
        )

    def _refresh_draft_layout(self, template, seq_lens_cpu, num_reqs, num_reqs_padded, num_tokens):
        lengths = seq_lens_cpu[:num_reqs].tolist() + [1] * (num_reqs_padded - num_reqs)
        refreshed = {}
        group_metadata = {}
        for group_index, groups in enumerate(self.attn_groups):
            for group in groups:
                for name in group.layer_names:
                    previous = template[name]
                    key = (group_index, id(previous))
                    if key not in group_metadata:
                        group_metadata[key] = replace(
                            previous,
                            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
                            seq_lens_cpu=seq_lens_cpu,
                            seq_lens_list=lengths,
                            seq_lens_gpu=self.input_buffers.seq_lens[:num_reqs_padded],
                            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
                            block_tables=AscendAttentionMetadataBuilder._pad_block_table(
                                self.block_tables.input_block_tables[group_index][:num_reqs_padded], num_reqs_padded
                            ),
                            slot_mapping=self.block_tables.slot_mappings[group_index][:num_tokens],
                            reshape_cache_event=None,
                            qfa_metadata_cache={},
                        )
                    refreshed[name] = group_metadata[key]
        return refreshed

    def _build_uniform_attn_metadata(
        self,
        batch_desc: BatchExecutionDescriptor,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        use_exact = self._can_use_exact_draft_metadata(batch_desc, num_query_per_req, step, dcp_local_seq_lens)
        layout_key = None
        if use_exact and self._reuse_draft_layout and batch_desc.cg_mode == CUDAGraphMode.FULL:
            layout_key = self._draft_layout_key(batch_desc, num_reqs, num_query_per_req, step, causal)
        cached_layout = getattr(self, "_draft_metadata_template", None)
        if (
            use_exact
            and layout_key is not None
            and cached_layout is not None
            and cached_layout[0] == layout_key
            and self._can_defer_draft_metadata(batch_desc)
        ):
            self._deferred_draft_attn_metadata = dict(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
            return None
        template_key = None
        context = nullcontext()
        if use_exact:
            padded = batch_desc.num_reqs or num_reqs
            mirror = self._prepare_draft_seq_lens_cpu(num_reqs, padded)
            if self._reuse_draft_layout and batch_desc.cg_mode == CUDAGraphMode.FULL:
                template_key = layout_key
                cached = self._draft_metadata_template
                if cached is not None and cached[0] == template_key:
                    metadata = self._refresh_draft_layout(cached[1], mirror, num_reqs, padded, batch_desc.num_tokens)
                    self._draft_attn_metadata_for_graph = (batch_desc, metadata)
                    return metadata
            context = dflash_draft_seq_lens_cpu(mirror)
        use_cpu_bound = (
            not use_exact
            and self._can_use_dflash_draft_cpu_bound(num_reqs, seq_lens_cpu_upper_bound, step)
            and step == self.num_query_per_req
            and num_query_per_req == self.num_query_per_req
            and dcp_local_seq_lens is None
        )
        with build_attn_metadata_wrapper(), dflash_draft_kv_optimistic_bound(use_cpu_bound), context:
            attn_metadata = super()._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if (
            getattr(self, "_reuse_draft_attn_metadata", False)
            and batch_desc.cg_mode == CUDAGraphMode.FULL
            and attn_metadata is not None
        ):
            self._draft_attn_metadata_for_graph = (batch_desc, attn_metadata)
            if template_key is not None and all(type(value) is AscendMetadata for value in attn_metadata.values()):
                self._draft_metadata_template = (template_key, attn_metadata)
        return attn_metadata

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        pending = getattr(self, "_deferred_draft_attn_metadata", None)
        if pending is not None:
            self._deferred_draft_attn_metadata = None
            self._materializing_draft_metadata = True
            try:
                self._build_uniform_attn_metadata(**pending)
            finally:
                self._materializing_draft_metadata = False
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        cached = getattr(self, "_draft_attn_metadata_for_graph", None)
        # Consume the handoff once. Standalone graph updates and a different
        # padded shape must build fresh metadata.
        self._draft_attn_metadata_for_graph = None
        if cached is not None:
            batch_desc, attn_metadata = cached
            if batch_desc.num_reqs == num_reqs_padded and batch_desc.num_tokens == num_tokens_padded:
                self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
                return [attn_metadata]
        with build_attn_metadata_wrapper():
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._draft_attn_metadata_for_graph = None
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        Upstream ``Speculator._build_draft_attn_metadata`` clamps
        ``query_start_loc`` at the real ``num_reqs`` to keep the cumulative
        series non-decreasing, so when a batch is padded to a capture size
        (``num_reqs_padded > num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        key = (num_reqs_padded, self.num_query_per_req)
        cached = getattr(self, "_draft_query_lengths", None)
        if cached is None or cached[0] != key:
            cached = (key, [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)])
            self._draft_query_lengths = cached
        query_lens_list = cached[1]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()
        self._dflash_draft_kv_optimistic_bound_enabled = (
            get_ascend_config().enable_dflash_draft_kv_optimistic_bound
            and _supports_dflash_draft_kv_optimistic_bound(vllm_config)
        )
        self._dflash_draft_kv_optimistic_bound_active = False
        self._dflash_draft_kv_zeroing_ready = False
        ascend_config = get_ascend_config()
        self._exact_metadata_enabled = (
            ascend_config.enable_dflash_exact_metadata_optimizations
            and not ascend_config.enable_dflash_draft_kv_optimistic_bound
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
            and vllm_config.parallel_config.decode_context_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and vllm_config.speculative_config.method == "dflash"
            and not vllm_config.speculative_config.enable_adaptive_verification
        )
        self._deferred_metadata_enabled = ascend_config.enable_dflash_deferred_metadata and self._exact_metadata_enabled
        self._draft_seq_lens_cpu = None
        self._draft_seq_lens_copy_stream = None
        self._draft_seq_lens_copy_event = None
        self._draft_seq_lens_copy_count = None
        self._draft_seq_lens_source_key = None
        self._draft_metadata_template = None
        self._deferred_draft_attn_metadata = None
        self._draft_query_lengths = None
        self._draft_input_groups_prepared = 0
        self._exact_draft_metadata_active = False
        self._materializing_draft_metadata = False
        self._use_cpu_seq_lens = False
        self._reuse_draft_layout = False

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        has_sinks = False
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                layer = attn_layers[layer_name]
                attn_backends[layer_name] = layer.get_attn_backend()
                # Sink/FIA-v2 and specialized backends keep their existing
                # contracts until their graph consumers are separately tested.
                if getattr(getattr(layer, "impl", None), "sinks", None) is not None:
                    self._dflash_draft_kv_optimistic_bound_enabled = False
                    has_sinks = True

        if getattr(self, "_dflash_draft_kv_optimistic_bound_enabled", False):
            self._dflash_draft_kv_optimistic_bound_enabled = all(
                type(group.get_metadata_builder(0)) is AscendAttentionMetadataBuilder
                for group_id in self.draft_kv_cache_group_ids
                for group in self.attn_groups[group_id]
            )

        self.attn_backends = attn_backends
        self._use_cpu_seq_lens = (
            getattr(self, "_exact_metadata_enabled", False)
            and not has_sinks
            and bool(attn_backends)
            and all(backend is AscendAttentionBackend for backend in attn_backends.values())
            and all(
                type(group.get_metadata_builder(0)) is AscendAttentionMetadataBuilder
                for groups in self.attn_groups
                for group in groups
            )
        )
        self._reuse_draft_layout = self._use_cpu_seq_lens and all(
            group.get_metadata_builder(0).supports_update_block_table for groups in self.attn_groups for group in groups
        )
        self._draft_metadata_template = None
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
            self.vllm_config.cache_config.block_size
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        self._draft_attn_metadata_for_graph = None
        self._reuse_draft_attn_metadata = True
        self._deferred_draft_attn_metadata = None
        self._draft_input_groups_prepared = 0
        self._exact_draft_metadata_active = (
            getattr(self, "_use_cpu_seq_lens", False)
            and not dummy_run
            and not is_profile
            and not getattr(input_batch, "has_prefill", True)
        )
        self._dflash_draft_kv_optimistic_bound_active = (
            getattr(self, "_dflash_draft_kv_optimistic_bound_enabled", False)
            and not dummy_run
            and not is_profile
            and not getattr(input_batch, "has_prefill", True)
        )
        sync_state = dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        try:
            with build_attn_metadata_wrapper(), dflash_inputs_prepared(self._on_draft_inputs_prepared):
                return super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    sync_state,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
        finally:
            # Never let metadata from this batch survive into the next proposal,
            # including eager/profiling paths or a failed graph replay.
            self._reuse_draft_attn_metadata = False
            self._dflash_draft_kv_optimistic_bound_active = False
            self._draft_attn_metadata_for_graph = None
            self._exact_draft_metadata_active = False
            self._deferred_draft_attn_metadata = None
            self._materializing_draft_metadata = False
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self._draft_metadata_template = None
            try:
                self._finish_draft_seq_lens_copy()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"DFlash sequence-length copy cleanup also failed: {cleanup_error!r}")


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    *,
    kv_cache_block_size: int,
) -> None:
    prepare_dflash_inputs_triton(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        block_size,
        cp_rank,
        cp_size,
        cp_interleave,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
        kv_cache_block_size=kv_cache_block_size,
    )
