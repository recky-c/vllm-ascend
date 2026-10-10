# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
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
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper, dflash_draft_kv_optimistic_bound
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)


def prepare_dflash_inputs_factory(kv_cache_block_size: int) -> Callable[..., None]:
    # Upstream uses the attention kernel block size for DCP ownership, which is
    # incorrect when physical KV blocks are larger than kernel blocks. Bind the
    # physical size so ownership uses KV cache blocks while slot lookup uses the
    # kernel-sized block table supplied by the upstream caller.
    def prepare_with_block_size(*args: Any, **kwargs: Any) -> None:
        prepare_dflash_inputs(*args, **kwargs, kv_cache_block_size=kv_cache_block_size)

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
        use_cpu_bound = (
            self._can_use_dflash_draft_cpu_bound(num_reqs, seq_lens_cpu_upper_bound, step)
            and step == self.num_query_per_req
            and num_query_per_req == self.num_query_per_req
            and dcp_local_seq_lens is None
        )
        # Scope only the draft query build, including the fresh FULL graph
        # fallback. Target-prefill reuse and capture never enter this scope.
        with build_attn_metadata_wrapper(), dflash_draft_kv_optimistic_bound(use_cpu_bound):
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
            # Upstream propose already builds current draft metadata before
            # replay. Hand that same build to Ascend's graph parameter update.
            self._draft_attn_metadata_for_graph = (batch_desc, attn_metadata)
        return attn_metadata

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
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
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
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

        if getattr(self, "_dflash_draft_kv_optimistic_bound_enabled", False):
            self._dflash_draft_kv_optimistic_bound_enabled = all(
                type(group.get_metadata_builder(0)) is AscendAttentionMetadataBuilder
                for group_id in self.draft_kv_cache_group_ids
                for group in self.attn_groups[group_id]
            )

        self.attn_backends = attn_backends
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
            with build_attn_metadata_wrapper():
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
