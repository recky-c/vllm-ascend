# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.hybrid_pcp import (
    init_linear_pcp,
    partition_sequential_batch,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


def validate_ascend_pcp_config(
    vllm_config: VllmConfig,
    supports_mm_inputs: bool,
) -> None:
    """Validate the PCP subset implemented by the Ascend MRV2 runner."""
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    if parallel_config.prefill_context_parallel_size <= 1:
        return

    PCPManager.validate_config(vllm_config, supports_mm_inputs)
    if model_config.use_mla:
        return

    if parallel_config.decode_context_parallel_size > 1:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support PCP and DCP simultaneously yet.")
    if vllm_config.cache_config.enable_prefix_caching:
        raise NotImplementedError("Ascend MRV2 GQA PCP does not support prefix caching yet. Disable prefix caching.")
    if vllm_config.scheduler_config.enable_chunked_prefill:
        raise NotImplementedError(
            "Ascend MRV2 GQA PCP does not support scheduler chunked prefill yet. Disable chunked prefill."
        )

    text_config = model_config.hf_text_config
    num_heads = getattr(text_config, "num_attention_heads", None)
    num_kv_heads = getattr(text_config, "num_key_value_heads", None)
    if (
        not isinstance(num_heads, int)
        or not isinstance(num_kv_heads, int)
        or num_kv_heads <= 0
        or num_heads <= num_kv_heads
        or num_heads % num_kv_heads != 0
    ):
        raise NotImplementedError(
            "Ascend MRV2 GQA PCP requires num_attention_heads to be an "
            "integer multiple greater than num_key_value_heads."
        )


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    def __init__(self, *args, vllm_config: VllmConfig, **kwargs) -> None:
        max_num_reqs = kwargs.get("max_num_reqs")
        max_num_tokens = kwargs.get("max_num_tokens")
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config
        model_config = getattr(vllm_config, "model_config", None)
        self._is_hybrid = bool(getattr(model_config, "is_hybrid", False))
        self._linear_batch_partitioner = init_linear_pcp(
            is_hybrid=self._is_hybrid,
            vllm_config=vllm_config,
            pcp_world_size=self.pcp_world_size,
            pcp_rank=self.pcp_rank,
            device=self.device,
            req_states=self._req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
        )

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Override upstream validate_config: drop the MLA-only restriction."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        pcp_size = parallel_config.prefill_context_parallel_size
        if pcp_size <= 1:
            return

        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError(
                "MRV2 PCP does not support encoder-decoder models yet."
            )
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "MRV2 PCP does not support speculative decoding yet."
            )
        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        if (
            is_sparse_mla
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise NotImplementedError(
                "MRV2 sparse MLA PCP does not support CUDA graphs yet. "
                "Set -cc.cudagraph_mode=NONE."
            )
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise NotImplementedError("MRV2 PCP supports PIECEWISE CUDA graphs only.")

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """DualChunk partition + Ascend metadata (non-hybrid / FA-only path)."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens,
        )
        return local_batch

    def partition_linear_batch(
        self, input_batch: AscendInputBatch
    ) -> tuple[AscendInputBatch, int, list[list[RankSegment]]]:
        """Delegate sequential local-batch materialization."""
        if self._linear_batch_partitioner is None:
            raise RuntimeError(
                "partition_linear_batch requires a hybrid PCP manager."
            )
        return self._linear_batch_partitioner.partition(input_batch)

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Local block tables / slots for any PCP partition (DualChunk or sequential).

        Slot = f(req, position) on the local batch only. Does not DualChunk-expand
        slots; KV gather all-gathers local K/V/slots with the same row layout.
        """
        assert self._block_tables is not None
        assert self._local_block_tables is not None
        assert self._local_block_table_ptrs is not None
        assert self._global_batch_slot_mappings is not None

        block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            input_batch.num_reqs_after_padding,
            out=self._local_block_tables,
            out_ptrs=self._local_block_table_ptrs,
        )
        slot_mappings = self._block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.num_tokens_after_padding,
            out=self._global_batch_slot_mappings,
        )
        return block_tables, slot_mappings

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        """Dummy local-width slots for profile/capture (matches prepare_attn)."""
        assert self._global_batch_slot_mappings is not None
        self._global_batch_slot_mappings.fill_(PAD_SLOT_ID)
        return self._global_batch_slot_mappings[:, :num_tokens]


def maybe_partition_ascend_pcp_batch(
    manager: AscendPCPManager | None,
    input_batch: AscendInputBatch,
) -> AscendInputBatch:
    """Partition for Ascend PCP; hybrid uses single-view sequential split."""
    if manager is None:
        return input_batch
    if manager._is_hybrid:
        return partition_sequential_batch(manager, input_batch)
    return manager.partition_batch(input_batch)


def maybe_build_ascend_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
) -> AscendPCPManager | None:
    """Build the Ascend PCP manager after validating the supported subset."""
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    validate_ascend_pcp_config(vllm_config, supports_mm_inputs)
    dcp_size = parallel_config.decode_context_parallel_size
    return AscendPCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=get_pcp_group().rank_in_group,
        device=device,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=get_dcp_group().rank_in_group if dcp_size > 1 else 0,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
        vllm_config=vllm_config,
    )
