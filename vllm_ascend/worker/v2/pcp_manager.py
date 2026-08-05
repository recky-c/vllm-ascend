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

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
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
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config
        model_config = getattr(vllm_config, "model_config", None)
        self._is_hybrid = bool(getattr(model_config, "is_hybrid", False))
        self._pcp_segment_ids_by_rank: list[list[int]] | None = None
        self._pcp_segment_capacity = 0
        self._pcp_segment_ids_buffer = (
            torch.empty(2 * max_num_reqs, dtype=torch.int64, device=self.device)
            if max_num_reqs is not None
            else None
        )

    def _build_batch_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[list[list[RankSegment]], list[int]]:
        """Build upstream DualChunk rows plus stable ids for recurrent layers.

        Canonical ids are ``request_index * (2 * P) + chunk_index``. They are
        independent of the backend row reorder and therefore let GDN restore
        the causal order ``head ranks ascending, tail ranks descending``.
        """
        if self._is_hybrid:
            continued_prefills = np.flatnonzero(
                is_prefilling & (num_computed_tokens > 0)
            )
            if continued_prefills.size:
                raise NotImplementedError(
                    "Ascend MRV2 Hybrid PCP DualChunk does not support "
                    "continued prefill yet; requests "
                    f"{continued_prefills.tolist()} have computed lengths "
                    f"{num_computed_tokens[continued_prefills].tolist()}."
                )
            short_prefills = np.flatnonzero(
                is_prefilling & (num_scheduled_tokens < self.pcp_world_size)
            )
            if short_prefills.size:
                raise NotImplementedError(
                    "Ascend MRV2 Hybrid PCP DualChunk requires every prefill "
                    "query to contain at least one token per PCP rank; "
                    f"requests {short_prefills.tolist()} have lengths "
                    f"{num_scheduled_tokens[short_prefills].tolist()} with "
                    f"pcp_size={self.pcp_world_size}."
                )

        segments_by_rank, per_rank_num_tokens = super()._build_batch_layout(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )
        num_chunks = 2 * self.pcp_world_size
        ids_by_rank: list[list[int]] = []
        for segments in segments_by_rank:
            rank_ids: list[int] = []
            for segment in segments:
                req_idx = segment.global_batch_req_idx
                if not bool(is_prefilling[req_idx]):
                    rank_ids.append(-1)
                    continue
                query_len = int(num_scheduled_tokens[req_idx])
                chunk_size = (query_len + num_chunks - 1) // num_chunks
                global_start = int(query_start_loc_np[req_idx])
                chunk_idx = (segment.global_batch_slice.start - global_start) // chunk_size
                rank_ids.append(req_idx * num_chunks + chunk_idx)
            ids_by_rank.append(rank_ids)
        self._pcp_segment_ids_by_rank = ids_by_rank
        self._pcp_segment_capacity = 2 * len(num_scheduled_tokens)
        return segments_by_rank, per_rank_num_tokens

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
        """Partition the batch and update Ascend-specific local metadata."""
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
        assert self._pcp_segment_ids_by_rank is not None
        local_segment_ids_np = np.asarray(
            self._pcp_segment_ids_by_rank[self.pcp_rank], dtype=np.int64
        )
        if self._pcp_segment_ids_buffer is None:
            local_batch.pcp_segment_ids = async_copy_to_gpu(
                local_segment_ids_np, device=self.device
            )
        else:
            local_batch.pcp_segment_ids = async_copy_to_gpu(
                local_segment_ids_np,
                out=self._pcp_segment_ids_buffer[: local_segment_ids_np.size],
            )
        local_batch.pcp_segment_capacity = self._pcp_segment_capacity
        return local_batch


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
