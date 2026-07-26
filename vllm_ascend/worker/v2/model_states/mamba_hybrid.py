# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
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

from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_hybrid_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import (
    AscendAttentionMetadataMixin,
)
from vllm_ascend.worker.v2.pcp.contracts import HybridPreparedStep


class AscendMambaHybridModelState(
    AscendAttentionMetadataMixin,
    MambaHybridModelState,
):
    """Ascend adapter for the upstream hybrid model-state lifecycle."""

    def bind_hybrid_prepared_step(
        self,
        prepared_step: HybridPreparedStep,
    ) -> None:
        if getattr(self, "_hybrid_prepared_step", None) is not None:
            raise RuntimeError("A HybridPreparedStep is already bound to the model state.")
        last_step_id = getattr(self, "_last_hybrid_step_id", 0)
        if prepared_step.step_id <= last_step_id:
            raise RuntimeError(
                f"HybridPreparedStep must be newer than the last bound step: {prepared_step.step_id} <= {last_step_id}."
            )
        self._last_hybrid_step_id = prepared_step.step_id
        self._hybrid_prepared_step = prepared_step

    def preprocess_state(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        try:
            super().preprocess_state(
                input_batch,
                block_tables,
                kv_cache_config,
                num_computed_tokens,
            )
        except Exception:
            # execute_model invokes preprocess_state after the Runner has bound
            # the step but before prepare_attn can consume it.
            self._hybrid_prepared_step = None
            raise

    def _build_model_specific_metadata(
        self,
        input_batch: AscendInputBatch,
        num_reqs: int,
        for_capture: bool,
    ) -> MambaHybridAttnMetadata:
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(input_batch.is_prefilling_np)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[input_batch.idx_mapping]

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        return MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        prepared_step = getattr(self, "_hybrid_prepared_step", None)
        if prepared_step is not None:
            if prepared_step.linear_batch is not input_batch:
                self._hybrid_prepared_step = None
                raise RuntimeError("HybridPreparedStep is bound to a different input batch.")
            try:
                self.attn_metadata = build_hybrid_attn_metadata(
                    prepared_step=prepared_step,
                    cudagraph_mode=cudagraph_mode,
                    attn_groups=attn_groups,
                    max_model_len=self.max_model_len,
                    model_specific_metadata_factory=(self._build_model_specific_metadata),
                    for_cudagraph_capture=for_capture,
                )
                return self.attn_metadata
            finally:
                self._hybrid_prepared_step = None

        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
        else:
            num_reqs = input_batch.num_reqs
        return self._prepare_ascend_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            self._build_model_specific_metadata(
                input_batch,
                num_reqs,
                for_capture,
            ),
        )
