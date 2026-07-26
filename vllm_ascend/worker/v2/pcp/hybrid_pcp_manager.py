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
from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    PCPGroupCapability,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.pcp.batch import (
    HybridLinearBatchPartitioner,
    build_fa_prefill_maps,
)
from vllm_ascend.worker.v2.pcp.contracts import (
    HybridPreparedStep,
    PreparedGroupInputs,
)
from vllm_ascend.worker.v2.pcp.group_preparer import (
    DummyGroupPreparationContext,
    PCPGroupInputPreparerRegistry,
    RealGroupPreparationContext,
)
from vllm_ascend.worker.v2.pcp.layout import (
    build_hybrid_pcp_layout,
    build_linear_prefill_lengths,
    count_decode_prefix_tokens,
)
from vllm_ascend.worker.v2.pcp.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.pcp.validation import validate_hybrid_batch


class AscendHybridPCPManager(AscendPCPManager):
    """Hybrid PCP manager with a linear main view and a DualChunk FA view."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        group_capabilities: tuple[PCPGroupCapability, ...],
        group_input_preparers: PCPGroupInputPreparerRegistry,
        max_num_reqs: int,
        max_num_tokens: int,
        block_tables: BlockTables,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            device=device,
            **kwargs,
        )
        self.group_capabilities = group_capabilities
        self._group_input_preparers = group_input_preparers

        self._linear_batch_partitioner = HybridLinearBatchPartitioner(
            vllm_config=vllm_config,
            pcp_world_size=self.pcp_world_size,
            pcp_rank=self.pcp_rank,
            device=device,
            req_states=self._req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
        )
        self._linear_block_tables = tuple(
            table.new_zeros((max_num_reqs, table.shape[1]))
            for table in block_tables.input_block_tables
        )
        self._linear_block_table_ptrs = torch.tensor(
            [table.data_ptr() for table in self._linear_block_tables],
            dtype=torch.uint64,
            device=device,
        )
        num_groups = block_tables.num_kv_cache_groups
        self._linear_slot_mappings = torch.empty(
            num_groups,
            max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._legacy_slot_mappings = torch.empty(
            num_groups,
            max_num_tokens * self.pcp_world_size,
            dtype=torch.int64,
            device=device,
        )
        self._fa_batch: AscendInputBatch | None = None
        self._linear_batch: AscendInputBatch | None = None
        self._hybrid_layout = None
        self._prepared_step: HybridPreparedStep | None = None
        self._step_id = 0

    def partition_batch(
        self,
        input_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        if self._prepared_step is not None:
            raise RuntimeError(
                "The previous HybridPreparedStep was not consumed."
            )
        validate_hybrid_batch(input_batch)
        fa_batch = super().partition_batch(input_batch)
        lengths_by_rank = build_linear_prefill_lengths(
            input_batch.num_scheduled_tokens,
            input_batch.is_prefilling_np,
            pcp_world_size=self.pcp_world_size,
            cp_interleave=self.cp_interleave,
        )
        linear_batch, linear_prefill_maps = self._linear_batch_partitioner.partition(
            input_batch,
            lengths_by_rank,
        )
        decode_tokens = count_decode_prefix_tokens(
            input_batch.num_scheduled_tokens,
            input_batch.is_prefilling_np,
        )
        fa_prefill_maps = build_fa_prefill_maps(
            global_batch=input_batch,
            decode_tokens=decode_tokens,
            fa_tokens_padded=fa_batch.num_tokens_after_padding,
            pcp_world_size=self.pcp_world_size,
            get_rank_segments=self._get_rank_segments,
        )
        self._hybrid_layout = build_hybrid_pcp_layout(
            linear_prefill_by_rank=linear_prefill_maps,
            fa_prefill_by_rank=fa_prefill_maps,
            pcp_rank=self.pcp_rank,
            num_decode_tokens=decode_tokens,
            global_num_tokens=input_batch.num_tokens,
            linear_num_tokens=linear_batch.num_tokens,
            fa_num_tokens=fa_batch.num_tokens,
            device=self.device,
        )
        self._fa_batch = fa_batch
        self._linear_batch = linear_batch
        return linear_batch

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if input_batch is not self._linear_batch:
            raise RuntimeError("Hybrid PCP prepare_attn must consume the linear main batch.")
        assert self._fa_batch is not None
        assert self._hybrid_layout is not None
        assert self._block_tables is not None

        fa_block_tables, fa_slot_mappings = super().prepare_attn(self._fa_batch)
        linear_block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            input_batch.num_reqs_after_padding,
            out=self._linear_block_tables,
            out_ptrs=self._linear_block_table_ptrs,
        )
        linear_slot_mappings = self._block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.num_tokens_after_padding,
            out=self._linear_slot_mappings,
        )

        self._legacy_slot_mappings.fill_(PAD_SLOT_ID)
        group_inputs: list[PreparedGroupInputs] = []
        legacy_block_tables: list[torch.Tensor] = []
        global_tokens = self._hybrid_layout.global_num_tokens
        assert self._global_batch_slot_mappings is not None
        global_slots = self._global_batch_slot_mappings[:, :global_tokens]
        for group_id, capability in enumerate(self.group_capabilities):
            preparer = self._group_input_preparers.resolve(
                capability.input_layout
            )
            selection = preparer.prepare_real(
                RealGroupPreparationContext(
                    group_id=group_id,
                    capability=capability,
                    linear_batch=input_batch,
                    fa_batch=self._fa_batch,
                    layout=self._hybrid_layout,
                    linear_block_table=linear_block_tables[group_id],
                    fa_block_table=fa_block_tables[group_id],
                    linear_slot_mapping=linear_slot_mappings[group_id],
                    fa_slot_mapping=fa_slot_mappings[group_id],
                    global_slot_mapping=global_slots[group_id],
                )
            )

            legacy_row = self._legacy_slot_mappings[group_id]
            legacy_row[: selection.slot_mapping.shape[0]].copy_(
                selection.slot_mapping
            )
            legacy_block_tables.append(selection.block_table)
            group_inputs.append(
                PreparedGroupInputs(
                    group_id=group_id,
                    capability=capability,
                    input_batch=selection.input_batch,
                    block_table=selection.block_table,
                    slot_mapping=legacy_row[: selection.slot_mapping.shape[0]],
                    cache_write_plan=selection.cache_write_plan,
                )
            )

        max_group_slots = max(
            item.slot_mapping.shape[0] for item in group_inputs
        )
        legacy_slot_mappings = self._legacy_slot_mappings[
            :, :max_group_slots
        ]
        self._step_id += 1
        self._prepared_step = HybridPreparedStep(
            step_id=self._step_id,
            linear_batch=input_batch,
            fa_batch=self._fa_batch,
            layout=self._hybrid_layout,
            group_inputs=tuple(group_inputs),
            legacy_block_tables=tuple(legacy_block_tables),
            legacy_slot_mappings=legacy_slot_mappings,
        )
        return tuple(legacy_block_tables), legacy_slot_mappings

    def prepare_dummy_step(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> None:
        """Create the normal immutable step contract for a dummy forward."""
        if self._prepared_step is not None:
            raise RuntimeError(
                "The previous HybridPreparedStep was not consumed."
            )
        num_tokens = input_batch.num_tokens
        empty_prefill_maps = tuple(
            np.empty(0, dtype=np.int64)
            for _ in range(self.pcp_world_size)
        )
        layout = build_hybrid_pcp_layout(
            linear_prefill_by_rank=empty_prefill_maps,
            fa_prefill_by_rank=empty_prefill_maps,
            pcp_rank=self.pcp_rank,
            num_decode_tokens=num_tokens,
            global_num_tokens=num_tokens,
            linear_num_tokens=num_tokens,
            fa_num_tokens=num_tokens,
            device=self.device,
        )

        group_inputs: list[PreparedGroupInputs] = []
        for group_id, capability in enumerate(self.group_capabilities):
            group_slots = slot_mappings[group_id, :num_tokens]
            preparer = self._group_input_preparers.resolve(
                capability.input_layout
            )
            selection = preparer.prepare_dummy(
                DummyGroupPreparationContext(
                    group_id=group_id,
                    capability=capability,
                    input_batch=input_batch,
                    block_table=block_tables[group_id],
                    slot_mapping=group_slots,
                    device=self.device,
                )
            )
            group_inputs.append(
                PreparedGroupInputs(
                    group_id=group_id,
                    capability=capability,
                    input_batch=selection.input_batch,
                    block_table=selection.block_table,
                    slot_mapping=selection.slot_mapping,
                    cache_write_plan=selection.cache_write_plan,
                )
            )

        self._fa_batch = input_batch
        self._linear_batch = input_batch
        self._hybrid_layout = layout
        self._step_id += 1
        self._prepared_step = HybridPreparedStep(
            step_id=self._step_id,
            linear_batch=input_batch,
            fa_batch=input_batch,
            layout=layout,
            group_inputs=tuple(group_inputs),
            legacy_block_tables=block_tables,
            legacy_slot_mappings=slot_mappings,
        )

    def consume_prepared_step(self) -> HybridPreparedStep:
        if self._prepared_step is None:
            raise RuntimeError("HybridPreparedStep has not been prepared.")
        step = self._prepared_step
        self._prepared_step = None
        return step

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self._hybrid_layout is not None
        layout = self._hybrid_layout
        decode = layout.num_decode_tokens
        if not layout.has_prefill:
            return hidden_states[:decode]
        gathered_prefill = get_pcp_group().all_gather(
            hidden_states[decode : layout.linear_num_tokens_padded].contiguous(),
            dim=0,
        )
        global_prefill = gathered_prefill[layout.hybrid_linear_ag_restore_idx]
        if decode:
            return torch.cat(
                (hidden_states[:decode], global_prefill),
                dim=0,
            )
        return global_prefill
