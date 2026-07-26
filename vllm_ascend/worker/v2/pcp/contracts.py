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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    CacheInputDistribution,
    PCPGroupCapability,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


class CacheWriteSegmentKind(str, Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class CacheWriteSegment:
    kind: CacheWriteSegmentKind
    start: int
    stop: int
    distribution: CacheInputDistribution
    slot_mapping: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"Invalid cache-write segment [{self.start}, {self.stop}).")
        if self.slot_mapping.shape[-1] != self.stop - self.start:
            raise ValueError(
                "Cache-write slot count does not match the segment length: "
                f"{self.slot_mapping.shape[-1]} != {self.stop - self.start}."
            )
        if self.valid_mask is not None:
            if self.valid_mask.ndim != 1:
                raise ValueError("Cache-write valid_mask must be one-dimensional.")
            if self.valid_mask.shape[0] != self.stop - self.start:
                raise ValueError(
                    "Cache-write mask count does not match the segment "
                    f"length: {self.valid_mask.shape[0]} != "
                    f"{self.stop - self.start}."
                )
            if self.valid_mask.dtype != torch.bool:
                raise TypeError("Cache-write valid_mask must have bool dtype.")

    @property
    def input_range(self) -> slice:
        return slice(self.start, self.stop)

    def effective_slot_mapping(self) -> torch.Tensor:
        if self.valid_mask is None:
            return self.slot_mapping
        return torch.where(
            self.valid_mask,
            self.slot_mapping,
            self.slot_mapping.new_full((), -1),
        )


@dataclass(frozen=True)
class CacheWritePlan:
    group_id: int
    segments: tuple[CacheWriteSegment, ...]

    def __post_init__(self) -> None:
        previous_stop = 0
        for segment in self.segments:
            if segment.start != previous_stop:
                raise ValueError("Cache-write segments must be contiguous and ordered.")
            previous_stop = segment.stop


@dataclass(frozen=True)
class HybridPCPLayout:
    """Read-only token conversion indices for one hybrid PCP step."""

    num_decode_tokens: int
    global_num_tokens: int
    linear_num_tokens: int
    linear_num_tokens_padded: int
    fa_num_tokens: int
    fa_num_tokens_padded: int
    hybrid_linear_ag_restore_idx: torch.Tensor
    hybrid_global_to_fa_idx: torch.Tensor
    hybrid_fa_to_linear_idx: torch.Tensor
    linear_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        decode = self.num_decode_tokens
        if not (
            0 <= decode <= self.global_num_tokens and decode <= self.linear_num_tokens and decode <= self.fa_num_tokens
        ):
            raise ValueError("Invalid decode prefix in HybridPCPLayout.")
        if self.linear_num_tokens > self.linear_num_tokens_padded:
            raise ValueError("Linear real-token count exceeds its padded count.")
        if self.fa_num_tokens > self.fa_num_tokens_padded:
            raise ValueError("FA real-token count exceeds its padded count.")

        expected_shapes = (
            (
                self.hybrid_linear_ag_restore_idx,
                self.global_num_tokens - decode,
                "hybrid_linear_ag_restore_idx",
            ),
            (
                self.hybrid_global_to_fa_idx,
                self.fa_num_tokens_padded - decode,
                "hybrid_global_to_fa_idx",
            ),
            (
                self.hybrid_fa_to_linear_idx,
                self.linear_num_tokens_padded - decode,
                "hybrid_fa_to_linear_idx",
            ),
            (
                self.linear_valid_mask,
                self.linear_num_tokens_padded,
                "linear_valid_mask",
            ),
        )
        for tensor, expected, name in expected_shapes:
            if tensor.ndim != 1 or tensor.shape[0] != expected:
                raise ValueError(f"{name} must have shape ({expected},), got {tuple(tensor.shape)}.")

    @property
    def has_prefill(self) -> bool:
        return self.global_num_tokens > self.num_decode_tokens


@dataclass(frozen=True)
class HybridPCPBridgeView:
    layout: HybridPCPLayout


@dataclass(frozen=True)
class HybridPCPForwardView:
    linear_num_tokens: int
    linear_num_tokens_padded: int
    linear_valid_mask: torch.Tensor


@dataclass(frozen=True)
class PreparedGroupInputs:
    group_id: int
    capability: PCPGroupCapability
    input_batch: InputBatch
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    cache_write_plan: CacheWritePlan | None


@dataclass(frozen=True)
class HybridPreparedStep:
    step_id: int
    linear_batch: InputBatch
    fa_batch: InputBatch
    layout: HybridPCPLayout
    group_inputs: tuple[PreparedGroupInputs, ...]
    legacy_block_tables: tuple[torch.Tensor, ...]
    legacy_slot_mappings: torch.Tensor

    def group(self, group_id: int) -> PreparedGroupInputs:
        for group_inputs in self.group_inputs:
            if group_inputs.group_id == group_id:
                return group_inputs
        raise KeyError(f"Unknown KV cache group id {group_id}.")

    @property
    def bridge_view(self) -> HybridPCPBridgeView:
        return HybridPCPBridgeView(self.layout)

    @property
    def forward_view(self) -> HybridPCPForwardView:
        return HybridPCPForwardView(
            linear_num_tokens=self.layout.linear_num_tokens,
            linear_num_tokens_padded=self.layout.linear_num_tokens_padded,
            linear_valid_mask=self.layout.linear_valid_mask,
        )


@runtime_checkable
class HybridPCPManagerProtocol(Protocol):
    """Runner-facing lifecycle exposed only by Hybrid PCP managers."""

    def prepare_dummy_step(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> None: ...

    def consume_prepared_step(self) -> HybridPreparedStep: ...

    def restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor: ...


@runtime_checkable
class HybridPCPModelStateProtocol(Protocol):
    """Minimal model-state interface required by Hybrid PCP."""

    def bind_hybrid_prepared_step(
        self,
        prepared_step: HybridPreparedStep,
    ) -> None: ...


class HybridAttentionMetadataMap(dict[str, Any]):
    """Layer metadata plus immutable hybrid views for forward consumers."""

    def __init__(
        self,
        *args: Any,
        bridge_view: HybridPCPBridgeView,
        forward_view: HybridPCPForwardView,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hybrid_pcp_bridge_view = bridge_view
        self.hybrid_pcp_forward_view = forward_view
