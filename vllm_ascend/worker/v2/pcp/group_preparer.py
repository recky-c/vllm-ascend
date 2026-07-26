# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.attention.context_parallel.hybrid_pcp.contracts import (
    PCPGroupCapability,
    PCPInputLayout,
)
from vllm_ascend.worker.v2.pcp.cache_plan import (
    build_dummy_fa_cache_write_plan,
    build_fa_cache_write_plan,
)
from vllm_ascend.worker.v2.pcp.contracts import (
    CacheWritePlan,
    HybridPCPLayout,
)


@dataclass(frozen=True)
class GroupInputSelection:
    input_batch: InputBatch
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    cache_write_plan: CacheWritePlan | None


@dataclass(frozen=True)
class RealGroupPreparationContext:
    group_id: int
    capability: PCPGroupCapability
    linear_batch: InputBatch
    fa_batch: InputBatch
    layout: HybridPCPLayout
    linear_block_table: torch.Tensor
    fa_block_table: torch.Tensor
    linear_slot_mapping: torch.Tensor
    fa_slot_mapping: torch.Tensor
    global_slot_mapping: torch.Tensor


@dataclass(frozen=True)
class DummyGroupPreparationContext:
    group_id: int
    capability: PCPGroupCapability
    input_batch: InputBatch
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    device: torch.device


class PCPGroupInputPreparer(Protocol):
    """Select one group's input view without exposing it to the Manager."""

    @property
    def input_layout(self) -> PCPInputLayout: ...

    def prepare_real(
        self,
        context: RealGroupPreparationContext,
    ) -> GroupInputSelection: ...

    def prepare_dummy(
        self,
        context: DummyGroupPreparationContext,
    ) -> GroupInputSelection: ...


class DualChunkGroupInputPreparer:
    input_layout = PCPInputLayout.DUAL_CHUNK_VIRTUAL

    def prepare_real(
        self,
        context: RealGroupPreparationContext,
    ) -> GroupInputSelection:
        return GroupInputSelection(
            input_batch=context.fa_batch,
            block_table=context.fa_block_table,
            slot_mapping=context.fa_slot_mapping,
            cache_write_plan=build_fa_cache_write_plan(
                group_id=context.group_id,
                capability=context.capability,
                layout=context.layout,
                global_slot_mapping=context.global_slot_mapping,
            ),
        )

    def prepare_dummy(
        self,
        context: DummyGroupPreparationContext,
    ) -> GroupInputSelection:
        return GroupInputSelection(
            input_batch=context.input_batch,
            block_table=context.block_table,
            slot_mapping=context.slot_mapping,
            cache_write_plan=build_dummy_fa_cache_write_plan(
                group_id=context.group_id,
                slot_mapping=context.slot_mapping,
                device=context.device,
            ),
        )


class ContiguousStateGroupInputPreparer:
    input_layout = PCPInputLayout.CONTIGUOUS_CAUSAL_STATE

    def prepare_real(
        self,
        context: RealGroupPreparationContext,
    ) -> GroupInputSelection:
        return GroupInputSelection(
            input_batch=context.linear_batch,
            block_table=context.linear_block_table,
            slot_mapping=context.linear_slot_mapping,
            cache_write_plan=None,
        )

    def prepare_dummy(
        self,
        context: DummyGroupPreparationContext,
    ) -> GroupInputSelection:
        return GroupInputSelection(
            input_batch=context.input_batch,
            block_table=context.block_table,
            slot_mapping=context.slot_mapping,
            cache_write_plan=None,
        )


class PCPGroupInputPreparerRegistry:
    """Immutable strategy registry keyed by the backend input-layout contract."""

    def __init__(
        self,
        preparers: tuple[PCPGroupInputPreparer, ...],
    ) -> None:
        by_layout: dict[PCPInputLayout, PCPGroupInputPreparer] = {}
        for preparer in preparers:
            if preparer.input_layout in by_layout:
                raise ValueError(f"Duplicate PCP group preparer for {preparer.input_layout.value}.")
            by_layout[preparer.input_layout] = preparer
        self._by_layout = MappingProxyType(by_layout)

    def resolve(
        self,
        input_layout: PCPInputLayout,
    ) -> PCPGroupInputPreparer:
        try:
            return self._by_layout[input_layout]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported PCP input layout {input_layout.value}.") from exc


def build_default_group_input_preparers() -> PCPGroupInputPreparerRegistry:
    return PCPGroupInputPreparerRegistry(
        (
            DualChunkGroupInputPreparer(),
            ContiguousStateGroupInputPreparer(),
        )
    )
