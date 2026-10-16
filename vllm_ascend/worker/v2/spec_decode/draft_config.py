# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from typing import Any

from vllm.config import VllmConfig, replace
from vllm.logger import logger

from vllm_ascend.ascend_config import validate_additional_config_bool


def _draft_additional_config(vllm_config: VllmConfig) -> dict[str, Any] | None:
    """Return a CPP-disabled copy when a PP target creates a PP=1 draft."""
    if vllm_config.parallel_config.pipeline_parallel_size <= 1:
        return None

    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return None

    scheduler_config = additional_config.get("scheduler_config")
    if scheduler_config is None:
        scheduler_config = {}
    elif not isinstance(scheduler_config, dict):
        return None

    # Match SchedulerConfig.from_additional_config: the nested form wins over
    # the deprecated top-level form when both are present.
    if "profiling_chunk_config" in scheduler_config:
        profiling_chunk_config = scheduler_config["profiling_chunk_config"]
        config_path = "additional_config.scheduler_config.profiling_chunk_config"
        nested = True
    elif "profiling_chunk_config" in additional_config:
        profiling_chunk_config = additional_config["profiling_chunk_config"]
        config_path = "additional_config.profiling_chunk_config"
        nested = False
    else:
        return None

    if not isinstance(profiling_chunk_config, dict):
        return None

    enabled = validate_additional_config_bool(
        profiling_chunk_config.get("enabled", False),
        f"{config_path}.enabled",
    )
    if not enabled:
        return None

    draft_profiling_chunk_config = profiling_chunk_config.copy()
    draft_profiling_chunk_config["enabled"] = False

    draft_additional_config = additional_config.copy()
    if nested:
        draft_scheduler_config = scheduler_config.copy()
        draft_scheduler_config["profiling_chunk_config"] = draft_profiling_chunk_config
        draft_additional_config["scheduler_config"] = draft_scheduler_config
    else:
        draft_additional_config["profiling_chunk_config"] = draft_profiling_chunk_config
    return draft_additional_config


@contextmanager
def disable_profiling_chunk_for_draft(vllm_config: VllmConfig) -> Iterator[None]:
    """Temporarily expose CPP-disabled input while constructing a PP=1 draft.

    The draft still goes through the normal ``VllmConfig.replace`` validation.
    Rebinding the target's ``additional_config`` only for that call lets the
    resulting draft retain the copied input while the target is restored.
    """
    draft_additional_config = _draft_additional_config(vllm_config)
    if draft_additional_config is None:
        yield
        return

    target_additional_config = vllm_config.additional_config
    vllm_config.additional_config = draft_additional_config
    try:
        yield
    finally:
        vllm_config.additional_config = target_additional_config


def replace_draft_parallel_config(
    vllm_config: VllmConfig,
    parallel_changes: dict[str, Any],
    **config_changes: Any,
) -> VllmConfig:
    """Re-validate a target-derived draft config, keeping the target DCP size.

    The target already validated its DCP size. A replicated PCP draft keeps
    the DCP span over the PCP axis, which a PCP=1 config cannot express, so
    validate with DCP=1 and restore the target value afterwards.
    """
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    draft_config = replace(
        vllm_config,
        parallel_config=replace(
            vllm_config.parallel_config,
            decode_context_parallel_size=1,
            **parallel_changes,
        ),
        **config_changes,
    )
    draft_config.parallel_config.decode_context_parallel_size = dcp_size
    return draft_config


def derive_replicated_draft_config(vllm_config: VllmConfig) -> VllmConfig:
    """Return the config a draft runs with: replicated on every PCP rank.

    Speculative drafts never take part in PCP sharding; they run with PCP=1 on
    the global batch restored by AscendPCPManager.
    """
    target_parallel_config = vllm_config.parallel_config
    if target_parallel_config.prefill_context_parallel_size == 1:
        return vllm_config

    parallel_changes: dict[str, Any] = {"prefill_context_parallel_size": 1}
    config_changes: dict[str, Any] = {}
    single_dp_tp_rank = target_parallel_config.tensor_parallel_size == 1 and (
        target_parallel_config.data_parallel_size == 1
    )
    if target_parallel_config.enable_eplb and single_dp_tp_rank:
        # PCP was the only EP dimension; the PCP=1 draft keeps a static layout.
        parallel_changes["enable_eplb"] = False
        # Copy declared fields without transient communicator-selection state.
        parallel_changes["eplb_config"] = dataclass_replace(target_parallel_config.eplb_config, num_redundant_experts=0)
        logger.warning_once("EPLB is disabled for the replicated PCP draft model; target EPLB remains enabled.")
    engram_config = vllm_config.engram_config
    if engram_config is not None and engram_config.dp_shared_memory and target_parallel_config.data_parallel_size == 1:
        # The draft has no Engram tables; sharing belongs to the target.
        config_changes["engram_config"] = replace(engram_config, dp_shared_memory=False)
    return replace_draft_parallel_config(vllm_config, parallel_changes, **config_changes)
