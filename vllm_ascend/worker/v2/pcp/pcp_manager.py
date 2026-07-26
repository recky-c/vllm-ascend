# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.pcp.capability import (
    resolve_group_capabilities,
)
from vllm_ascend.worker.v2.pcp.validation import (
    validate_hybrid_pcp_config,
)


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    def __init__(
        self,
        *args,
        vllm_config: VllmConfig,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""

        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = (
            local_batch.num_computed_tokens_np
            + local_batch.num_scheduled_tokens
        )
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens,
        )
        return local_batch


def maybe_build_ascend_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]] | None = None,
) -> AscendPCPManager | None:
    """Build the standard or Hybrid PCP manager at the composition root."""

    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    dcp_size = parallel_config.decode_context_parallel_size
    common_kwargs = {
        "pcp_world_size": pcp_size,
        "pcp_rank": get_pcp_group().rank_in_group,
        "device": device,
        "req_states": req_states,
        "max_num_reqs": vllm_config.scheduler_config.max_num_seqs,
        "max_num_tokens": (vllm_config.scheduler_config.max_num_batched_tokens),
        "block_tables": block_tables,
        "dcp_world_size": dcp_size,
        "dcp_rank": (get_dcp_group().rank_in_group if dcp_size > 1 else 0),
        "cp_interleave": parallel_config.cp_kv_cache_interleave_size,
        "vllm_config": vllm_config,
    }
    model_config = getattr(vllm_config, "model_config", None)
    is_hybrid = getattr(model_config, "is_hybrid", False)
    if not is_hybrid:
        PCPManager.validate_config(vllm_config, supports_mm_inputs)
        return AscendPCPManager(**common_kwargs)

    validate_hybrid_pcp_config(vllm_config, supports_mm_inputs)
    if attn_groups is None:
        raise RuntimeError("Hybrid PCP requires initialized attention groups.")
    group_capabilities = resolve_group_capabilities(attn_groups)
    if (
        vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.PIECEWISE
        and any(
            not capability.supports_piecewise
            for capability in group_capabilities
        )
    ):
        raise NotImplementedError(
            "At least one Hybrid PCP group does not support PIECEWISE "
            "graph mode."
        )

    # Lazy imports keep the standard MRv2 path independent of Hybrid modules.
    from vllm_ascend.worker.v2.pcp.group_preparer import (
        build_default_group_input_preparers,
    )
    from vllm_ascend.worker.v2.pcp.hybrid_pcp_manager import (
        AscendHybridPCPManager,
    )

    return AscendHybridPCPManager(
        **common_kwargs,
        group_capabilities=group_capabilities,
        group_input_preparers=build_default_group_input_preparers(),
    )
