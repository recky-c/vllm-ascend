# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch

from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding

from vllm_ascend.utils import should_skip_allreduce_across_dp_group


def sync_cudagraph_and_dp_padding_ascend(
    cudagraph_manager: CudaGraphManager,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    vllm_config=None,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Ascend-specific version of sync_cudagraph_and_dp_padding.

    Compared to the GPU version, this function adds the
    should_skip_allreduce_across_dp_group optimization: for dense models
    or MoE models using MC2 communication, the all_reduce across DP ranks
    can be skipped because each rank can operate independently with its
    own token count. In this case we construct num_tokens_across_dp
    locally without any inter-rank communication.

    When the optimization does not apply, we fall back to the upstream
    GPU implementation which performs an all_reduce to synchronize
    batch descriptors and DP padding across all ranks.
    """
    assert dp_size > 1, "DP size must be greater than 1"

    if vllm_config is not None and should_skip_allreduce_across_dp_group(
        vllm_config
    ):
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )
        return desired_batch_desc, num_tokens_across_dp

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        desired_batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
    )
