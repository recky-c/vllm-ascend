/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aclnn_kvpp_mte_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerKvppMteCopyGetWorkspaceSize(
    const aclTensor* anchor, const aclTensor* localOffsets,
    const aclTensor* stagingOffsets, const aclTensor* lengths,
    int64_t stagingBase, int64_t sourceRank, int64_t destinationRank,
    int64_t shmId, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerKvppMteCopy(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    const aclrtStream stream);

aclnnStatus aclnnKvppMteCopyGetWorkspaceSize(
    const aclTensor* anchor, const aclTensor* localOffsets,
    const aclTensor* stagingOffsets, const aclTensor* lengths,
    int64_t stagingBase, int64_t sourceRank, int64_t destinationRank,
    int64_t shmId, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerKvppMteCopyGetWorkspaceSize(
        anchor, localOffsets, stagingOffsets, lengths, stagingBase,
        sourceRank, destinationRank, shmId, workspaceSize, executor);
}

aclnnStatus aclnnKvppMteCopy(void* workspace, uint64_t workspaceSize,
                             aclOpExecutor* executor,
                             const aclrtStream stream)
{
    return aclnnInnerKvppMteCopy(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif
