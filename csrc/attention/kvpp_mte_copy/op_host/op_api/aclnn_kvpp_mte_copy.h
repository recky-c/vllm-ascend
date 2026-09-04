/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ACLNN_KVPP_MTE_COPY_H
#define ACLNN_KVPP_MTE_COPY_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first-stage interface calculates the workspace size and creates
 * the executor for the KVPP MTE copy operation.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvppMteCopyGetWorkspaceSize(
    const aclTensor* anchor, const aclTensor* localOffsets,
    const aclTensor* stagingOffsets, const aclTensor* lengths,
    int64_t stagingBase, int64_t sourceRank, int64_t destinationRank,
    int64_t shmId, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second-stage interface executes the KVPP MTE copy operation.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvppMteCopy(void* workspace, uint64_t workspaceSize,
                             aclOpExecutor* executor,
                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_KVPP_MTE_COPY_H
