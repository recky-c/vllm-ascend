/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KVPP_MTE_COPY_TILING_H
#define KVPP_MTE_COPY_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KvppMteCopyTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, descriptorCount);
    TILING_DATA_FIELD_DEF(uint64_t, stagingBase);
    TILING_DATA_FIELD_DEF(int64_t, sourceRank);
    TILING_DATA_FIELD_DEF(int64_t, destinationRank);
    TILING_DATA_FIELD_DEF(int64_t, shmId);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvppMteCopy, KvppMteCopyTilingData)

struct KvppMteCopyCompileInfo {};
}  // namespace optiling

#endif
