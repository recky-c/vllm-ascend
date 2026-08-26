/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kvpp_mte_copy_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr size_t LOCAL_OFFSETS_INPUT_INDEX = 1;
constexpr size_t STAGING_OFFSETS_INPUT_INDEX = 2;
constexpr size_t LENGTHS_INPUT_INDEX = 3;
constexpr size_t STAGING_BASE_ATTR_INDEX = 0;
constexpr size_t SOURCE_RANK_ATTR_INDEX = 1;
constexpr size_t DESTINATION_RANK_ATTR_INDEX = 2;
constexpr size_t SHM_ID_ATTR_INDEX = 3;
constexpr uint32_t MAX_CORE_COUNT = 32;
constexpr int64_t MAX_SHM_ID = 64;

ge::graphStatus KvppMteCopyTiling(gert::TilingContext* context)
{
    const auto* localOffsets = context->GetInputShape(LOCAL_OFFSETS_INPUT_INDEX);
    const auto* stagingOffsets = context->GetInputShape(STAGING_OFFSETS_INPUT_INDEX);
    const auto* lengths = context->GetInputShape(LENGTHS_INPUT_INDEX);
    if (localOffsets == nullptr || stagingOffsets == nullptr || lengths == nullptr) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE descriptor shapes must not be null.");
        return ge::GRAPH_FAILED;
    }

    const auto& localShape = localOffsets->GetStorageShape();
    const auto& stagingShape = stagingOffsets->GetStorageShape();
    const auto& lengthShape = lengths->GetStorageShape();
    if (localShape.GetDimNum() != 1 || stagingShape.GetDimNum() != 1 ||
        lengthShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE descriptors must be one-dimensional.");
        return ge::GRAPH_FAILED;
    }
    const int64_t descriptorCount = localShape.GetShapeSize();
    if (descriptorCount <= 0 || stagingShape.GetShapeSize() != descriptorCount ||
        lengthShape.GetShapeSize() != descriptorCount) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE descriptor lengths must be equal and positive.");
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE attributes must not be null.");
        return ge::GRAPH_FAILED;
    }
    const auto* stagingBase = attrs->GetAttrPointer<int64_t>(STAGING_BASE_ATTR_INDEX);
    const auto* sourceRank = attrs->GetAttrPointer<int64_t>(SOURCE_RANK_ATTR_INDEX);
    const auto* destinationRank = attrs->GetAttrPointer<int64_t>(DESTINATION_RANK_ATTR_INDEX);
    const auto* shmId = attrs->GetAttrPointer<int64_t>(SHM_ID_ATTR_INDEX);
    if (stagingBase == nullptr || sourceRank == nullptr ||
        destinationRank == nullptr || shmId == nullptr) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE attributes must not be null.");
        return ge::GRAPH_FAILED;
    }
    if (*stagingBase <= 0 || *sourceRank < -1 || *destinationRank < -1 ||
        ((*sourceRank >= 0) == (*destinationRank >= 0)) ||
        *shmId < 0 || *shmId >= MAX_SHM_ID) {
        OP_LOGE(context->GetNodeName(), "KVPP MTE attributes are invalid.");
        return ge::GRAPH_FAILED;
    }

    KvppMteCopyTilingData tilingData;
    tilingData.set_descriptorCount(static_cast<uint64_t>(descriptorCount));
    tilingData.set_stagingBase(static_cast<uint64_t>(*stagingBase));
    tilingData.set_sourceRank(*sourceRank);
    tilingData.set_destinationRank(*destinationRank);
    tilingData.set_shmId(*shmId);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    context->SetBlockDim(static_cast<uint32_t>(
        std::min<int64_t>(descriptorCount, MAX_CORE_COUNT)));
    context->SetTilingKey(1);
    tilingData.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvppMteCopyTilingParse(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(KvppMteCopy)
    .Tiling(KvppMteCopyTiling)
    .TilingParse<KvppMteCopyCompileInfo>(KvppMteCopyTilingParse);
}  // namespace optiling
