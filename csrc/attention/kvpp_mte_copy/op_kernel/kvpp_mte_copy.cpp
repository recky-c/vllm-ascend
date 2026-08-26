/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"
#include "smem/device/smem_shm_aicore_base_api.h"

namespace {
constexpr uint32_t TILE_BYTES = 64 * 1024;
constexpr int32_t EVENT_ID = 0;

struct KvppMteCopyTilingData {
    uint64_t descriptorCount;
    uint64_t stagingBase;
    int64_t sourceRank;
    int64_t destinationRank;
    int64_t shmId;
};
}  // namespace

extern "C" __global__ __aicore__ void kvpp_mte_copy(
    GM_ADDR anchor, GM_ADDR localOffsets, GM_ADDR stagingOffsets,
    GM_ADDR lengths, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(KvppMteCopyTilingData, tilingData, tiling);
    const int32_t sourceRank = static_cast<int32_t>(tilingData.sourceRank);
    const int32_t destinationRank =
        static_cast<int32_t>(tilingData.destinationRank);
    const uint32_t shmId = static_cast<uint32_t>(tilingData.shmId);
    const uint64_t symmetricSize = smem_shm_get_symmetric_size(shmId);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> buffer;
    pipe.InitBuffer(buffer, 1, TILE_BYTES);
    AscendC::LocalTensor<uint8_t> local = buffer.AllocTensor<uint8_t>();
    __ubuf__ uint8_t* ubAddress =
        reinterpret_cast<__ubuf__ uint8_t*>(local.address_.bufferAddr);

    AscendC::GlobalTensor<int64_t> localOffsetDescriptor;
    AscendC::GlobalTensor<int64_t> stagingOffsetDescriptor;
    AscendC::GlobalTensor<int64_t> lengthDescriptor;
    localOffsetDescriptor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(localOffsets),
        tilingData.descriptorCount);
    stagingOffsetDescriptor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(stagingOffsets),
        tilingData.descriptorCount);
    lengthDescriptor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(lengths),
        tilingData.descriptorCount);

    const uint64_t coreIndex = AscendC::GetBlockIdx();
    const uint64_t coreCount = AscendC::GetBlockNum();
    for (uint64_t descriptor = coreIndex;
         descriptor < tilingData.descriptorCount;
         descriptor += coreCount) {
        const uint64_t length =
            static_cast<uint64_t>(lengthDescriptor.GetValue(descriptor));
        if (length == 0) {
            continue;
        }
        const uint64_t localOffset = static_cast<uint64_t>(
            localOffsetDescriptor.GetValue(descriptor));
        const uint64_t stagingOffset = static_cast<uint64_t>(
            stagingOffsetDescriptor.GetValue(descriptor));
        __gm__ uint8_t* localGm = anchor + localOffset;
        __gm__ uint8_t* stagingAddress =
            reinterpret_cast<__gm__ uint8_t*>(tilingData.stagingBase) +
            stagingOffset;
        if (sourceRank >= 0) {
            stagingAddress += symmetricSize * static_cast<uint64_t>(sourceRank);
        } else {
            stagingAddress +=
                symmetricSize * static_cast<uint64_t>(destinationRank);
        }

        uint64_t offset = 0;
        while (offset < length) {
            const uint32_t bytes = static_cast<uint32_t>(
                (length - offset) > TILE_BYTES ? TILE_BYTES : length - offset);
            __gm__ uint8_t* source = sourceRank >= 0
                ? stagingAddress + offset
                : localGm + offset;
            smem_shm_copy_gm2ub<uint8_t>(ubAddress, source, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            __gm__ uint8_t* destination = destinationRank >= 0
                ? stagingAddress + offset
                : localGm + offset;
            smem_shm_copy_ub2gm<uint8_t>(
                destination, ubAddress, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            offset += bytes;
        }
    }
    buffer.FreeTensor(local);
}
