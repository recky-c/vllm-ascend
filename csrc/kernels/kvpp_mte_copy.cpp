/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"

#if __has_include("smem/device/smem_shm_aicore_base_api.h")
#include "smem/device/smem_shm_aicore_base_api.h"

namespace {
constexpr uint32_t KVPP_MTE_TILE_BYTES = 64 * 1024;
constexpr uint32_t KVPP_MTE_MAX_CORES = 32;
constexpr int32_t KVPP_MTE_EVENT_ID = 0;
} // namespace

extern "C" __global__ __aicore__ void kvpp_mte_copy_pages(
    __gm__ uint8_t* local_base, __gm__ int64_t* page_ids,
    __gm__ int8_t* valid_mask, __gm__ int64_t* local_base_offsets,
    __gm__ int64_t* block_strides, __gm__ int64_t* block_bytes,
    __gm__ int64_t* staging_buffer_offsets, uint64_t num_pages,
    uint64_t num_buffers, __gm__ uint8_t* staging_base,
    int32_t source_rank, int32_t destination_rank, uint32_t shm_id)
{
    const uint64_t symmetric_size =
        (source_rank >= 0 || destination_rank >= 0)
            ? smem_shm_get_symmetric_size(shm_id)
            : 0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> buffer;
    pipe.InitBuffer(buffer, 1, KVPP_MTE_TILE_BYTES);
    AscendC::LocalTensor<uint8_t> local = buffer.AllocTensor<uint8_t>();
    __ubuf__ uint8_t* ub_address =
        reinterpret_cast<__ubuf__ uint8_t*>(local.address_.bufferAddr);

    AscendC::GlobalTensor<int64_t> page_id_descriptor;
    AscendC::GlobalTensor<int8_t> valid_mask_descriptor;
    AscendC::GlobalTensor<int64_t> local_base_offset_descriptor;
    AscendC::GlobalTensor<int64_t> block_stride_descriptor;
    AscendC::GlobalTensor<int64_t> block_byte_descriptor;
    AscendC::GlobalTensor<int64_t> staging_buffer_offset_descriptor;
    page_id_descriptor.SetGlobalBuffer(page_ids, num_pages);
    valid_mask_descriptor.SetGlobalBuffer(valid_mask, num_pages);
    local_base_offset_descriptor.SetGlobalBuffer(local_base_offsets, num_buffers);
    block_stride_descriptor.SetGlobalBuffer(block_strides, num_buffers);
    block_byte_descriptor.SetGlobalBuffer(block_bytes, num_buffers);
    staging_buffer_offset_descriptor.SetGlobalBuffer(staging_buffer_offsets,
                                                     num_buffers);

    const uint64_t total_descriptors = num_pages * num_buffers;
    const uint64_t core_index = AscendC::GetBlockIdx();
    const uint64_t core_count = AscendC::GetBlockNum();
    for (uint64_t descriptor = core_index; descriptor < total_descriptors;
         descriptor += core_count) {
        const uint64_t buffer_id = descriptor / num_pages;
        const uint64_t slot_id = descriptor % num_pages;

        const int8_t valid = valid_mask_descriptor.GetValue(slot_id);
        if (valid == 0) {
            continue;
        }
        const uint64_t page_id = static_cast<uint64_t>(
            page_id_descriptor.GetValue(slot_id));
        const uint64_t local_base_offset = static_cast<uint64_t>(
            local_base_offset_descriptor.GetValue(buffer_id));
        const uint64_t block_stride = static_cast<uint64_t>(
            block_stride_descriptor.GetValue(buffer_id));
        const uint64_t block_byte = static_cast<uint64_t>(
            block_byte_descriptor.GetValue(buffer_id));
        const uint64_t staging_buffer_offset = static_cast<uint64_t>(
            staging_buffer_offset_descriptor.GetValue(buffer_id));

        const uint64_t local_offset = local_base_offset + page_id * block_stride;
        const uint64_t staging_offset =
            staging_buffer_offset + slot_id * block_byte;
        const uint64_t length = block_byte;

        __gm__ uint8_t* local_gm = local_base + local_offset;
        __gm__ uint8_t* staging_address = staging_base + staging_offset;
        if (source_rank >= 0) {
            staging_address +=
                symmetric_size * static_cast<uint64_t>(source_rank);
        }
        if (destination_rank >= 0) {
            staging_address +=
                symmetric_size * static_cast<uint64_t>(destination_rank);
        }

        uint64_t offset = 0;
        while (offset < length) {
            const uint32_t bytes = static_cast<uint32_t>(
                (length - offset) > KVPP_MTE_TILE_BYTES
                    ? KVPP_MTE_TILE_BYTES
                    : (length - offset));
            __gm__ uint8_t* source = source_rank >= 0
                ? staging_address + offset
                : local_gm + offset;
            smem_shm_copy_gm2ub<uint8_t>(
                ub_address, source, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                KVPP_MTE_EVENT_ID);
            __gm__ uint8_t* destination = destination_rank >= 0
                ? staging_address + offset
                : local_gm + offset;
            smem_shm_copy_ub2gm<uint8_t>(
                destination, ub_address, bytes, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                KVPP_MTE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                KVPP_MTE_EVENT_ID);
            offset += bytes;
        }
    }
    buffer.FreeTensor(local);
}

namespace vllm_ascend {
void kvpp_mte_copy_pages_impl(
    void* stream, void* local_base, void* page_ids, void* valid_mask,
    void* local_base_offsets, void* block_strides, void* block_bytes,
    void* staging_buffer_offsets, uint64_t num_pages, uint64_t num_buffers,
    void* staging_base, int32_t source_rank, int32_t destination_rank,
    uint32_t shm_id)
{
    const uint64_t total_descriptors = num_pages * num_buffers;
    const uint32_t block_dim = total_descriptors < KVPP_MTE_MAX_CORES
        ? static_cast<uint32_t>(total_descriptors)
        : KVPP_MTE_MAX_CORES;
    kvpp_mte_copy_pages<<<block_dim, nullptr, stream>>>(
        local_base, page_ids, valid_mask, local_base_offsets, block_strides,
        block_bytes, staging_buffer_offsets, num_pages, num_buffers,
        staging_base, source_rank, destination_rank, shm_id);
}
} // namespace vllm_ascend
#endif
