#include "kernel_operator.h"

extern "C" __global__ __aicore__ void kv_copy_kernel(
    __gm__ uint8_t* source, __gm__ uint8_t* destination,
    __gm__ uint64_t* descriptors, uint64_t count, uint32_t no_l2)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> buffer;
    pipe.InitBuffer(buffer, 1, 65536);
    auto tile = buffer.AllocTensor<uint8_t>();
    AscendC::GlobalTensor<uint64_t> plan;
    plan.SetGlobalBuffer(descriptors, count * 3);
    for (uint64_t i = AscendC::GetBlockIdx(); i < count;
         i += AscendC::GetBlockNum()) {
        const uint64_t length = plan.GetValue(i * 3 + 2);
        // A masked descriptor may contain poison offsets: never evaluate them.
        if (length == 0) continue;
        const uint64_t src_offset = plan.GetValue(i * 3);
        const uint64_t dst_offset = plan.GetValue(i * 3 + 1);
        AscendC::GlobalTensor<uint8_t> src, dst;
        src.SetGlobalBuffer(source + src_offset, length);
        dst.SetGlobalBuffer(destination + dst_offset, length);
        if (no_l2) {
            src.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            dst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        for (uint64_t off = 0; off < length; off += 65536) {
            uint32_t bytes = static_cast<uint32_t>(length - off > 65536 ? 65536 : length - off);
            AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(tile, src[off], params, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::DataCopyPad(dst[off], tile, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
    }
    buffer.FreeTensor(tile);
}

extern "C" void kv_copy_launch(uint32_t cores, void* stream, void* source,
    void* destination, void* descriptors, uint64_t count, uint32_t no_l2)
{
    if (count == 0) return;
    kv_copy_kernel<<<cores, nullptr, stream>>>(source, destination, descriptors, count, no_l2);
}
