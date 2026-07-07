// Kernel-side FastGelu implementation.
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ELEM_NUM = 1024;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length) {
        length_ = length;
        blockNum_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (blockNum_ == 0) {
            blockNum_ = 1;
        }

        blockLength_ = (length_ + blockNum_ - 1) / blockNum_;
        coreOffset_ = blockLength_ * AscendC::GetBlockIdx();
        currentBlockLength_ = length_ > coreOffset_ ? length_ - coreOffset_ : 0;
        if (currentBlockLength_ > blockLength_) {
            currentBlockLength_ = blockLength_;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, TILE_ELEM_NUM * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, TILE_ELEM_NUM * sizeof(DT_X));
        pipe_.InitBuffer(absBuf_, TILE_ELEM_NUM * sizeof(DT_X));
        pipe_.InitBuffer(denomBuf_, TILE_ELEM_NUM * sizeof(DT_X));
        pipe_.InitBuffer(workBuf_, TILE_ELEM_NUM * sizeof(DT_X));
        pipe_.InitBuffer(numerBuf_, TILE_ELEM_NUM * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (length_ == 0 || currentBlockLength_ <= 0) {
            return;
        }

        uint32_t tileNum = currentBlockLength_ / TILE_ELEM_NUM;
        uint32_t tailCount = currentBlockLength_ - tileNum * TILE_ELEM_NUM;

        for (uint32_t i = 0; i < tileNum; ++i) {
            ProcessTile(i * TILE_ELEM_NUM, TILE_ELEM_NUM);
        }

        if (tailCount > 0) {
            ProcessTile(tileNum * TILE_ELEM_NUM, tailCount);
        }
    }
private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t count) {
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};

        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> absLocal = absBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> denomLocal = denomBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> workLocal = workBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> numerLocal = numerBuf_.Get<DT_X>();

        AscendC::Abs(absLocal, xLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(denomLocal, absLocal, static_cast<DT_X>(-1.702f), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(denomLocal, denomLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<DT_X>(workLocal, static_cast<DT_X>(1.0f), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(denomLocal, denomLocal, workLocal, count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(workLocal, xLocal, absLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(workLocal, workLocal, static_cast<DT_X>(0.851f), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(workLocal, workLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(numerLocal, xLocal, workLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(yLocal, numerLocal, denomLocal, count);
        AscendC::PipeBarrier<PIPE_V>();

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    uint32_t length_ = 0;
    uint32_t blockNum_ = 1;
    uint32_t blockLength_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t currentBlockLength_ = 0;

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> absBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denomBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> numerBuf_;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length);
    op.Process();
}
