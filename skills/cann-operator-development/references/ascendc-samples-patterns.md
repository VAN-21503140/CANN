# Ascend C Sample-Derived Patterns

Source: cloned from the public GitCode CANN sample repository `https://gitcode.com/cann/cann-samples.git` on 2026-07-07. Use this reference when implementing practical Ascend C kernels, especially elementwise vector operators such as FastGelu.

## 1. Elementwise Vector Kernel Skeleton

The `Samples/0_Introduction/vector_add/main.asc` sample shows a compact one-file vector operator pattern:

```cpp
constexpr static int64_t PIPELINE_DEPTH = 2;
AscendC::TPipe pipe;
AscendC::GlobalTensor<T> xGm, yGm, zGm;
AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ;

pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
```

For each core, compute the real per-core length:

```cpp
int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
if (currentBlockLength > blockLength) {
    currentBlockLength = blockLength;
}
if (currentBlockLength <= 0) {
    return;
}
```

This is important when Host sets `blockDim` to all AIV cores; short tensors leave some cores with no work.

## 2. DataCopyPad for Tail-Safe Movement

The vector add sample uses `DataCopyPad` rather than plain `DataCopy` so non-32B tail blocks can be handled safely:

```cpp
AscendC::DataCopyExtParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = elementNumPerTile * sizeof(T);
copyParams.srcStride = 0;
copyParams.dstStride = 0;
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
inQueueX.EnQue(xLocal);
```

Copy-out uses the same `copyParams` shape:

```cpp
zLocal = outQueueZ.DeQue<T>();
AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
outQueueZ.FreeTensor(zLocal);
```

For FastGelu, use this pattern when the last tile or a small input is not naturally aligned. If plain `DataCopy` is used, tile lengths should be selected so byte lengths are aligned for the target hardware, and a safe tail path is still required.

## 3. Queue Discipline

The sample follows a strict queue lifecycle:

1. `AllocTensor` from input queue.
2. Copy GM to UB.
3. `EnQue` producer output.
4. `DeQue` before compute.
5. Allocate output local tensor.
6. Run vector compute.
7. `EnQue` output.
8. `FreeTensor` inputs.
9. `DeQue` output.
10. Copy UB to GM.
11. `FreeTensor` output.

For FastGelu, this flow can be simplified to one input queue and one output queue, plus `TBuf` temporary buffers for intermediate tensors, or multiple queues if using a staged pipeline.

## 4. Vector API Signatures Confirmed from Samples

From `cann-samples` performance examples:

```cpp
Abs(dstLocal, srcLocal, count);
Muls(dstLocal, srcLocal, scalar, count);
AscendC::Muls(dstLocal, srcLocal, scalar, count);
Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_NONE, count);
Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_RINT, count);
Div(dstLocal, lhsLocal, rhsLocal, count);
Duplicate<T>(dstLocal, value, count);
Add(dstLocal, lhsLocal, rhsLocal, count);
Sub(dstLocal, lhsLocal, rhsLocal, count);
```

Examples also place `PipeBarrier<PIPE_V>()` between dependent vector operations in more complex flows. For a sequence like FastGelu, add barriers conservatively between operations when compile or runtime behavior indicates hazards; remove only after profiling confirms correctness and benefit.

## 5. FastGelu Formula Mapping

Stable MindSpore FastGelu formula:

```text
y = x / (1 + exp(-1.702 * abs(x))) * exp(0.851 * (x - abs(x)))
```

Natural Ascend C vector steps for `float` intermediates:

```cpp
Abs(absX, x, n);                         // absX = abs(x)
Muls(t1, absX, -1.702f, n);              // t1 = -1.702 * abs(x)
Exp(t1, t1, n);                          // t1 = exp(t1)
Duplicate(one, 1.0f, n);
Add(denom, t1, one, n);                  // denom = 1 + t1
Sub(t2, x, absX, n);                     // t2 = x - abs(x)
Muls(t2, t2, 0.851f, n);                 // t2 = 0.851 * (x - abs(x))
Exp(t2, t2, n);                          // t2 = exp(t2)
Mul(numer, x, t2, n);                    // numer = x * t2
Div(y, numer, denom, n);                 // y = numer / denom
```

For `float16` input, prefer internal `float` computation if the precision budget is tight:

```cpp
Cast(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, n);
// compute in float
Cast(yHalf, yFloat, AscendC::RoundMode::CAST_RINT, n);
```

If performance matters more and accuracy passes, compute directly in half. Keep both routes as candidates and benchmark.

## 6. Tile Size Heuristics for Elementwise Activations

FastGelu uses many temporaries: input, output, abs, two exponent inputs/outputs, denominator, numerator, and optional fp16-to-fp32 cast buffers. Tile sizing must account for all live UB buffers, not just input/output.

Practical starting points:

- Use `DataCopyPad` for general tail safety.
- Keep `BUFFER_NUM = 2` for input/output queues when pipelining copy and compute.
- Use `TBuf` for reusable intermediate tensors when there are many temporaries.
- Align normal tile byte sizes to at least 32B.
- Ensure the final tail path uses the actual valid element count for vector compute and copy.
- For small tensors, reduce or guard blockDim to avoid empty-core out-of-bounds access.

## 7. Repository Layout Notes

Useful sample locations in `cann-samples`:

- `Samples/0_Introduction/vector_add/main.asc`: basic TPipe/TQue/DataCopyPad/CopyIn-Compute-CopyOut elementwise skeleton.
- `Samples/0_Introduction/vector_function_getting_started/README.md`: vector programming concepts, SIMD, masks, and tail handling concepts.
- `Samples/2_Performance/moe_dispatch_and_combine_story/include/`: practical `Abs`, `Muls`, `Cast`, `Duplicate`, `DataCopyPad`, `PipeBarrier`, and alignment examples.
- `Samples/2_Performance/full_quant_fused_infer_attention_score_story/include/`: advanced vector-function and `Div` usage patterns.