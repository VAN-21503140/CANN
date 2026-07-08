# CANN / Ascend C Operator Development Reference

Source: extracted from the provided archive `华为(1).zip`. The archive contained 6 PPTX files, not PDFs: Ascend C quick start, programming model, operator development flow, debugging/tuning, large-model operator optimization, and Ascend AI processor architecture.

## 1. Architecture and Operator Choice

CANN is the Ascend heterogeneous computing stack. It connects AI frameworks and applications above to Ascend AI processors below, and includes AscendCL, operator libraries, framework adapters, ATC/TBE, Runtime/ACE, HCCL, AIPP, DVPP, and basic host-device services.

Operator categories:

- **AI Core operator**: runs on AI Core and targets dense matrix, vector, and scalar tensor computation.
- **AI CPU operator**: runs on AI CPU and fits work that is not suitable for AI Core, such as branch-heavy complex logic, discrete data processing, resource management, random-number-dependent logic, or data types unsupported by AI Core such as Complex32/Complex64. It can also unblock functional testing before an AI Core rewrite.

AI Core has three main compute resources:

- **Cube / matrix unit**: high-intensity matrix multiplication. FP16 can complete a 16x16 by 16x16 multiply per beat; int8 can complete 16x32 by 32x16 per beat.
- **Vector unit**: vector-scalar and vector-vector operations such as ReLU, pooling, BatchNorm, FP32/FP16/Int32/Int8 arithmetic.
- **Scalar unit**: loop control, branching, address/parameter calculation, instruction issue, and basic arithmetic.

Storage and control matter for performance: GM/HBM, L2, L1, L0A/L0B/L0C, UB, DMA queues, instruction queues, and Event Sync determine whether copy, compute, and synchronization overlap cleanly.

## 2. Development Modes

Use **quick mode** to validate an Ascend C kernel directly:

- Implement the Kernel-side function.
- Invoke with kernel launch syntax `<<<...>>>`.
- Use CPU-mode simulation for logic checks and NPU execution for board-level validation.

Use **standard mode** for deployable operators:

- Generate a project from an operator prototype JSON with `msopgen`.
- Implement Kernel-side operator logic.
- Implement Host-side Tiling, shape inference, and operator prototype registration.
- Compile/deploy an operator package.
- Call via ACLNN single-operator API, ACLOP single-operator model, PyTorch adapter, or whole-network replacement.

`msopgen` is typically under `{ASCEND_HOME_DIR}/python/site-packages/bin/msopgen`. Example generation shape:

```bash
${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
  -i $HOME/sample/add_custom.json \
  -c ai_core-<soc_version> \
  -lan cpp \
  -out $HOME/sample/AddCustom
```

Generated standard project areas:

- `op_kernel/`: Kernel-side implementation.
- `op_host/`: Host-side Tiling, prototype registration, shape inference, and info library code.
- `CMakePresets.json`: CANN path, cross-compile, vendor, and binary package settings.
- `build.sh`: package build entrypoint.

## 3. Kernel-Side Pattern

Ascend C uses standard C++ plus class-library APIs. Common API families:

- Compute APIs: vector APIs, matrix APIs.
- Copy APIs: `DataCopy`, `Copy`.
- Sync/queue/memory APIs: `TPipe`, `TQue`, `AllocTensor`, `FreeTensor`, `EnQue`, `DeQue`.

Common tensor types:

- `GlobalTensor<T>`: wraps Global Memory. Initialize with `SetGlobalBuffer((__gm__ T*)ptr, element_count)` and ensure the count does not exceed actual allocation.
- `LocalTensor<T>`: wraps AI Core local memory. Supports element access, `GetValue`, `SetValue`, `GetSize`, `operator[]`, and user tags.

Kernel skeleton:

```cpp
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
  KernelAdd op;
  op.Init(x, y, z);
  op.Process();
}

#ifndef __CCE_KT_TEST__
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                   uint8_t* x, uint8_t* y, uint8_t* z) {
  add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}
#endif
```

Operator class shape:

- `Init`: compute per-core offsets, call `SetGlobalBuffer`, allocate queues through `pipe.InitBuffer`.
- `Process`: loop over tile progress and call `CopyIn`, `Compute`, `CopyOut`.
- `CopyIn`: allocate local tensors, `DataCopy` from GM to local memory, then `EnQue`.
- `Compute`: `DeQue` inputs, allocate output local tensor, call compute API such as `Add`, then enqueue output.
- `CopyOut`: dequeue output, `DataCopy` back to GM, then free local tensor.

For a fixed-shape vector add example:

- `TOTAL_LENGTH = 8 * 2048`.
- `BLOCK_LENGTH = 2048` when split across 8 cores.
- Split each core's data into 8 tiles; with double buffer each tile is split into 2 blocks.
- `TILE_LENGTH = 128` in the course example.

## 4. Pipe, Queue, and Double Buffer

`TPipe` manages on-chip memory used for task-to-task data passing. Use `InitBuffer(queue, num, len)`:

- `queue`: created queue object.
- `num`: number of memory blocks; `num=2` enables double-buffer style use.
- `len`: bytes per block.

`TQue` handles communication and synchronization between pipeline tasks. Queue positions include vector queues (`VECIN`, `VECOUT`, `VECCALC`) and matrix queues (`A1`, `A2`, `B1`, `B2`, `CO1`, `CO2`).

Required discipline:

- Pair every `AllocTensor` with `FreeTensor`.
- Pair producer `EnQue` with consumer `DeQue`.
- Keep CopyIn/Compute/CopyOut order clear enough that data dependencies are obvious.
- Treat double buffer as a pipeline design, not just a constant. It helps when copy and compute overlap instead of serializing each tile.

## 5. Host-Side Standard Operator Work

Host-side code usually includes three responsibilities:

- **Tiling implementation**: calculate data split parameters such as total length, tile number, per-core work, workspace size, and tiling key.
- **Shape inference**: infer output shape, dtype, and format from inputs, operator logic, and attributes so memory can be prepared early.
- **Operator prototype registration**: define inputs, outputs, attributes, formats, dtypes, supported SoCs, and associate Tiling and shape inference functions.

Tiling data definition pattern:

```cpp
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)
}
```

Tiling function responsibilities:

- Read input shape: `context->GetInputTensor(0)->GetShapeSize()`.
- Set block count: `context->SetBlockDim(BLOCK_DIM)`.
- Save fields to raw tiling buffer and set data size.
- Set tiling key, for example `context->SetTilingKey(1)`.
- Set workspace sizes via `context->GetWorkspaceSizes(1)`.

Kernel dynamic-shape entry uses:

```cpp
GET_TILING_DATA(tiling_data, tiling);
op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
op.Process();
```

## 6. Matmul and Fusion Patterns

Matmul formula: `C = A * B + Bias`, where A is `[M, K]`, B is `[K, N]`, C is `[M, N]`, and Bias is usually `[1, N]`.

Multi-core split:

- Split A along M into `singleCoreM` chunks.
- Split B along N into `singleCoreN` chunks.
- Each core computes `singleCoreM * singleCoreN` output.

In-core split:

- Split A by `baseM` and `baseK`.
- Split B by `baseN` and `baseK`.
- Accumulate partial products into the corresponding C block.

Matmul high-level API pattern:

- Define a `Matmul` object with input/output positions and formats.
- Initialize `GlobalTensor` inputs/outputs and bias.
- Calculate per-core offsets from `GetBlockIdx()` and `TCubeTiling`.
- Call `SetSysWorkspace(workspace)` and verify workspace pointer.
- `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj)`.
- Set A, B, optional bias.
- Use `IterateAll(cGlobal)` for full computation or `Iterate()` / `GetTensorC()` when post-processing is needed.
- End with `matmulObj.End()`.

Fusion operators benefit from the separation between Cube and Vector units. A matmul + activation operator such as `matmul_leakyrelu_custom` can avoid unnecessary GM round trips and exploit Cube/Vector parallelism.

## 7. Build, Deploy, and Call

Build configuration usually requires `CMakePresets.json` updates:

- `ASCEND_CANN_PACKAGE_PATH`: actual CANN installation path, for example `/usr/local/Ascend/latest`.
- `ENABLE_CROSS_COMPILE`: match the target environment.
- `CMAKE_CROSS_PLATFORM_COMPILER`: cross compiler path when cross-compiling.
- `vendor_name`: affects deployment directory, commonly `customize`.

Build and deploy:

```bash
./build.sh
# run package appears under build_out/
./custom_opp_xxx.run
# or install elsewhere:
./custom_opp_xxx.run --install-path=xxx
```

If using a custom install path, export `ASCEND_CUSTOM_OPP_PATH` so the operator becomes visible.

ACLNN single-operator API call pattern:

1. Initialize AscendCL.
2. Construct inputs/outputs. For `aclTensor`, create host values, allocate device memory with `aclrtMalloc`, copy host to device, then call `aclCreateTensor`.
3. Call the first-stage API to calculate workspace size.
4. Allocate workspace on device.
5. Call the second-stage API to run the operator.
6. Copy output back with `aclrtMemcpy` if printing or validating on host.
7. Destroy tensors with `aclDestroyTensor` and release device memory with `aclrtFree`.

Compile a test project with a typical flow:

```bash
mkdir -p build
cmake ./ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin
./opapi_test
```

PyTorch adapter path:

- Ensure the custom operator has been compiled and deployed as a binary package.
- Add the operator API library path to the shared-library search path.
- Add a custom entry in `npu_native_functions.yaml`, such as `npu_add_custom(Tensor x, Tensor y) -> Tensor`.
- Implement an adapter file named like `AddCustomKernelNpu.cpp`.
- Use `EXEC_NPU_CMD(aclnnAddCustom, x, y, result)` inside the adapter.
- Replace network code with the new API, for example `torch_npu.npu_add_custom(x, y)`.

## 8. Correctness Debugging

Ascend C supports twin debugging:

- **CPU domain**: debug precision and logic. Use gdb, multi-process gdb where needed, and `printf`. Guard debug prints with `__CCE_KT_TEST__` because NPU mode does not generally support ordinary printf in the same way.
- **NPU domain**: debug on-device behavior. Enable DumpTensor/PRINTF build options, add `DumpTensor` or `PRINTF` in Kernel code, and use `msdebug` for breakpoints, single-step, and variable printing.

`msdebug` supports:

- Breakpoints by function entry, source line, or address.
- Single-step execution after the kernel pauses.
- Variable printing in breakpoint scope.

Use NPU-domain debugging when CPU simulation is correct but board execution differs.

## 9. Sanitizing and Profiling

`mssanitizer` detects memory and race issues:

- Illegal read/write: often GM/HBM allocation size does not match Kernel access range.
- Multi-core stomping: multiple AI Cores access overlapping GM regions without synchronization or correct ownership.
- Misaligned access: DMA address or length does not align to the memory type's minimum access granularity.
- Memory leak or illegal free: usually device-side `aclrtMalloc` memory is not released, released from the wrong address, or double-freed.
- Race detection: WAW, WAR, and RAW risks when multiple events access the same memory and at least one is a write.

`msprof` collects performance data:

- Compute instruction time and ratio.
- Copy instruction time and ratio.
- Copy bandwidth.
- Resource conflicts, including bank and bank-group conflicts.
- Cache hit rate, especially reuse from GM/HBM through L2 to L1/UB.
- Simulation outputs such as instruction pipeline view, hot code lines, and instruction-to-source mapping.

## 10. Performance Optimization

Optimize in this order:

1. **Tiling split**: balance workload across AI Cores, use all available compute resources, and handle tails explicitly.
2. **In-core tile size**: prefer larger tiles when local memory permits, reducing scalar loop overhead and helping SIMD efficiency.
3. **Double buffer / pipeline**: overlap data copy and compute through `TQue` buffers and CopyIn/Compute/CopyOut staging.
4. **Scalar optimization**: remove repeated loop calculations, reduce expensive arithmetic, move host-computable values out of Kernel code, and avoid using scalar unit for tensor data computation.
5. **API level**: consider lower-level APIs when high-level APIs hide hardware opportunities or introduce overhead.
6. **ICache**: reduce hot code size and place frequently executed branches earlier to reduce ICache misses.
7. **Hierarchical memory**: reuse `TBuf` instead of repeatedly allocating temporary buffers; merge same-lifetime TQue buffers where safe.
8. **Matrix resource reuse**: exploit L2, L1, L0A/B/C, and UB locality. If one operand fits entirely into L0 or L1, keep it resident and stream the other operand.

For large attention operators:

- Self-attention forward can be viewed as QK^T, position/scale/mask, Softmax, dropout, P*V, and rescale.
- Safe Softmax is 3-pass; Online Softmax reduces to 2-pass; FlashAttention-style Softmax reduces memory traffic further and can operate blockwise.
- FlashAttention-2 simplified implementation splits into matrix partitioning, QK^T matmul, Softmax/SoftmaxFlashV2 update, second matmul with V, and output merge.
- Matrix partitioning should compute per-core data volume and loop counts from shape and hardware limits. Common splits are by `b,n` or `b,n,s`; use greatest-common-divisor style splitting to distribute batch/head/sequence work and define main cores vs tail cores.
- For blockwise attention, store/update `rowmax` and `rowsum` in GM when multiple passes or block updates are required.
