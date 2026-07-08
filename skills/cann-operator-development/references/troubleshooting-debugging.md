# Ascend C Debugging and Troubleshooting

Use this reference for Ascend C functional debugging, CPU-domain twin debugging, NPU on-board debugging, `DumpTensor`, `printf`, Plog, accuracy failures, and address-alignment errors.

Source course: `tutorials/ascendc_operator_development/07_Troubleshooting`.

## Table of Contents

- Debugging sequence
- CPU-domain twin debugging
- NPU on-board debugging
- Logs and Plog
- Accuracy mismatch workflow
- Address-alignment workflow
- Practice: broken Sinh kernels

## Debugging Sequence

Prefer this order:

1. Reproduce with the smallest shape and dtype that fails.
2. Check Host/Kerner contract:
   - dtype, format, shape
   - GM buffer sizes
   - `blockDim`
   - tiling fields
   - workspace size
3. Run CPU-domain twin debugging when possible.
4. If CPU behavior passes but board behavior fails, use NPU-domain `DumpTensor` and `AscendC::printf`.
5. Inspect Plog for runtime, stream, address, and alignment errors.
6. Fix correctness before profiling.

## CPU-Domain Twin Debugging

CPU-domain twin debugging compiles the kernel program with GCC and links the CPU debug library, so kernel logic can be exercised on a non-Ascend CPU environment. It is useful before moving to a real NPU.

Typical CPU debug project:

```text
cpu_debug/
  scripts/
    gen_data.py
    verify_result.py
  CMakeLists.txt
  data_utils.h
  add.cpp
```

### Required Headers

Typical source includes:

- `kernel_operator.h`: Ascend C kernel API.
- `tikicpulib.h`: CPU debug library.
- local `data_utils.h`: file IO helpers.

### Constants to Check

For vector examples:

- `TOTAL_LENGTH`: total element count.
- `USE_CORE_NUM`: simulated core count.
- `BLOCK_LENGTH`: per-core element count.
- `TILE_NUM`: per-core tile count.
- `BUFFER_NUM`: queue depth, often 1 or 2.
- `TILE_LENGTH`: per-tile element count.

These constants must align with tiling and input file sizes.

### CPU Debug APIs and Macros

- `GmAlloc`: allocate shared GM-like memory for CPU validation.
- `GmFree`: free shared memory.
- `AscendC::SetKernelMode`: set AIV/AIC/MIX mode for separated-mode debugging.
- `ICPU_RUN_KF`: CPU debug entry for invoking a kernel function.

Typical steps:

1. Generate `input.bin` and `golden.bin`.
2. Allocate GM memory for inputs, output, workspace, and tiling.
3. Copy generated input into GM memory.
4. Set kernel mode if needed.
5. Call `ICPU_RUN_KF`.
6. Write output to a file.
7. Run `verify_result.py`.

### CPU printf

Kernel-side `printf` works in CPU debug and can be inserted at key locations:

```cpp
AscendC::printf("core=%d tileLength=%d\n", AscendC::GetBlockIdx(), tileLength);
AscendC::PRINTF("x size=%d\n", xLocal.GetSize());
```

Use it to verify:

- core id
- tile length
- tensor size
- offsets
- first few scalar values

## NPU On-Board Debugging

Use NPU on-board debugging when hardware behavior differs from CPU-domain behavior or when checking real GM/UB data.

Two key interfaces:

- `AscendC::DumpTensor`: print tensor data.
- `AscendC::printf` / `AscendC::PRINTF`: print scalar values and text.

### DumpTensor

Function forms include:

```cpp
AscendC::DumpTensor(localTensor, desc, dumpSize);
AscendC::DumpTensor(globalTensor, desc, dumpSize);
AscendC::DumpTensor(localTensor, desc, dumpSize, shapeInfo);
AscendC::DumpTensor(globalTensor, desc, dumpSize, shapeInfo);
```

Parameters:

- `tensor`: `LocalTensor` or `GlobalTensor`.
- `desc`: custom `uint32_t` tag. `__LINE__` is useful.
- `dumpSize`: element count to print.
- `shapeInfo`: optional shape for formatted output.

Example:

```cpp
if (progress == 0) {
    AscendC::DumpTensor(xLocal, __LINE__, 8);
    AscendC::DumpTensor(yLocal, __LINE__, 8);
}
```

Shape-printing practice:

- Print only on a selected core, e.g. logical id 2.
- Print only at selected iteration, e.g. last `Compute`.
- Use shape `(2, 8)` when you want a matrix-like display of 16 positions; if `dumpSize` is 8, the remaining shape slots are shown as unavailable.

### NPU printf

Supported format classes include:

- `%d` / `%i`: signed integers.
- `%u`: unsigned integers.
- `%x`: hexadecimal integers.
- `%f`: float-like values including `float`, `half`, `bfloat16_t`.
- `%s`: string constant or pointer.
- `%p`: pointer.

Example:

```cpp
AscendC::printf(
    "totalLength=%d tileNum=%d blockNum=%lu\n",
    totalLength, tileNum, AscendC::GetBlockNum());
```

Avoid leaving debug prints in performance measurements.

## Logs and Plog

Log format:

```text
[Level] ModuleName(PID,PName):DateTimeMS [FileName:LineNumber]LogContent
```

Common levels: `ERROR`, `WARNING`, `INFO`, `DEBUG`.

Common modules:

- `ASCENDCL`
- `RUNTIME`
- `GE`
- compiler components
- driver user-mode logs

### Log Paths

Ascend EP:

- default: `$HOME/ascend/log/`
- custom: set `ASCEND_PROCESS_LOG_PATH`

Ascend RC:

- default: `/var/log/npu/slog/`

EP layout:

```text
$HOME/ascend/log/
  debug/
    device-<id>/
    plog/
  run/
    device-<id>/
    plog/
  security/
```

Before reproducing a bug, clear the tutorial-local log directory or the relevant Plog directory so only fresh errors remain.

### Error Examples

- `aclrtFree` failure can indicate invalid or repeated device-memory free.
- D-cache / UB bus non-zero response often suggests out-of-bounds or invalid memory access.
- `stream sync timeout` indicates stream work did not finish in the expected time.
- `instruction address misalign(ADDR_MISALIGN)` or `UB address accessed by the VEC instruction is not aligned` indicates alignment constraints are violated.

## Accuracy Mismatch Workflow

Symptom:

- Device init, stream create, kernel launch, and stream sync pass.
- Verification reports output differs from golden data.

Likely causes:

- Kernel compute logic is wrong.
- Tiling fields do not match Kernel assumptions.
- Tensor sizes or queue buffer sizes are wrong.
- Precision loss from scalar loops or repeated `GetValue`.
- Wrong dtype or format assumptions.

Workflow:

1. Print tiling fields with `printf`.
2. Print core id, block length, tile count, tile length.
3. Print `LocalTensor.GetSize()` for input/output tensors.
4. `DumpTensor` selected input, intermediate, and output tensors.
5. Optionally copy an intermediate result to output GM and compare in `verify_result.py`.
6. Reduce shape and tile count until the first wrong element is easy to inspect.

Course example root cause:

- `tileLength` expected 128 float elements, but `xLocal.GetSize()` was 32.
- The TQue buffer allocation size was wrong.

Rule: a compute API's `count` argument must never exceed the valid element count of the LocalTensors supplied.

## Address-Alignment Workflow

Symptom:

- Stream sync fails, commonly around error code `507015`.
- Plog reports `instruction address misalign(ADDR_MISALIGN)` or UB address not aligned.

Likely causes:

- `DataCopy` source or destination address violates 32-byte or API-specific alignment.
- Vector instruction reads a misaligned UB address.
- Offset arithmetic starts from a non-aligned element index for the dtype.

Workflow:

1. Inspect the exact API call in Plog or near the failure.
2. Check Ascend C API documentation for address and length alignment constraints.
3. Print GM/UB offset before the failing `DataCopy` or vector op.
4. Validate offset in bytes, not just elements:
   - `byteOffset = elementOffset * sizeof(dtype)`
5. Use CPU twin debugging if the failing line is hard to isolate on board.
6. Replace illegal direct copy with a compliant padded, aligned, or staged copy strategy.

Course example:

```cpp
AscendC::DataCopy(zGm[progress * tileLength], zLocal[1], tileLength);
```

For `float`, `zLocal[1]` shifts source by 4 bytes, breaking a 32-byte alignment expectation. Fix by copying from an aligned address or by using an API/pattern that supports the desired unaligned tail handling.

## Broken Sinh Practice

The Chapter 7 practice gives `sinh_custom.asc` with multiple intentional bugs. Expected final output:

```text
[Success] Case accuracy is verification passed.
```

Recommended approach:

1. Run once and record whether it is a correctness failure or runtime failure.
2. Clear Plog and rerun.
3. If Plog shows alignment, inspect `DataCopy` offsets.
4. If output differs, instrument:
   - tiling fields
   - `blockLength`, `tileLength`, `tileNum`
   - `GetBlockIdx()`
   - input/output LocalTensor sizes
   - first computed tile
5. Confirm all queue operations are balanced.
6. Confirm `AllocTensor`/`FreeTensor` pairs for every path.

## Debugging Checklist

- Do not profile with `printf` or `DumpTensor` still enabled.
- Do not ignore a passing CPU test if board fails; CPU debug cannot catch every hardware alignment or memory-system issue.
- Treat error code plus Plog text as more useful than the top-level stream-sync failure.
- Print byte offsets for alignment bugs.
- Verify tensor sizes before compute API calls.
- Verify every element is covered exactly once by core/tile splitting.
