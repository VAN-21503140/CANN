# Fused Operator Development

Use this reference for Ascend C fused operators, especially VV fusion and CV fusion from the CANN learning hub Chapter 5.

Source course: `tutorials/ascendc_operator_development/05_fused_operator_development`.

## Table of Contents

- Why fuse operators
- VV fusion pattern
- CV fusion pattern
- Host-side considerations
- Kernel-side considerations
- Practice specs
- Verification checklist

## Why Fuse Operators

Fusion combines multiple adjacent operators into one Kernel entry. It improves performance mainly by:

- Reducing operator scheduling overhead.
- Removing intermediate Global Memory round trips.
- Reusing LocalTensor / UB data between stages.
- Pipelining different compute units, especially Cube and Vector.
- Simplifying framework-level graphs when a sequence is stable and performance-critical.

Fusion does not magically reduce arithmetic cost. For pure Vector-Vector fusion, computation time may remain similar, but data movement and launch overhead decrease. For Cube-Vector fusion, the largest win is often overlapping Cube and Vector stages and avoiding a full `GM -> Local -> GM -> Local` intermediate path.

## VV Fusion Pattern

VV fusion means multiple Vector operators execute in one fused Vector kernel.

Course example:

```text
z = (x + y) * (x - y)
```

This fuses Add, Sub, and Mul. Before fusion, the sequence has roughly:

1. Add CopyIn
2. Add compute
3. Add CopyOut
4. Sub CopyIn
5. Sub compute
6. Sub CopyOut
7. Mul CopyIn
8. Mul compute
9. Mul CopyOut

After fusion:

1. CopyIn x and y once.
2. Compute Add into a LocalTensor.
3. Compute Sub into a LocalTensor.
4. Compute Mul from the intermediate LocalTensors.
5. CopyOut z once.

The core benefit is removing Add/Sub intermediate CopyOut and Mul/Sub repeated CopyIn.

### VV Host Pattern

For a fixed-shape teaching example such as shape `(8, 2048)`:

- IR file contains only fused inputs `x`, `y` and output `z`.
- OpType can be `SquareDiff`.
- Support dtypes should match the original fused operators, e.g. `half` and `float` with `ND`.
- Tiling can be similar to an Add operator:
  - `totalLength`
  - `tileNum`
  - optionally `blockLength`, `tileLength`, and tail fields for generalized shapes.

### VV Kernel Pattern

Keep one class with `Init`, `Process`, `CopyIn`, `Compute`, `CopyOut`.

Compute usually allocates one or two extra local tensors for intermediates:

```cpp
auto xLocal = inQueueX.DeQue<DTYPE_X>();
auto yLocal = inQueueY.DeQue<DTYPE_Y>();
auto addLocal = tmpBufAdd.Get<DTYPE_Z>();
auto subLocal = tmpBufSub.Get<DTYPE_Z>();
auto zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

AscendC::Add(addLocal, xLocal, yLocal, tileLength);
AscendC::Sub(subLocal, xLocal, yLocal, tileLength);
AscendC::Mul(zLocal, addLocal, subLocal, tileLength);
```

Balance queues and buffers:

- Every `AllocTensor` must be paired with `FreeTensor`.
- Every `EnQue` must be paired with `DeQue`.
- Temporary intermediates can use `TBuf` when they do not need queue semantics.

## CV Fusion Pattern

CV fusion means combining a Cube operator and a Vector operator. Course examples:

- `Matmul + LeakyReLU` as `MatmulLeakyreluCustom`.
- Practice: `Matmul + Abs`.
- Chapter practice: `Matmul + Sinh`.

CV fusion transforms a serial flow:

```text
Matmul CopyIn -> Matmul compute -> Matmul CopyOut to GM
Vector CopyIn from GM -> Vector compute -> Vector CopyOut
```

into a tiled pipeline:

```text
Cube input tile -> Cube compute -> result tile in local path
Vector reads tile -> Vector post-process -> CopyOut final tile
```

The goal is to let a Vector stage process one Matmul output tile while Cube starts or continues another tile, and to avoid materializing the whole Matmul output in GM before Vector work begins.

## CV Host-Side Considerations

CV fusion has stricter tiling requirements than VV fusion:

- Matmul output fragment size must fit the Cube output path and the Vector input path.
- If Vector reads a Matmul tile from local memory, `baseM * baseN * sizeof(output_dtype)` must fit UB or the selected local staging buffer.
- If using separated AIC/AIV modes, configure Cube and Vector participation intentionally.

Course note for Atlas A2-style separated mode:

- Matmul APIs may be launched from AIV side while AIC does Cube compute.
- `SetBlockDim` configures the AI Core task count.
- `SetDim` configures the AIV participation count.
- For an A2 single AI Core with a 1:2 Cube-to-Vector relation, the example sets `TCubeTiling.SetDim(2)` and `TilingContext.SetBlockDim(1)` for a single-core demonstration.

Host tiling fields for CV often include:

- Matmul `TCubeTiling`.
- Vector parameters such as `alpha` for LeakyReLU.
- Matrix shape fields M, N, K when Kernel needs offset calculation.
- Local memory or UB size fields.

## CV Kernel-Side Pattern

Implement a class with:

- `CalcOffset`: map core id to A/B/C tile offsets.
- `Init`: set `GlobalTensor` views for A, B, Bias, and C.
- `MatmulCompute`: set A/B/Bias, call `Iterate` or `IterateAll`, and get C tile.
- `VectorCompute`: apply post-op such as LeakyReLU, Abs, or Sinh.
- `CopyOut`: write final submatrix tile back to GM.

Offset mapping for M/N split:

```text
left block index  = coreId % numMBlocks
right block index = coreId / numMBlocks
A offset = left block index * singleCoreM * K
B offset = right block index * singleCoreN
C offset = left row start * full N + right column start
```

When output is a submatrix tile, CopyOut is not always a single contiguous copy:

- Each row of a `baseM * baseN` tile is contiguous.
- The gap between rows in GM is `full_N - baseN`.
- Copy row by row or use a supported strided/padded copy pattern. Do not blindly copy the tile as one continuous block unless the tile spans full row width.

## Practice Specs

### VV `SquareDiff`

```text
z = (x + y) * (x - y)
x, y: shape (8, 2048), dtype half/float, ND
z: same shape and dtype family
```

Key checks:

- Fused OpType and generated ACLNN names match.
- Add/Sub/Mul work on equal tile lengths.
- Temporary buffers are large enough for intermediates.

### CV `MatmulLeakyreluCustom`

```text
C = LeakyReLU(A * B + Bias, alpha)
A: [1024, 256], half, ND
B: [256, 640], half, ND
Bias: [640], float, ND
C: [1024, 640], float, ND
```

Expected teaching test with A=1, B=2, Bias=0.5 gives positive outputs near `512.5`.

### CV `MatmulAbs`

```text
C = abs(A * B)
A: [1024, 256], half, ND
B: [256, 640], half, ND
C: [1024, 640], float, ND
```

The course's sample output around `511.5` reflects its concrete test data and bias/setup. Always recompute expected values from the actual test program.

### CV `MatmulSinh`

```text
C = sinh(A * B + Bias)
A: [M, K], float, ND
B: [K, N], float, ND
Bias: [N], float, ND
C: [M, N], float, ND
Practice shape: M=1024, N=2048, K=512
```

Key implementation points:

- Use Matmul high-level API for Cube stage.
- Apply `Sinh` or its API-supported equivalent on the Matmul tile.
- Keep Vector tile and Matmul output tile dimensions consistent.

## ACLNN Naming Pattern

Generated two-stage ACLNN APIs usually follow:

- `aclnnSquareDiffGetWorkspaceSize` / `aclnnSquareDiff`
- `aclnnMatmulLeakyreluCustomGetWorkspaceSize` / `aclnnMatmulLeakyreluCustom`
- `aclnnMatmulAbsGetWorkspaceSize` / `aclnnMatmulAbs`
- `aclnnMatmulSinhGetWorkspaceSize` / `aclnnMatmulSinh`

Always confirm generated headers under the installed custom vendor package.

## Verification Checklist

- Confirm fusion preserves math order and dtype behavior.
- Compare fused output against unfused golden output for representative shapes and dtypes.
- Confirm intermediate LocalTensor lifetimes do not conflict with queues.
- For CV, prove the Matmul output tile fits the Vector post-op staging buffer.
- For submatrix CopyOut, verify row stride and GM offsets.
- Use msProf before and after fusion to prove scheduling, CopyIn/CopyOut, or pipeline gains.

## Common Mistakes

- Expecting VV fusion to reduce vector arithmetic time; its main gain is launch and memory traffic.
- Writing CV output tile as a contiguous buffer even when it occupies a submatrix.
- Forgetting that Matmul high-level API may use separated Cube/Vector execution semantics.
- Reusing Add tiling unchanged for a fused op where intermediate tensors require extra UB.
- Optimizing fusion before checking the unfused golden output and dtype tolerance.
