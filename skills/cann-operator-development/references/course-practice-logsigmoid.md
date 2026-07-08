# Course Practice: LogSigmoidCustom

Use this reference for the final CANN learning hub vector practice operator `LogSigmoidCustom`.

Source course: `tutorials/ascendc_operator_development/09_course_practice/09.01_vector_ops_practice.ipynb`.

## Goal

Implement an Ascend C vector operator:

```text
LogSigmoid(x) = log(1 / (1 + exp(-x)))
```

OpType:

```text
LogSigmoidCustom
```

Required deliverables:

- operator prototype JSON
- generated operator project
- Host-side tiling implementation
- tiling struct definition
- Kernel implementation
- ACLNN single-operator test
- all provided testcase scripts passing

## Supported Inputs

The practice requires support for:

- `float32`
- `float16` / `half`
- `bfloat16`
- dynamic or generalized shapes represented by total element count

Published testcase shapes:

1. `float32`, shape `(8, 16, 64)`
2. `float32`, shape `(8, 16, 1743)`
3. `float16`, shape `(4, 2028)`
4. `float16`, shape `(32, 1001, 7763)`
5. `bfloat16`, shape `(1, 1024)`
6. `bfloat16`, shape `(64, 64, 1024)`

These cases intentionally cover:

- aligned and non-aligned tails
- small and very large tensors
- multiple dtype branches
- generalized total-length tiling rather than fixed-shape code

## Numerics

Basic formula:

```text
log(1 / (1 + exp(-x)))
```

Equivalent form:

```text
-log(1 + exp(-x))
```

For a learning-hub operator, use available Ascend C vector APIs directly and validate against the testcase golden output. For production-grade numerical stability, consider the stable branch form:

```text
if x >= 0: -log1p(exp(-x))
else:      x - log1p(exp(x))
```

Only use the stable branch if the available Ascend C API set and dtype conversion path support it cleanly; otherwise match the course/test expected implementation.

## Host-Side Tiling

The operator should be generalized by total element count:

1. Read input shape and dtype from tiling context.
2. Compute `totalLength` from all dimensions.
3. Choose `blockDim` based on available core count and `totalLength`.
4. Split work evenly across cores.
5. Choose per-core `tileNum`, `tileLength`, and tail fields.
6. Pass dtype/template key if Kernel has dtype-specialized branches.

Recommended tiling fields:

```cpp
struct LogSigmoidCustomTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t formerNum;
    uint32_t tailNum;
};
```

Adapt names to the generated template, but preserve the information:

- total number of elements
- per-core number of elements
- number of full tiles per core
- full tile length
- tail tile length
- tail-core handling when total length is not divisible by block count

For non-aligned cases such as `(8, 16, 1743)` and `(32, 1001, 7763)`, tail logic is mandatory.

## Kernel-Side Pattern

Use the standard vector operator structure:

- `Init`
- `Process`
- `CopyIn`
- `Compute`
- `CopyOut`

Buffers:

- input queue for `x`
- output queue for `y`
- temporary buffers for `-x`, `exp(-x)`, `1 + exp(-x)`, reciprocal or division, and log result depending on API sequence

A direct operation sequence:

```text
tmp0 = -x
tmp1 = exp(tmp0)
tmp2 = tmp1 + 1
tmp3 = 1 / tmp2
y    = log(tmp3)
```

or:

```text
tmp0 = -x
tmp1 = exp(tmp0)
tmp2 = tmp1 + 1
tmp3 = log(tmp2)
y    = -tmp3
```

Prefer the second form if it reduces one division operation and matches test tolerance:

```text
y = -log(1 + exp(-x))
```

Potential Ascend C vector APIs/operators to look for:

- `Muls` for multiply by `-1`
- `Exp`
- `Adds` or `Add` with a tensor/scalar-one pattern
- `Ln` / log API if available in the target CANN version
- `Div` if using reciprocal formula

## Dtype Handling

For `float32`:

- compute in float where possible.

For `half`:

- ensure APIs support half inputs for `Exp` and log/Ln path.
- compare with tolerance suitable for half.

For `bfloat16`:

- confirm template definitions include bfloat16 dtype.
- ensure temporary dtype and API support are valid.
- if an API requires float intermediates, cast or use a supported higher-precision temporary path when the framework allows it.

Use TilingKey/templates to dispatch dtype-specific Kernel variants if the generated project expects dtype macros such as `DTYPE_X` / `DTYPE_Y`.

## Tail and Alignment Handling

Every testcase must cover all elements exactly once:

- no missing final elements
- no duplicate writes
- no out-of-bounds `DataCopy`
- no vector compute count larger than the valid local tensor size

For tails:

- use `lastTileLength` for the final tile in each core when it is shorter.
- use `DataCopyPad` when GM/UB copy alignment requires padding.
- keep compute count equal to valid element count, not padded element count, unless the output padding is safely discarded.

## Test Workflow

After implementation:

```bash
bash build.sh
./build_out/custom_opp*.run --install-path=${HOME}/
```

Then run all testcase scripts:

```bash
bash src/09.01_testcase/testcase_1/run.sh
bash src/09.01_testcase/testcase_2/run.sh
bash src/09.01_testcase/testcase_3/run.sh
bash src/09.01_testcase/testcase_4/run.sh
bash src/09.01_testcase/testcase_5/run.sh
bash src/09.01_testcase/testcase_6/run.sh
```

For each failing case, inspect:

- dtype branch selected by TilingKey
- total element count from shape
- blockDim
- per-core start offset
- last tile length
- API support for dtype
- golden tolerance

## Common Mistakes

- Hardcoding `(8, 16, 64)` and failing non-aligned shapes.
- Using a compute count based on padded tile length for a real tail.
- Forgetting bfloat16 template support.
- Using scalar loops for the whole tensor instead of vector APIs.
- Allocating too few temporary buffers for the chosen formula.
- Forgetting to update ACLNN test code after changing OpType or generated API names.
- Passing one testcase and not running all six.
