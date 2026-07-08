# CANN Open-Source Operator Repository Workflows

Use this reference for CANN open-source operator repositories such as `ops-math`, contribution-style operator development, UT/ST, and migration from single-operator projects.

Source course: `tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution`.

## Table of Contents

- Repository family and layout
- Package and validation modes
- Creating an operator with `build.sh --genop`
- Required deliverables
- TilingKey and Tiling data
- UT patterns
- ST patterns
- Migration from single-operator projects
- Practice: Sigmoid in `ops-math`

## Repository Family and Layout

CANN open-source operator repositories expose optimized CANN operator implementations for learning, customization, and contribution.

Core repositories:

- `ops-nn`: neural-network operator library, including Matmul and activation classes.
- `ops-math`: math/conversion/random foundational operator library.
- `ops-transformer`: transformer and large-model operator library, including attention/MoE classes.
- `ops-cv`: computer-vision operator library.

Course practice focuses on `ops-math`.

Important `ops-math` directories:

- `examples/`: standard single-operator examples and API-call validation.
- `math/`: built-in math operators.
- `random/`: built-in random operators.
- `conversion/`: built-in dtype/format/layout conversion operators.
- `experimental/`: user-contributed operators. Contributions are organized under `experimental/math`, `experimental/random`, or `experimental/conversion`.

For external contribution practice, place new math operators under:

```text
ops-math/experimental/math/<op_name>/
```

## Package and Validation Modes

The build system supports several artifacts:

- Custom operator package: a selected subset packaged under a vendor name and mounted onto CANN without replacing the base package.
- Full `ops-math` package: the whole project as a replacement-style package.
- Static library: full project as a static library plus ACLNN headers, mainly for ACLNN AI Core calls.

For tutorials, prefer custom operator packages. Avoid accidentally building a full replacement package.

Package command pattern:

```bash
bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} -j16
```

For experimental operators:

```bash
bash build.sh --pkg --experimental --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} -j16
```

Important parameters:

- `--soc`: product target. Course examples mention `ascend910b`, `ascend910_93`, and `ascend950`.
- `--vendor_name`: custom vendor package name; default is often `custom`.
- `--ops`: comma-separated lowercase op names.
- `--experimental`: include user operators under `experimental`.

Safety rule: pass at least one of `--ops` or `--vendor_name` for tutorial custom-package workflows. Otherwise the command may build a full `ops-math` package instead of a custom operator package.

Run example / ST pattern:

```bash
bash build.sh --experimental --run_example <op_name> eager cust --vendor_name=<vendor_name>
```

`eager` is the ACLNN single-operator call mode. Graph mode requires extra graph deliverables.

## Creating an Operator With `build.sh --genop`

Generate a skeleton:

```bash
bash build.sh --genop=${op_class}/${op_name}
```

Rules:

- `op_class` is required. For external math contribution, use `experimental/math`.
- `op_name` uses lowercase snake_case, e.g. `add_custom`, `sigmoid`, `sub_custom`.
- The generated skeleton splits responsibilities more than a single-operator `msopgen` project.

Example:

```bash
bash build.sh --genop=experimental/math/add_custom
```

## Required Deliverables

Typical contribution tree:

```text
<op_name>/
  CMakeLists.txt
  examples/
    test_aclnn_<op_name>.cpp
  op_graph/
    <op_name>_graph_infer.cpp      # optional graph dtype inference
    <op_name>_proto.h              # optional graph prototype
  op_host/
    <op_name>_def.cpp              # required operator definition
    <op_name>_infershape.cpp       # optional but recommended
    <op_name>_tiling.cpp           # required host tiling
  op_kernel/
    <op_name>.cpp                  # required kernel entry
    <op_name>.h                    # required algorithm/class implementation
    <op_name>_tiling_data.h        # required tiling struct
    <op_name>_tiling_key.h         # tiling template / key
  tests/
    ut/
      op_host/
        test_<op_name>_tiling.cpp
        test_<op_name>_infershape.cpp
      op_kernel/
        CMakeLists.txt
        test_<op_name>.cpp
        <op_name>_data/
          gen_data.py
          compare_data.py
```

Required core deliverables are usually:

- `op_host/<op_name>_def.cpp`
- `op_host/<op_name>_tiling.cpp`
- `op_kernel/<op_name>.cpp`
- `op_kernel/<op_name>.h`
- `op_kernel/<op_name>_tiling_data.h`
- `examples/test_aclnn_<op_name>.cpp` for ST
- UT files before contribution-quality completion

## Operator Definition

`<op_name>_def.cpp` describes the operator:

- OpType and implementation name.
- Inputs and outputs.
- Supported dtype and format.
- Attribute metadata if present.
- Implementation binding.

Open-source repo definition logic is conceptually similar to a single-operator JSON prototype, but split into C++ deliverables for repository integration.

## TilingKey and Tiling Data

### TilingKey

`<op_name>_tiling_key.h` defines template parameters used to generate and validate TilingKey values.

The course emphasizes:

- TilingKey is similar in purpose to C++ template specialization.
- It avoids unnecessary instruction-cache misses and scalar overhead by selecting a concrete Kernel variant.
- Keep simple operators simple. If only one schedule exists, define only the needed schedule mode.

Common macros/interfaces:

- `ASCENDC_TPL_ARGS_DECL`
- `ASCENDC_TPL_DATATYPE_DECL`
- `ASCENDC_TPL_UINT_DECL`
- `ASCENDC_TPL_BOOL_DECL`
- `ASCENDC_TPL_SEL`
- `ASCENDC_TPL_ARGS_SEL`
- `ASCENDC_TPL_UINT_SEL`
- `GET_TPL_TILING_KEY`

### Tiling Data

`<op_name>_tiling_data.h` uses a plain struct to pass Host-computed fields to Kernel:

```cpp
struct AddCustomTilingData {
    int64_t totalNum = 0;
    int64_t tileNum = 0;
    int64_t tileLength = 0;
};
```

Field names and types must match Host assignment and Kernel reading. Do not rename or reorder fields casually when UT expected tiling strings depend on field order.

### Tiling Implementation

`<op_name>_tiling.cpp` should:

- Include repository template headers rather than manually copying single-operator include paths.
- Read shape, dtype, format, attributes, platform info, core count, and UB size from context.
- Fill tiling data and set `blockDim`.
- Set TilingKey if templates are used.
- Bind through repository macros such as `IMPL_OP_OPTILING(OpType).Tiling(TilingFunc)` or the current repo equivalent.

## Kernel Deliverables

The open-source repo often splits a single-operator kernel file into:

- `<op_name>.h`: class implementation, includes, template definitions, helper functions.
- `<op_name>.cpp`: kernel entry / instantiation.

Differences from single-operator projects:

- Include `kernel_tiling/kernel_tiling.h` and local `<op_name>_tiling_data.h`.
- Keep TilingKey include local as `<op_name>_tiling_key.h`.
- Keep algorithm code reusable from `.h` and thin entry in `.cpp`.

## UT Patterns

UT validates internal logic before full ST.

### Tiling UT

Location:

```text
tests/ut/op_host/test_<op_name>_tiling.cpp
```

Include:

- `iostream`
- `gtest/gtest.h`
- `tiling_context_faker.h`
- `tiling_case_executor.h`

Test steps:

1. Construct `gert::TilingContextPara` with input shapes, runtime shapes, dtypes, formats, outputs, attributes, compile info, available core count, and UB size.
2. Define expected TilingData string in struct-field order.
3. Define expected TilingKey and workspace if relevant.
4. Call `ExecuteTestCase(...)`.

Tiling UT catches incorrect core counts, tile lengths, unsupported shapes, wrong dtype branches, and missing attributes.

### Kernel UT

Location:

```text
tests/ut/op_kernel/test_<op_name>.cpp
```

Typical flow:

1. Allocate GM-like memory with AscendC test helpers.
2. Generate or read input/golden data.
3. Construct tiling data manually or by calling Host tiling logic.
4. Set tiling key, e.g. `ICPU_SET_TILING_KEY(tilingKey)`.
5. Set kernel mode such as `AscendC::SetKernelMode(KernelMode::AIV_MODE)`.
6. Invoke kernel with `ICPU_RUN_KF(<kernel_name>, blockDim, x, y, z, workspace, tiling)`.
7. Compare actual and golden output.
8. Free all GM buffers.

Data helpers:

- Put scripts under `<op_name>_data/`.
- `gen_data.py` generates binary inputs and golden outputs.
- `compare_data.py` validates actual output against golden output with dtype-aware tolerance.

Kernel UT `CMakeLists.txt` often calls an `AddOpTestCase`-style helper and passes dtype compile defines when Kernel uses `DTYPE_*` macros.

### Infershape UT

Location:

```text
tests/ut/op_host/test_<op_name>_infershape.cpp
```

Use when shape inference is implemented. Build an `InfershapeContextPara`, define expected output shape(s), then call `ExecuteTestCase`.

## ST Pattern

ST is end-to-end validation. Single-operator API ST usually lives in:

```text
examples/test_aclnn_<op_name>.cpp
```

Flow:

1. Build and install the custom operator package.
2. Source vendor environment, for example `vendors/<vendor>/bin/set_env.bash`.
3. Run the example through:

```bash
bash build.sh --experimental --run_example <op_name> eager cust --vendor_name=<vendor_name>
```

ACLNN call flow:

- `aclnnXxxGetWorkspaceSize` to obtain workspace size.
- Allocate workspace.
- `aclnnXxx` to run the operator.
- Synchronize stream.
- Compare results.

Other ST forms include single-operator model execution, IR graph construction, PyTorch plugin calls, TensorFlow plugin calls, and pybind calls. Use them only when the contribution target requires that path.

## Migration From Single-Operator Projects

When migrating from an `msopgen` single-operator project:

- Keep core math and tiling logic.
- Split Host code:
  - definition into `<op_name>_def.cpp`
  - tiling into `<op_name>_tiling.cpp`
  - optional infershape into `<op_name>_infershape.cpp`
- Split Kernel code:
  - reusable class and helper code into `<op_name>.h`
  - kernel entry and explicit instantiation into `<op_name>.cpp`
- Rename tiling headers:
  - single project `xxx_tiling.h` -> repo `xxx_tiling_data.h`
  - single project `tiling_key_xxx.h` -> repo `xxx_tiling_key.h`
- Replace single-project registration calls with repo registration macros.
- Add `examples/` ST and `tests/ut/` UT deliverables.
- Use repository include paths and build scripts; do not carry over brittle relative includes from the single project.

## Practice: Add/Sub and Sigmoid

### Add/Sub Custom

Course AddCustom specification:

- Add math: `z = x + y`.
- Sub practice: `y = x1 - x2`.
- Shape: `(8, 2048)`.
- Dtype: `float` or `int32` depending on practice.
- Format: `ND`.
- Kernel name examples: `add_custom`, `sub_custom`.

Required APIs:

- Data movement: `DataCopy`.
- Vector compute: `Add` or `Sub`.
- Local memory: `AllocTensor`, `FreeTensor`.
- Queue sync: `EnQue`, `DeQue`.

### Sigmoid Practice

Course final open-source repo practice:

```text
sigmoid(x) = 1 / (1 + exp(-x))
OpType: Sigmoid
Input x: [8, 2048], float/int32 listed in notebook, but required implementation focuses on float32 UT.
Output y: same shape as x
```

Expected work:

- Create `experimental/math/sigmoid`.
- Implement `sigmoid_def.cpp`.
- Implement `sigmoid_infershape.cpp`.
- Implement `sigmoid_tiling.cpp`.
- Define `sigmoid_tiling_data.h`.
- Define `sigmoid_tiling_key.h`.
- Implement `sigmoid.h` and `sigmoid.cpp`.
- Write Infershape UT, Tiling UT, Kernel UT.
- Provide data generation and comparison scripts.

## Review Checklist

- Operator name is snake_case in paths but OpType casing matches registration.
- `--experimental` is present when building/running experimental operators.
- `--ops` and `--vendor_name` avoid accidental full package replacement.
- All generated ACLNN names match headers after install.
- UT covers tiling, kernel data correctness, and infershape when present.
- ST runs through `examples/test_aclnn_<op_name>.cpp`.
- TilingKey has a valid mapping for every Kernel branch compiled.
- Migration does not leave single-project-only includes or registration calls.
