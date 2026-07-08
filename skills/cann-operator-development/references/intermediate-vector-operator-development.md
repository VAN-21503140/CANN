# Intermediate Vector Operator Development

Use this reference for CANN / Ascend C intermediate vector operator work from `cann-learning-hub/tutorials/ascendc_operator_development/03_intermediate_vector_operator_development`.

Chinese search anchors: `Vector算子开发`, `工程化算子`, `单算子API`, `pybind调用`, `泛化Tiling`, `tiling设计`, `通用算子`, `对齐原则`, `多核均衡`, `访存优化`, `TBuf`, `workspace`, `属性`, `TilingKey`, `模板化编程`, `章节实践`, `SigmoidCustom`.

## Source Inventory

Primary CANNLab / JupyterLab source:

- Outer CANNLab URL: `https://ai.gitcode.com/user/weixin_61718454/notebookcann/lab?cannNotebookId=86192c9691b54fb1bdbb0ddccae3daa4-notebook22-5685`
- JupyterLab path: `cann-learning-hub/tutorials/ascendc_operator_development/03_intermediate_vector_operator_development`

Files observed in the directory:

```text
03.01_chapter_intro.ipynb
03.02_operator_engineering_intro.ipynb
03.03_acl_pybind_call.ipynb
03.04_generalized_tiling_design.ipynb
03.05_tiling_template_attr_tbuf_workspace.ipynb
03.06_chapter_practice.ipynb
README.md
src/
answer/
images/
```

The chapter teaches vector operator engineering, operator invocation, generalized Tiling for arbitrary input size, Tiling template programming, attributes, `TBuf`, `workspace`, and an integrated `SigmoidCustom` practice.

## How to Route This Reference

Read this reference when the user asks about:

- Intermediate Ascend C vector operators.
- Turning simple kernels into standard operator projects.
- `msopgen` project generation.
- Host-side `OpDef`, `InferShape`, `InferDataType`, and `TilingFunc`.
- Kernel-side engineering signatures with `workspace` and `tiling`.
- ACLNN single-operator API calls.
- pybind11 / `torch_npu` custom operator calls.
- Generalized Tiling / 泛化Tiling / tiling设计.
- Dynamic shape or arbitrary input size support.
- `workspace`, `TBuf`, attributes, `TilingKey`, or Tiling template programming.
- Building training problems from the CANN learning hub chapter.

For the deepest standalone generalized Tiling formulas and big-core/small-core pattern, also read `generalized-tiling-strategy.md`.

For lower-level sample-derived queue/copy patterns, also read `ascendc-samples-patterns.md`.

## Chapter 03.01: Chapter Introduction

Chapter 3 focuses on vector operator Tiling development, operator call methods in networks/applications, and generalized operator development.

Prerequisites:

- Understand operator concepts and basic principles from Chapter 1.
- Understand basic Ascend C operator development from Chapter 2.
- Have an Ascend NPU, Ascend cloud environment, or simulation environment.
- Have CANN installed for the target hardware.

Learning goals:

- Understand the basic principle of Tiling.
- Master single-operator API calls and pybind calls.
- Design and implement Tiling for arbitrary input sizes.

Chapter sequence:

1. `03.01_chapter_intro.ipynb`: chapter map and prerequisites.
2. `03.02_operator_engineering_intro.ipynb`: standard operator project engineering.
3. `03.03_acl_pybind_call.ipynb`: ACLNN and pybind invocation.
4. `03.04_generalized_tiling_design.ipynb`: generalized Tiling design.
5. `03.05_tiling_template_attr_tbuf_workspace.ipynb`: Tiling templates, attributes, `TBuf`, `workspace`.
6. `03.06_chapter_practice.ipynb`: integrated `SigmoidCustom` practice.

## Chapter 03.02: Operator Engineering Introduction

### Purpose

This notebook teaches the standard Ascend C operator project workflow. It moves beyond writing only a kernel function and shows a deployable operator package with Host-side registration, shape/type inference, Tiling, Kernel implementation, build, deployment, and ACLNN testing.

### Engineering Operator Project Concept

An Ascend C single-operator project includes:

- Host-side operator registration.
- Host-side shape inference.
- Host-side dtype inference.
- Host-side Tiling.
- Task dispatch and memory planning.
- Kernel-side Ascend C compute implementation.
- Build configuration and packaging.

The `msopgen` tool generates the standard project structure from an operator prototype JSON.

### Environment Initialization Pattern

The notebooks initialize the Jupyter environment by sourcing CANN environment variables into Python:

```python
import os, subprocess
env = subprocess.check_output(
    "bash -l -c 'source $ASCEND_TOOLKIT_HOME/set_env.sh && env'",
    shell=True,
    text=True,
)
for line in env.splitlines():
    if "=" in line:
        os.environ.__setitem__(*line.split("=", 1))
```

Use this pattern only in notebooks. In normal shell scripts, source `set_env.sh` directly.

### Operator Prototype JSON

The example operator is `AddCustomTemplate`.

It has:

- inputs: `x`, `y`
- output: `z`
- formats: `ND`
- dtypes: `float16`, `float`

Prototype pattern:

```json
[
  {
    "op": "AddCustomTemplate",
    "input_desc": [
      {
        "name": "x",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
      },
      {
        "name": "y",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
      }
    ],
    "output_desc": [
      {
        "name": "z",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
      }
    ]
  }
]
```

### Project Generation

`msopgen` is included in CANN 9.0.0 in this course environment.

Command pattern:

```bash
msopgen gen \
  -i Sources/03.02/add_custom.json \
  -c ai_core-ascend910b1 \
  -lan cpp \
  -out Sources/03.02/custom_op
```

Parameter meaning:

- `-i`: operator prototype JSON path.
- `-c`: target Ascend AI processor / SoC.
- `-lan cpp`: Ascend C / C++ operator implementation.
- `-out`: output project directory.

Generated project structure:

```text
custom_op/
  framework/
  op_host/
    add_custom_template.cpp
    CMakeLists.txt
  op_kernel/
    add_custom_template_tiling.h
    add_custom_template.cpp
    CMakeLists.txt
  CMakeLists.txt
  CMakePresets.json
  build.sh
```

Important generated files:

- `op_host/add_custom_template.cpp`: operator prototype registration, Tiling, shape inference, dtype inference.
- `op_kernel/add_custom_template_tiling.h`: Tiling data structure.
- `op_kernel/add_custom_template.cpp`: Kernel-side implementation.
- `CMakePresets.json`: CANN package path and build config.
- `build.sh`: build script.

### Host-Side Responsibilities

Host-side code in a standard operator project has three core responsibilities.

#### Operator Prototype Registration

`OpDef` registration describes:

- input names, required/optional status, dtype, format
- output names, dtype, format
- shape inference function
- dtype inference function
- Tiling function
- supported hardware config

Pattern:

```cpp
namespace ops {
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}
```

#### Shape Inference

For non-broadcast Add, output shape equals input shape:

```cpp
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}
```

#### Dtype Inference

For same-type Add, output dtype equals input dtype:

```cpp
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
```

#### Basic Tiling Function

Tiling means splitting data into chunks because AI Core local memory cannot always hold all inputs, outputs, and intermediates at once.

Simple Tiling structure:

```cpp
struct TilingDataTemplate {
    uint32_t totalLength;
    uint32_t tileNum;
};
```

Simple Host-side Tiling:

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    TilingDataTemplate *tiling = context->GetTilingData<TilingDataTemplate>();
    tiling->totalLength = totalLength;
    tiling->tileNum = 1;
    return ge::GRAPH_SUCCESS;
}
```

This basic example is instructional only. It is not a robust arbitrary-shape Tiling strategy because it hardcodes `blockDim` and `tileNum`.

### Kernel-Side Engineering Differences

A standard operator project kernel receives `workspace` and `tiling` in addition to normal inputs/outputs:

```cpp
__global__ __aicore__ void add_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
```

Kernel code must read Tiling data:

```cpp
REGISTER_TILING_DEFAULT(TilingDataTemplate);
GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
```

Then pass Tiling fields into the operator class:

```cpp
KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
op.Process();
```

Dynamic Init pattern:

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                            uint32_t totalLength, uint32_t tileNum)
{
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

    xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
}
```

Process loop:

```cpp
__aicore__ inline void Process()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

### Build, Deploy, and Test

Build operator package:

```bash
cd Sources/03.02/custom_op
bash build.sh
```

Install package:

```bash
./build_out/custom_opp*.run --install-path=${HOME}/
```

Compile ACLNN test:

```bash
g++ \
  -I$ASCEND_TOOLKIT_HOME/include \
  -I${HOME}/vendors/customize/op_api/include \
  -L$ASCEND_TOOLKIT_HOME/lib64 \
  -L${HOME}/vendors/customize/op_api/lib \
  Sources/03.02/test/main.cpp \
  -lcust_opapi -lnnopbase -lacl_rt \
  -o Sources/03.02/execute_add_op
```

Run with custom operator environment:

```bash
source ${HOME}/vendors/customize/bin/set_env.bash
./Sources/03.02/execute_add_op
```

Expected example output:

```text
result is:
3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
test pass
```

### 03.02 Practice Extraction

Practice task: create and implement `SubCustomTemplate` from `AddCustomTemplate`.

Training value:

- Forces the model to update JSON `op` name and generated filenames.
- Tests Host-side registration, shape inference, dtype inference, and Tiling.
- Tests Kernel implementation transfer from Add to Sub.
- Tests build/deploy/call loop.

Common pitfalls:

- Forgetting to rename `op` to `SubCustomTemplate`.
- Using generated file names that do not match the operator name.
- Leaving `context->SetBlockDim(8)` with shapes where 8 cores are not valid.
- Stubbing Kernel code and still claiming the operator works.

## Chapter 03.03: ACLNN and pybind Invocation

### Purpose

This notebook teaches how to invoke a compiled Ascend C operator through:

- ACLNN single-operator API.
- pybind11 / PyTorch NPU extension.

### Operator Call Modes

Common Ascend C operator call modes:

| Mode | Requirement |
| --- | --- |
| Direct kernel launch | Kernel launch validation / quick mode |
| ACLNN single-operator API | Operator project compiled and installed |
| Single-operator model execution | Graph inclusion operator development and installed package |
| IR graph construction | Graph inclusion operator development and installed package |
| PyTorch framework call | Framework adapter/plugin and installed operator package |
| TensorFlow framework call | Framework adapter/plugin and installed operator package |
| pybind call | Operator project compiled and installed |

Chapter 03.03 focuses on ACLNN single-operator API and pybind.

### ACLNN Generated API Layout

After a custom operator package is installed, ACLNN headers and libraries appear under the vendor directory, for example:

```text
${HOME}/vendors/customize/op_api/include/aclnn_add_custom_template.h
${HOME}/vendors/customize/op_api/lib/libcust_opapi.so
```

Header naming pattern:

```text
aclnn_<snake_case_op_name>.h
```

For `AddCustomTemplate`, the header is:

```text
aclnn_add_custom_template.h
```

### ACLNN Two-Stage API Pattern

Generated ACLNN APIs generally use a two-stage pattern:

```cpp
aclnnStatus aclnnXxxGetWorkspaceSize(
    const aclTensor *input0,
    ...,
    aclTensor *output0,
    ...,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnXxx(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

For `AddCustomTemplate`:

```cpp
aclnnStatus aclnnAddCustomTemplateGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *z,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnAddCustomTemplate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### ACLNN Call Flow

1. Include ACL and generated ACLNN headers.
2. Initialize ACL with `aclnnInit` or `aclInit` depending on the example style.
3. Set device with `aclrtSetDevice`.
4. Create stream with `aclrtCreateStream`.
5. Allocate device memory using `aclrtMalloc`.
6. Create `aclTensor` descriptors with `aclCreateTensor`.
7. Copy input host data to device with `aclrtMemcpy`.
8. Call `aclnnXxxGetWorkspaceSize`.
9. Allocate workspace if `workspaceSize > 0`.
10. Call `aclnnXxx`.
11. Synchronize stream.
12. Copy output device data to host.
13. Destroy tensors and free device memory.
14. Destroy stream, reset device, and finalize ACL/ACLNN.

### ACLNN Test Skeleton

Use robust error checking:

```cpp
#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
        }                                                                                               \
    } while (0)
```

Create tensors:

```cpp
const std::vector<int64_t> shape = {8, 2048};
const int64_t elementCount = shape[0] * shape[1];
const size_t bufferSize = elementCount * sizeof(aclFloat16);

void* input0DeviceMem = nullptr;
CHECK_ACL(aclrtMalloc(&input0DeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
aclTensor* input0 = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), input0DeviceMem);
```

Workspace and execution:

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
CHECK_ACL(aclnnAddCustomTemplateGetWorkspaceSize(input0, input1, output0, &workspaceSize, &executor));

void* workspaceDeviceMem = nullptr;
if (workspaceSize > 0) {
    CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}

CHECK_ACL(aclnnAddCustomTemplate(workspaceDeviceMem, workspaceSize, executor, stream));
CHECK_ACL(aclrtSynchronizeStream(stream));
```

Compile:

```bash
g++ \
  -I$ASCEND_TOOLKIT_HOME/include \
  -I${HOME}/vendors/customize/op_api/include \
  -L$ASCEND_TOOLKIT_HOME/lib64 \
  -L${HOME}/vendors/customize/op_api/lib \
  Sources/03.03/aclnn_test.cpp \
  -lcust_opapi -lnnopbase -lacl_rt \
  -o Sources/03.03/execute_add_op
```

Run:

```bash
source ${HOME}/vendors/customize/bin/set_env.bash
./Sources/03.03/execute_add_op
```

### pybind11 Invocation

pybind wraps ACLNN calls into a Python extension.

Concept:

- C++ wrapper calls the generated ACLNN API.
- pybind11 exports a Python-callable function.
- `torch_npu` provides NPU extension build helpers and runtime tensors.

Dependencies in the notebook:

```bash
pip install torch==2.9.0
pip install torch-npu==2.9.0
pip install pybind11 setuptools wheel
```

Pybind C++ wrapper:

```cpp
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"

at::Tensor npu_add_custom_template(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor z = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddCustomTemplate, x, y, z);
    return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_add_custom_template", &npu_add_custom_template, "torch add");
}
```

`setup.py` pattern:

```python
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

ext = NpuExtension(
    name="custom_ops_lib",
    sources=["./custom_op.cpp"],
    extra_compile_args=[
        "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)

setup(
    name="custom_ops",
    ext_modules=[ext],
    version="1.0",
    cmdclass={"build_ext": BuildExtension},
)
```

Build/install:

```bash
export LD_LIBRARY_PATH=${HOME}/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
cd Sources/03.03/pybind_op/
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops*.whl --force-reinstall
```

Python test:

```python
import torch
import torch_npu
import custom_ops_lib

torch.npu.config.allow_internal_format = False

shape = [8, 2048]
x = torch.rand(shape, device="cpu", dtype=torch.float16)
y = torch.rand(shape, device="cpu", dtype=torch.float16)
golden = x + y

output_npu = custom_ops_lib.npu_add_custom_template(x.npu(), y.npu())
print("is same:", torch.allclose(golden, output_npu.cpu(), rtol=0.001, atol=0.001))
```

### 03.03 Practice Extraction

Practice task: modify ACLNN call code to use:

- shape `[128, 256]`
- dtype `float32`

Training value:

- Requires updating `shape`.
- Requires updating buffer size from `sizeof(aclFloat16)` to `sizeof(float)`.
- Requires updating `aclCreateTensor` dtype from `ACL_FLOAT16` to `ACL_FLOAT`.
- Requires updating host vectors and golden comparison.
- Tests whether the model understands ACLNN call code instead of just kernel code.

## Chapter 03.04: Generalized Tiling Design

This chapter is also covered in detail by `generalized-tiling-strategy.md`. The essential content is repeated here so this reference is self-contained.

### Purpose

Generalized operator development means supporting a class of legal dtypes, shapes, and Ascend processor variants, rather than one fixed input shape. In Ascend C engineering:

- Kernel code defines compute semantics.
- Host-side Tiling defines data partitioning.
- Generalized operators require generalized Tiling.

### Tiling Concepts

- `Tiling block`: one chunk of data moved and computed at a time.
- `Tiling algorithm` / `Tiling strategy`: the algorithm that decides chunk sizes from shapes, dtype, UB size, and hardware.
- `Tiling function`: Host-side function that computes and serializes the strategy for Kernel use.

### Design Principles

1. **32B alignment / 对齐原则**
   - UB-related data spaces require 32-byte alignment.
   - Round input byte length upward to a multiple of 32B.
   - Treat 32B as the minimum Tiling calculation unit.

2. **Memory-access optimization / 访存优化**
   - Maximize useful UB usage per tile.
   - Reduce GM-to-UB copy frequency.
   - Account for all live buffers.

3. **Multi-core balance / 多核均衡**
   - Use AI Cores evenly.
   - Avoid idle cores.
   - Split remainder blocks across early cores.

### Four Partition Cases

| Case | Meaning | Required handling |
| --- | --- | --- |
| core-even, tile-even | every core gets same data; each tile same length | simple loop |
| core-even, tile-uneven | every core gets same data; final tile differs | tail-tile handling |
| core-uneven, tile-even | cores get different data; tiles equal | tail-core offset handling |
| core-uneven, tile-uneven | cores get different data; final tiles differ | tail-core and tail-tile handling |

### Worked Example

Input:

- shape `(1, 660)`
- dtype `half`
- dtype size `2B`
- AI Cores `4`

Alignment:

```text
660 * 2 = 1320B
align_up(1320B, 32B) = 1344B
1344B / 2B = 672 half elements
1344B / 32B = 42 blocks
```

Core distribution:

```text
42 // 4 = 10 blocks/core
42 % 4 = 2 remainder blocks
first 2 cores: 11 blocks each = 352B
last 2 cores: 10 blocks each = 320B
```

UB tiling example:

- UB capacity example: `768B`
- Add live buffers: `x`, `y`, `z` = 3 buffers
- per-buffer capacity: `768B / 3 = 256B`
- 32B blocks per tile: `256B / 32B = 8`
- big core: `8 + 3` blocks
- small core: `8 + 2` blocks

### Generalized Tiling Fields

```cpp
struct TilingDataTemplate {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
};
```

Important: The course comments say these are element counts, even though the hand example discusses bytes. When implementing, keep a strict conversion boundary:

```text
inputNum elements
typeLength bytes/element
inputLength bytes
inputLengthAligned32 bytes
32B block counts
field element counts = bytes / typeLength
```

### Host-Side Generalized Tiling Implementation

Required headers:

```cpp
#include "register/op_def_registry.h"
#include "../op_kernel/add_custom_template_tiling.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
```

Get hardware, shape, and dtype size:

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
auto coreNum = ascendcPlatform.GetCoreNum();

uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
uint32_t typeLength = 0;
ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
uint32_t inputLength = inputNum * typeLength;
```

Align and choose actual core count:

```cpp
const uint32_t BLOCK_SIZE = 32;
uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
coreNum = std::max(coreNum, static_cast<uint32_t>(1));
uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
context->SetBlockDim(coreNum);
```

Compute UB tile capacity:

```cpp
uint64_t ubSize;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
uint32_t ubDataNumber = 3; // x, y, z for Add
uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
```

Compute small-core fields:

```cpp
uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0
                           ? smallTileNum
                           : smallTileNum + 1;
uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
```

Compute big-core fields:

```cpp
everyCoreInputBlockNum += 1;
uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0
                         ? bigTileNum
                         : bigTileNum + 1;
uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
```

Persist fields:

```cpp
TilingDataTemplate *tiling = context->GetTilingData<TilingDataTemplate>();
tiling->smallCoreDataNum = smallCoreDataNum;
tiling->bigCoreDataNum = bigCoreDataNum;
tiling->tileDataNum = tileDataNum;
tiling->smallTailDataNum = smallTailDataNum;
tiling->bigTailDataNum = bigTailDataNum;
tiling->finalSmallTileNum = finalSmallTileNum;
tiling->finalBigTileNum = finalBigTileNum;
tiling->tailBlockNum = tailBlockNum;
```

### Kernel-Side Generalized Tiling Consumption

Kernel `Init` receives all Tiling fields:

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                            uint32_t smallCoreDataNum,
                            uint32_t bigCoreDataNum,
                            uint32_t finalBigTileNum,
                            uint32_t finalSmallTileNum,
                            uint32_t tileDataNum,
                            uint32_t smallTailDataNum,
                            uint32_t bigTailDataNum,
                            uint32_t tailBlockNum)
```

Big/small core selection and GM offset:

```cpp
uint32_t coreNum = AscendC::GetBlockIdx();
uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
this->tileDataNum = tileDataNum;

if (coreNum < tailBlockNum) {
    this->coreDataNum = bigCoreDataNum;
    this->tileNum = finalBigTileNum;
    this->tailDataNum = bigTailDataNum;
} else {
    this->coreDataNum = smallCoreDataNum;
    this->tileNum = finalSmallTileNum;
    this->tailDataNum = smallTailDataNum;
    globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) *
                         (AscendC::GetBlockIdx() - tailBlockNum);
}
```

Bind GM ranges and queues:

```cpp
xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);
pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
```

Tail-aware process loop:

```cpp
__aicore__ inline void Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

Kernel entry:

```cpp
__global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}
```

### 03.04 Practice Extraction

Practice task: implement generalized `SubCustomTemplate` from generalized `AddCustomTemplate`.

Training value:

- Best near-transfer exercise for generalized Tiling.
- Requires JSON name change, Host-side generalized Tiling, Tiling struct, Kernel class, and ACLNN test.
- Tests big-core/small-core offset correctness.

Important caution:

- The course template for practice initially writes `"op": "AddCustomTemplate"` in the Sub JSON cell, while the text says the op name must be `SubCustomTemplate`. Treat this as a deliberate bug or course typo. In a correct solution, the JSON `op` field must be `SubCustomTemplate`.

## Chapter 03.05: Tiling Template Programming, Attributes, TBuf, Workspace

### Purpose

This notebook teaches:

- `workspace`: device Global Memory scratch space.
- `TBuf`: Local Memory temporary buffers for vector compute.
- operator attributes: static parameters such as `min`, `max`, `alpha`, etc.
- `TilingKey`: selecting different Kernel implementations.
- Tiling template programming: replacing hard-to-read numeric `TilingKey` branches with template parameters.
- A `Clamp` operator that uses attributes, workspace, Tiling template selection, and dtype-dependent Kernel logic.

### Workspace

`workspace` is a Global Memory allocation on device side. It has two conceptual parts:

- System workspace: needed by Ascend C library APIs; size can be queried with `GetLibApiWorkSpaceSize`.
- User workspace: scratch memory explicitly needed by the custom operator.

Use workspace when:

- UB/L1 space is insufficient and data must be temporarily stored in GM.
- APIs such as `SyncAll` need workspace.
- An operator needs additional GM memory for exchange or caching.

Host-side workspace configuration:

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
size_t userWorkspaceSize = 256;
size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

Kernel-side workspace binding:

```cpp
AscendC::GlobalTensor<float> tmpGm;
tmpGm.SetGlobalBuffer((__gm__ float *)workspace);
```

Important:

- `workspace` is GM, so using it requires copy-out/copy-in and can reduce performance.
- Prefer `TBuf` for temporary vector compute values that fit in Local Memory.

### TBuf

`TBuf` allocates temporary Local Memory for vector compute.

Similarities with `TQue`:

- Both are initialized with `pipe.InitBuffer`.

Differences:

- `TBuf` memory is accessed with `Get<T>()`.
- `TQue` memory is accessed with `AllocTensor<T>()`.
- `TBuf` tensors do not use `EnQue` / `DeQue`.
- `TBuf` does not require `FreeTensor`.
- `TQue` requires balanced `AllocTensor` / `FreeTensor` and `EnQue` / `DeQue`.

Declare `TBuf` members:

```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
```

Initialize:

```cpp
pipe.InitBuffer(tmpBuf0, TOTAL_LENGTH * sizeof(float));
pipe.InitBuffer(tmpBuf1, TOTAL_LENGTH * sizeof(float));
pipe.InitBuffer(tmpBuf2, TOTAL_LENGTH * sizeof(float));
```

Use in compute:

```cpp
AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();

AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
AscendC::Add(tmpTensor2, tmpTensor0, tmpTensor1, TOTAL_LENGTH);
AscendC::Cast(zLocal, tmpTensor2, AscendC::RoundMode::CAST_RINT, TOTAL_LENGTH);
```

Never call `EnQue`, `DeQue`, or `FreeTensor` on a tensor obtained from `TBuf.Get<T>()`.

### Operator Attributes

Attributes are static operator parameters determined before execution and unchanged during compute.

Properties:

- Static: fixed for the operator invocation.
- Decoupled from dynamic tensor data.
- Required attributes define the operator's core behavior.
- Optional attributes tune behavior and have defaults.

Prototype JSON with attributes:

```json
"attr": [
  {
    "name": "max",
    "type": "float",
    "param_type": "optional",
    "default_value": 0
  },
  {
    "name": "min",
    "type": "float",
    "param_type": "optional",
    "default_value": 0
  }
]
```

Register attributes in `OpDef`:

```cpp
this->Attr("max").AttrType(OPTIONAL).Float(0);
this->Attr("min").AttrType(OPTIONAL).Float(0);
```

Read attributes in `TilingFunc`:

```cpp
float minValue = *context->GetAttrs()->GetFloat(0);
tiling->min = minValue;
```

Attribute-aware ACLNN calls:

- If an operator has attributes, the first ACLNN API usually includes attributes before outputs or before workspace/executor depending on generated signature.
- The caller must pass attribute values even when using defaults, because the generated function signature expects them.

### TilingKey

`TilingKey` chooses among different Kernel implementations or specializations.

Host-side numeric `TilingKey` pattern:

```cpp
if (condition) {
    context->SetTilingKey(1);
} else {
    context->SetTilingKey(2);
}
```

Kernel-side branch:

```cpp
if (TILING_KEY_IS(1)) {
    ProcessA();
} else if (TILING_KEY_IS(2)) {
    ProcessB();
}
```

Problem:

- Numeric TilingKeys are hard to remember and maintain.
- Many combinations of dtype, tile count, and algorithm choice become difficult to manage.

### Tiling Template Programming

Tiling template programming replaces raw numeric TilingKey management with template arguments.

Header:

```cpp
#include "ascendc/host_api/tiling/template_argument.h"
```

Template declaration example:

```cpp
ASCENDC_TPL_ARGS_DECL(AddTemplateCustom,
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(1)),
ASCENDC_TPL_DATATYPE_DECL(D_T_Z, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_OUTPUT(0)),
ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5, 10, 12, 13, 9, 8),
ASCENDC_TPL_BOOL_DECL(IS_SPLIT, 0, 1),
);
```

Common template argument macros:

- `ASCENDC_TPL_ARGS_DECL`: declare operator template argument set.
- `ASCENDC_TPL_DATATYPE_DECL`: declare dtype template parameter.
- `ASCENDC_TPL_UINT_DECL`: declare unsigned integer template parameter.
- `ASCENDC_TPL_BOOL_DECL`: declare boolean template parameter.
- `ASCENDC_TPL_INPUT(n)`: bind parameter to input `n`.
- `ASCENDC_TPL_OUTPUT(n)`: bind parameter to output `n`.
- `ASCENDC_TPL_SEL`: declare legal combinations.
- `ASCENDC_TPL_ARGS_SEL`: declare one legal argument combination.
- `ASCENDC_TPL_DATATYPE_SEL`: declare dtype value inside a legal combination.
- `ASCENDC_TPL_SEL_PARAM`: set template parameters in Host-side `TilingFunc`.

Host-side template parameter selection:

```cpp
uint32_t D_T_X = static_cast<int>(dtype_x);
uint32_t D_T_Y = static_cast<int>(dtype_y);
uint32_t D_T_Z = static_cast<int>(dtype_z);
uint32_t TILE_NUM = 1;
uint32_t IS_SPLIT = 0;

if (totalLength < MIN_LENGTH_FOR_SPLIT) {
    IS_SPLIT = 0;
    TILE_NUM = 1;
} else {
    IS_SPLIT = 1;
    TILE_NUM = DEFAULT_TILE_NUM;
}

ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
```

Kernel-side template parameters:

```cpp
template <typename D_T_X, typename D_T_Y, typename D_T_Z, int TILE_NUM, int IS_SPLIT>
__global__ __aicore__ void add_template_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
    KernelAdd<D_T_X, D_T_Y, D_T_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
    if (IS_SPLIT == 0) {
        op.Process1();
    } else if (IS_SPLIT == 1) {
        op.Process2();
    }
}
```

### Clamp Example

The notebook implements a fixed-shape `Clamp` operator to demonstrate:

- attributes
- workspace
- template-argument-based TilingKey
- dtype-dependent Kernel paths

Prototype:

- op: `Clamp`
- input: `x`, shape `[8, 2048]`, dtype `int32` or `float`, format `ND`
- output: `y`, same shape/dtype/format
- optional attribute: `min`, type `float`, default `0`

JSON:

```json
[
  {
    "op": "Clamp",
    "input_desc": [
      {
        "name": "x",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int32", "float"]
      }
    ],
    "output_desc": [
      {
        "name": "y",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int32", "float"]
      }
    ],
    "attr": [
      {
        "name": "min",
        "type": "float",
        "param_type": "optional",
        "default_value": 0
      }
    ]
  }
]
```

Template file `tiling_key_clamp.h`:

```cpp
#ifndef TILING_KEY_CLAMP_H
#define TILING_KEY_CLAMP_H
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Clamp,
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_INT32, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_INT32, C_DT_FLOAT, ASCENDC_TPL_OUTPUT(0)),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT32),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_INT32),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT),
    ),
);
#endif
```

Clamp Tiling struct:

```cpp
struct ClampTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
    float min;
};
```

Clamp Host-side extras:

```cpp
#include "../op_kernel/tiling_key_clamp.h"
#include "tiling/platform/platform_ascendc.h"
```

Important Host-side logic:

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ClampTilingData *tiling = context->GetTilingData<ClampTilingData>();
uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
ge::DataType dtype_y = context->GetOutputDesc(0)->GetDataType();
uint32_t D_T_X = static_cast<int>(dtype_x);
uint32_t D_T_Y = static_cast<int>(dtype_y);

float min_value = *context->GetAttrs()->GetFloat(0);
tiling->totalLength = totalLength;
tiling->tileNum = 8;
tiling->min = min_value;
context->SetBlockDim(8);
ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y);

size_t userWorkspaceSize = 256 * 4;
size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

Clamp Kernel strategy:

- Template parameters choose dtype path.
- Float path can compute directly with `Maxs`.
- Int32 path casts or stages through float workspace before `Maxs`, depending on implementation.
- `workspace` is bound as a `GlobalTensor<float>`.
- The example is intentionally not generalized; it focuses on template/attr/workspace usage for `[8, 2048]`.

### 03.05 Practice Extraction

Practice task: extend `AddCustomTemplate` with `max` and `min` attributes:

```text
sum = x + y
if sum > max: output = max
if sum < min: output = min
otherwise: output = sum
```

Supported practice spec:

- shape `[8, 2048]`
- dtypes `int8`, `float`
- formats `ND`
- attributes `max`, `min`
- output must be clamped into `[min, max]`

Training value:

- Tests attribute JSON and `OpDef` registration.
- Tests template parameters for dtype combinations.
- Tests `TBuf` or workspace use for dtype conversion and temporary compute.
- Tests ACLNN attribute passing.
- Tests numerical boundary behavior.

Common pitfalls:

- Forgetting generated ACLNN first-stage API includes attributes.
- Registering attributes in JSON but not in `OpDef`.
- Reading attributes but not storing them in Tiling data.
- Defining template dtype combinations that allow invalid input/output pairs.
- Using `TBuf` tensors as if they were `TQue` tensors.
- Allocating user workspace but forgetting system workspace.

## Chapter 03.06: Chapter Practice - SigmoidCustom

### Purpose

The integrated chapter practice asks the learner to implement `SigmoidCustom`:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Required:

- Implement Kernel-side code.
- Implement Host-side Tiling.
- Support `float` and `half` input/output.
- Pass multiple call cases:
  - case 1: shape `(7, 83)`, dtype `float32`
  - case 2: shape `(1024, 1024)`, dtype `float16`
  - case 3: shape `(999, 999)`, dtype `float32`

This is a strong final exercise because it combines:

- operator prototype JSON
- `msopgen`
- Host-side Tiling
- dynamic shape support
- dtype support
- vector math
- ACLNN invocation
- correctness testing across small, large, and awkward shapes

### Prototype JSON

```json
[
  {
    "op": "SigmoidCustom",
    "input_desc": [
      {
        "name": "x",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
      }
    ],
    "output_desc": [
      {
        "name": "y",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
      }
    ]
  }
]
```

### Starter Host Code

The starter Host-side code provides:

- `SigmoidCustomTilingData *tiling`
- input shape traversal
- `tiling->size = data_sz`
- `context->SetBlockDim(8)`
- zero workspace
- `InferShape`
- `InferDataType`
- `OpDef` registration for `float16` and `float`

Starter Tiling struct:

```cpp
struct SigmoidCustomTilingData {
    uint32_t size;
};
```

Starter Kernel:

```cpp
extern "C" __global__ __aicore__ void sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // implement Kernel-side code
}
```

### Recommended Implementation Strategy

For a robust solution, do not stop at the starter `size` field. Reuse the generalized Tiling strategy from 03.04:

- align bytes to 32B
- choose actual `blockDim`
- compute big/small core lengths
- compute tile sizes from UB and live buffers
- handle tail core and tail tile

For sigmoid compute:

```text
y = 1 / (1 + exp(-x))
```

Vector implementation pattern:

1. Load `x`.
2. Cast to `float` if input is `half` and precision requires it.
3. `Muls(tmp0, x, -1.0f, n)`.
4. `Exp(tmp0, tmp0, n)`.
5. `Duplicate(one, 1.0f, n)`.
6. `Add(tmp1, one, tmp0, n)`.
7. `Div(yFloat, one, tmp1, n)`.
8. Cast back to `half` if needed.
9. Store `y`.

Use `TBuf` for temporary tensors if multiple temporaries are required.

### ACLNN Test Practice

The starter ACLNN test defaults to:

- shape `[8, 2048]`
- dtype `float16`
- input values all `1.0`
- golden output approximately `0.7310585786300049`

The practice requires modifying the call code for each case.

When changing dtype:

- update C++ host vector type
- update `aclDataType`
- update `sizeof(T)` buffer size
- update golden values and tolerance
- update output conversion / print logic

For exact equality, half precision may be too strict. Prefer tolerance-based checks for generated training tasks.

## Cross-Chapter Training Dataset

Create training problems in this order:

1. **Engineering Skeleton**
   - Given operator JSON, ask for generated project structure and file responsibilities.

2. **Host Registration**
   - Complete `OpDef`, `InferShape`, `InferDataType`, and `SetTiling`.

3. **Basic Tiling**
   - Fill `totalLength`, `tileNum`, and `SetBlockDim`.

4. **Kernel Tiling Consumption**
   - Add `workspace` and `tiling` signature, `REGISTER_TILING_DEFAULT`, and `GET_TILING_DATA_WITH_STRUCT`.

5. **ACLNN Call**
   - Write a two-stage ACLNN C++ test for a custom operator.

6. **pybind Call**
   - Wrap the ACLNN call into a `torch_npu` pybind extension.

7. **Generalized Tiling Calculation**
   - Given shape, dtype, core count, and UB size, calculate big/small core fields.

8. **Generalized Kernel Offset**
   - Complete big-core/small-core GM offset logic.

9. **Workspace vs TBuf Decision**
   - Decide when to use GM workspace and when to use Local Memory `TBuf`.

10. **Attribute-Aware Operator**
    - Add `min`/`max` attributes and pass them through JSON, `OpDef`, Tiling, Kernel, and ACLNN call.

11. **Tiling Template Programming**
    - Define template args, legal dtype combinations, Host `ASCENDC_TPL_SEL_PARAM`, and Kernel template parameters.

12. **Integrated Sigmoid**
    - Implement `SigmoidCustom` for multiple shapes and dtypes.

## Verification Checklist

For any Chapter 3 style operator:

- JSON `op` name matches generated file names and Kernel function name.
- `OpDef` dtype/format matches JSON.
- Shape inference copies or computes correct output shape.
- Dtype inference sets correct output dtype.
- `SetTiling` points to the intended `TilingFunc`.
- `AddConfig` matches target SoC.
- `SetBlockDim` does not assign empty work to invalid cores for generalized operators.
- Tiling fields consistently use element counts or byte counts, never mixed accidentally.
- Kernel reads Tiling data before using fields.
- `GlobalTensor` offsets are per core and in bounds.
- `CopyIn`, `Compute`, and `CopyOut` use the valid element count for the current tile.
- `TQue` flows have balanced `AllocTensor`, `EnQue`, `DeQue`, and `FreeTensor`.
- `TBuf.Get<T>()` tensors are not enqueued, dequeued, or freed.
- Workspace size includes system workspace when required by library APIs.
- ACLNN first-stage API is called before second-stage API.
- Workspace is allocated only when `workspaceSize > 0`.
- Stream is synchronized before reading outputs.
- Device memory and ACL tensors are released.
- Tests cover at least one aligned case, one non-aligned case, one small case, and one large case.

## Common Mistakes

- Writing only Kernel code and forgetting Host-side engineering.
- Treating quick-mode kernels as deployable standard operators.
- Hardcoding `[8, 2048]`, `blockDim=8`, or dtype-specific assumptions.
- Forgetting that standard operator Kernel signatures include `workspace` and `tiling`.
- Using `GetOriginShape()` vs `GetStorageShape()` without thinking about format/layout.
- Failing to include generated ACLNN headers and `libcust_opapi.so` during compilation.
- Passing no attributes to an attribute-aware ACLNN API.
- Using workspace for temporary data that should be kept in UB via `TBuf`.
- Forgetting to add `tiling_key_*.h` to Kernel and Host code when using template programming.
- Letting template legal combinations accept mismatched input/output dtypes.
- Comparing floating-point outputs with exact equality in training tests.
