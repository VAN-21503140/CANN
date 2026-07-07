# FastGelu CANN Operator Handoff

This package is for another Codex instance to continue or parallelize work on the CANNJudge FastGelu problem.

## Current Status

- Problem: FastGelu
- Judge page: `https://cannjudge.cn/public/s1/fastgelu`
- Submit page: `https://cannjudge.cn/public/s1/fastgelu/submit`
- CANNJudge account observed: `VAN`
- Accepted submission ID: `83644`
- Submit time: `2026/07/07 11:03:36`
- Result: `Pass`
- Visible tests: `5 / 5`
- Visible times: `4.72us, 4.72us, 9.08us, 8.56us, 12.80us`
- Baseline git commit: `7a93dcb Solve FastGelu CANN operator`

## Important Safety Notes

- The CANNJudge button `提交代码` is a direct final-submit action. It does not open a confirmation dialog.
- Do not click `提交代码` unless the user explicitly asks to submit.
- Browser editor red underlines are expected because the browser does not know CANN include paths/macros. The VM CANN build passed.

## Local Paths

Windows project:

```text
D:\Robot\FastGelu_problem_190_template
```

Main code:

```text
D:\Robot\FastGelu_problem_190_template\code
```

Detailed conversation history:

```text
D:\Robot\FastGelu_problem_190_template\conversation_log.md
```

VM copy used for CANN build:

```text
/home/a/FastGelu_problem_190_template/code
```

## VM / CANN Environment

SSH:

```text
user: a
host: 192.168.199.128
key: C:\Users\lenovo\.ssh\codex_vm_ed25519
```

CANN environment script found:

```bash
source /home/a/Ascend/ascend-toolkit/set_env.sh
```

Successful VM build command:

```bash
source /home/a/Ascend/ascend-toolkit/set_env.sh
cd /home/a/FastGelu_problem_190_template/code
mkdir -p build_codex_20260707_1058
cmake -S . -B build_codex_20260707_1058
cmake --build build_codex_20260707_1058 -j 4
```

Successful build evidence included:

```text
[ascend910b] Generating FastGelu_... Done
[100%] Built target custom
INFO target name: cust_optiling
INFO target name: cust_opapi
INFO target name: ascendc_kernels
```

The user's computer has no NPU, so local/VM testing can validate CANN compilation/package generation only. Real numeric correctness and performance require CANNJudge or an Ascend 910B machine.

## Files To Understand First

```text
code/op_host/fast_gelu.cpp
code/op_kernel/fast_gelu_tiling.h
code/op_kernel/tiling_key_fast_gelu.h
code/op_kernel/fast_gelu.cpp
```

## Current Implementation Summary

Host side:

- Reads input tensor shape size and dtype.
- Saves total element count into tiling data.
- Selects template dtype with `ASCENDC_TPL_SEL_PARAM`.
- Sets `blockDim` from AIV core count, capped by `length_x` for small tensors.
- Sets workspace size to zero.
- Infers output shape equal to input shape.
- Registers `float16` and `float32`, ND format, `ascend910b`.

Kernel side:

- Formula:

```text
y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
```

- Uses per-core split via `GetBlockNum()` and `GetBlockIdx()`.
- Uses `TILE_ELEM_NUM = 1024`.
- Uses `DataCopyPad` for copy-in and copy-out.
- Uses input/output queues with `BUFFER_NUM = 2`.
- Uses four temporary `TBuf`s: `absBuf_`, `denomBuf_`, `workBuf_`, `numerBuf_`.
- Uses conservative `PipeBarrier<PIPE_V>()` between dependent vector ops.

## Static Verification

Windows static test:

```powershell
cd D:\Robot\FastGelu_problem_190_template\code
python -m unittest discover -s tests -p 'test_*.py'
```

Last observed result:

```text
Ran 6 tests
OK
```

## WebBridge Notes

Kimi WebBridge is installed and worked through local daemon:

```text
http://127.0.0.1:10086/command
```

The CANNJudge online editor did not work with WebBridge `fill`; it returned an extension-side error. The working method was:

1. Select file tree button by exact `title`, for example `op_kernel/fast_gelu.cpp`.
2. Use the native `HTMLTextAreaElement.prototype.value` setter on `textarea#editor-main`.
3. Dispatch `input` and `change` events.
4. Switch away and back, then compare page content with local content by length and SHA-256.

## Optimization Ideas

Current version is a stable accepted baseline, not an aggressive performance version.

Reasonable next experiments:

1. Remove `numerBuf_` and compute numerator directly into `yLocal`:

```cpp
AscendC::Mul(yLocal, xLocal, workLocal, count);
AscendC::Div(yLocal, yLocal, denomLocal, count);
```

2. Try larger tile sizes:

```cpp
TILE_ELEM_NUM = 2048
TILE_ELEM_NUM = 4096
```

3. Try more conservative small-shape `blockDim`:

```cpp
block_dim = min(num_cores_aiv, ceil(length_x / TILE_ELEM_NUM))
```

4. Experiment with reducing some `PipeBarrier<PIPE_V>()` calls, but this has higher correctness risk.

Recommended optimization workflow:

1. Create a git branch from current baseline.
2. Change one variable at a time.
3. Build in VM.
4. Sync to CANNJudge editor.
5. Submit only with user confirmation.
6. Compare visible times against baseline:

```text
4.72us, 4.72us, 9.08us, 8.56us, 12.80us
```
