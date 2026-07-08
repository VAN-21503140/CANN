---
name: cann-operator-development
description: Use when designing, implementing, calling, debugging, profiling, or optimizing Huawei Ascend C/CANN custom operators, including generalized tiling, 泛化 tiling, tilling, tiling设计, 通用算子, 算子策略, workspace, TBuf, attributes, TilingKey, Tiling templates, Matmul/Cube/Vector operators, VV/CV fused operators, ops-math open-source operator contribution, UT/ST, ACLNN, pybind, Host-side Tiling, DumpTensor, printf, Plog logs, msdebug, msprof/msopprof, simulation profiling, mssanitizer workflows, and course practice operators such as Sigmoid or LogSigmoid.
---

# CANN Operator Development

## Overview

Use this skill for Huawei Ascend C / CANN custom operator work. It turns the extracted course material into a practical workflow for choosing an operator path, writing Kernel and Host code, packaging, calling, debugging, and tuning operators.

For detailed patterns, read `references/operator-development.md` when the task involves actual operator design, code review, debugging, profiling, or optimization. For practical CANN sample-derived code patterns, especially elementwise vector operators and `DataCopyPad` tails, read `references/ascendc-samples-patterns.md`.

For beginner Ascend C learning tasks, Hello World, programming-paradigm explanation, or Add-kernel teaching tasks, read `references/ascendc-learning-hub-basics.md`.

For generalized tiling, 泛化 tiling, tilling, tiling设计, 通用算子, 算子策略, 32B 对齐原则, 多核均衡, or 访存优化 tasks, read `references/generalized-tiling-strategy.md`.

For intermediate vector operator development, `msopgen` engineering projects, ACLNN single-operator calls, pybind11 / torch_npu calls, workspace, TBuf, operator attributes, TilingKey, Tiling template programming, or SigmoidCustom chapter practice, read `references/intermediate-vector-operator-development.md`.

For Matmul/Cube operator work, FRACTAL_NZ/ZZ/ZN formats, `Matmul` high-level API, `REGIST_MATMUL_OBJ`, `MultiCoreMatmulTiling`, `TCubeTiling`, or int8/int32 Matmul practice, read `references/matmul-operator-development.md`.

For fused operator work, VV fusion, CV fusion, `SquareDiff`, `MatmulLeakyreluCustom`, `MatmulAbs`, `MatmulSinh`, or Cube/Vector pipeline collaboration, read `references/fused-operator-development.md`.

For CANN open-source operator repository work, `ops-math`, `build.sh --genop`, `experimental/math`, `op_graph`, `op_host`, `op_kernel`, `examples`, contribution packaging, UT, ST, or migrating a single-operator project into the open-source repo layout, read `references/opensource-operator-repo.md`.

For functional debugging, CPU-domain twin debugging, NPU on-board debugging, `DumpTensor`, `AscendC::printf`, Plog/log paths, accuracy mismatches, address-alignment errors, or typical issue triage, read `references/troubleshooting-debugging.md`.

For performance work, `msProf`, `msopprof`, on-board profiling, simulator profiling, trace analysis, PipeUtilization, scalar/vector/MTE bottlenecks, Tiling load balance, double buffering, or memory-access optimization, read `references/performance-optimization.md`.

For the final vector practice operator `LogSigmoidCustom`, dynamic-shape vector practice, or multi-dtype float/half/bfloat16 testing, read `references/course-practice-logsigmoid.md`.

For the source notebook inventory from GitCode CANN learning hub, read `references/learning-hub-source-index.md`.

## Route the Task

1. If the user asks for concepts, explain from the quick model below and cite the relevant reference section.
2. If the user asks about Ascend C beginner learning, Hello World, programming paradigm, or Add-kernel teaching, read `references/ascendc-learning-hub-basics.md` first.
3. If the user asks about generalized tiling, 泛化, tilling/tiling design, 通用算子 strategy, 32B alignment, multi-core balance, tail-core/tail-tile splitting, or memory-access optimization, read `references/generalized-tiling-strategy.md` first.
4. If the user asks about vector operator engineering, `msopgen`, standard operator project structure, ACLNN calls, pybind calls, workspace, TBuf, attributes, TilingKey, Tiling templates, or the Chapter 3 Sigmoid practice, read `references/intermediate-vector-operator-development.md` first.
5. If the user asks about Matmul, Cube, `Matmul` high-level API, matrix formats, or matrix tiling, read `references/matmul-operator-development.md` first.
6. If the user asks about VV/CV fusion, Matmul+Vector fusion, or reducing intermediate GM traffic between operators, read `references/fused-operator-development.md` first.
7. If the user asks about open-source CANN operator repository contribution, `ops-math`, UT/ST, or migration from a single-operator project, read `references/opensource-operator-repo.md` first.
8. If the user asks to design or implement an operator, read `references/operator-development.md` sections 1-7 and then the chapter-specific reference above before giving code or a plan.
9. If the user asks about wrong results, crashes, illegal memory access, `DumpTensor`, `printf`, Plog, alignment, or race conditions, read `references/troubleshooting-debugging.md` first, then `references/operator-development.md` sections 8-9 if needed.
10. If the user asks about performance, profiling, or optimization, read `references/performance-optimization.md` first, then `references/operator-development.md` sections 9-10 if needed.
11. If the user asks about PyTorch or whole-network replacement, read `references/operator-development.md` section 7 first.

## Quick Model

CANN custom operators usually split into two paths:

- **AI Core operator**: for data-intensive vector, matrix, and scalar tensor work. Prefer this for performance-sensitive operators.
- **AI CPU operator**: for non-matrix complex logic, branch-heavy work, unsupported AI Core data types, or temporary functional bring-up before rewriting as AI Core.

Ascend C custom operator development has two common modes:

- **Quick mode**: write only the operator kernel and invoke it with kernel launch syntax `<<<...>>>`; use this to validate logic quickly.
- **Standard mode**: generate an operator project, implement Kernel-side and Host-side deliverables, compile/deploy an operator package, then call it through ACLNN/ACLOP/PyTorch adapter paths.

Generalized operator strategy:

- **泛化 / generalized operator**: supports a class of valid dtypes, shapes, and Ascend processor variants instead of one fixed case.
- **Generalized tiling / 泛化 tiling / tiling设计**: Host-side strategy that converts runtime shape, dtype size, UB size, and AI Core count into Kernel-side fields.
- **Core principles**: 32B alignment, memory-access optimization / 访存优化, multi-core balance / 多核均衡, and explicit tail-core plus tail-tile handling.
- **Host/Kernel contract**: Host computes fields such as big/small core data sizes, tile count, tile length, tail length, and `tailBlockNum`; Kernel consumes them through `REGISTER_TILING_DEFAULT`, `GET_TILING_DATA_WITH_STRUCT`, `GetBlockIdx()`, and `CopyIn -> Compute -> CopyOut`.

## Core Workflow

1. Analyze the operator: math formula, inputs/outputs, dtype, shape, format, supported SoCs, expected call path, and performance target.
2. Decide AI Core vs AI CPU and quick vs standard mode.
3. For AI Core Kernel code, structure the kernel as `Init`, `Process`, `CopyIn`, `Compute`, and `CopyOut` around `GlobalTensor`, `LocalTensor`, `TPipe`, and `TQue`.
4. Define Tiling: distribute work across AI Cores, calculate per-core offsets, choose tile sizes, and decide whether double buffering is worthwhile.
5. For standard mode, implement Host-side Tiling data, `TilingFunc`, shape inference, operator prototype registration, and SoC configs.
6. Build and deploy the package, then test through the intended API path: kernel launch, ACLNN single-operator API, or PyTorch adapter.
7. Debug correctness before tuning performance. Use CPU-domain twin debugging first where possible; use NPU-domain DumpTensor/PRINTF/msdebug when on-board behavior differs.
8. Profile and tune using msprof metrics: compute time, copy time, bandwidth, resource conflicts, cache hit rate, and pipeline stalls.
9. For open-source repo contribution, split deliverables into `op_host`, `op_kernel`, optional `op_graph`, `examples`, and `tests/ut`, then validate with both UT and ST.

## Verification Checklist

Before claiming an Ascend C operator is ready:

- Confirm input/output dtype, shape, format, and memory sizes match Kernel and Host assumptions.
- Confirm `blockDim`, `GetBlockIdx()`, per-core offsets, tile counts, and tail handling cover every element exactly once.
- Confirm each `AllocTensor` has a matching `FreeTensor`, and each queue flow has balanced `EnQue` / `DeQue`.
- Confirm `DataCopy` ranges are aligned and inside allocated GM/HBM and local-memory bounds.
- Confirm dynamic-shape operators pass required Tiling fields into Kernel code via `GET_TILING_DATA`.
- Confirm standard-mode packages compile, deploy, and expose the expected ACLNN or framework adapter entrypoint.
- For performance claims, compare msprof evidence before and after the optimization.

## Common Mistakes

- Treating quick kernel launch validation as equivalent to a deployable standard operator.
- Hardcoding fixed-shape tile sizes in a dynamic-shape operator without Host-side Tiling data.
- Forgetting that `num=2` in `TPipe.InitBuffer` enables double-buffer style queueing only when the CopyIn/Compute/CopyOut loop is structured to exploit it.
- Using AI CPU as the final path for data-intensive vector/matrix work instead of a bring-up or unsupported-case fallback.
- Optimizing code before proving correctness with CPU/NPU debug and sanitizer checks.
