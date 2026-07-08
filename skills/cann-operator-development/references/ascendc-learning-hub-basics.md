# Ascend C Learning Hub Basics

Source: `https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development/02_AscendC_basic`

Use this reference when the task is to teach or learn beginner Ascend C operator development, especially the path from Hello World to an Add kernel. Do not copy notebook text verbatim into answers; use the source as a course map and extract only the relevant pattern.

## Chapter Scope

Chapter 2, `02_AscendC_basic`, is an Ascend C operator development basics practice chapter. It covers:

- Ascend C chapter introduction.
- Ascend C Hello World.
- Ascend C programming paradigm and API introduction.
- Kernel function mechanics based on an Add operator.
- Chapter practice / test.

Visible directory structure:

```text
answer/
images/
src/
02.01_chapter_intro.ipynb
02.02_HelloWorld.ipynb
02.03_ascendc_programming_paradigm_and_api_introduction.ipynb
02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb
02.05_chapter_test.ipynb
README.md
```

## Routing Guidance

Use this source before deeper references when the user asks about:

- how to start learning Ascend C,
- the minimum runnable Ascend C example,
- Ascend C programming model at a beginner level,
- Add operator kernel-function structure,
- turning CANN learning material into beginner operator exercises.

Use `operator-development.md` instead when the user is designing a deployable custom operator, Host-side Tiling, ACLNN integration, debugging, or profiling.

Use `ascendc-samples-patterns.md` instead when the task needs concrete sample-derived code patterns such as `TPipe`, `TQue`, `DataCopyPad`, tail-safe vector movement, or FastGelu-style vector API composition.

## Beginner Curriculum Spine

Recommended teaching order:

1. Explain what an Ascend C operator is and where it runs.
2. Show the Hello World execution loop.
3. Introduce the kernel-function boundary: host code launches device kernel code.
4. Introduce global memory vs local tensor movement.
5. Explain the Add operator as the first real tensor kernel.
6. Convert the Add example into a training problem with tests and boundary cases.

## Problem Extraction Template

When converting this chapter into AI training data, produce a problem record with:

- `title`: beginner task name, such as `Ascend C Hello World` or `Add Kernel Introduction`.
- `source`: the notebook or source directory path.
- `learning_goal`: the one concept the exercise should teach.
- `starter_context`: short explanation of the required API or execution model.
- `task`: what the learner must implement or explain.
- `expected_observations`: what successful execution or correct explanation should include.
- `common_mistakes`: beginner failures to watch for.
- `next_step`: the following exercise in the path.

## Add Kernel Exercise Notes

For an Add-kernel teaching task, make the AI state or implement:

- inputs and output live in global memory,
- the kernel receives raw GM addresses,
- the kernel object binds `GlobalTensor` views in `Init`,
- `Process` performs the actual data movement and vector computation,
- work must be split by block/core when moving beyond tiny examples,
- every tail or short-input case must be handled explicitly in production examples.

For the very first Add lesson, keep performance out of scope. The correct teaching goal is to make the host/device split and kernel-function mechanics clear. Introduce `TPipe`, `TQue`, tiling, and `DataCopyPad` after the learner understands the minimal Add flow.

## Skill-Usage Rules

- Keep answers beginner-centered when this reference is selected.
- Prefer a small runnable mental model over a full standard operator package.
- Do not imply that the Hello World or first Add notebook is production-ready.
- When a user asks to generate training data, output structured exercises rather than a narrative summary.
- When a user asks for code, mention whether the code is quick-mode learning code or standard deployable operator code.
