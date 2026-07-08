# Ascend C Performance Optimization

Use this reference for profiling and optimizing Ascend C operators with msProf/msopprof, simulator traces, Tiling strategy, double buffering, and memory-access improvements.

Source course: `tutorials/ascendc_operator_development/08_performance_optimization`.

## Table of Contents

- Optimization workflow
- msProf on-board profiling
- Profiling data interpretation
- Simulator profiling
- Tiling strategy optimization
- Pipeline and double-buffer optimization
- Memory-access optimization
- Common optimization traps

## Optimization Workflow

Use this order:

1. Prove correctness.
2. Capture a baseline profile.
3. Identify the dominant bottleneck by evidence.
4. Change one optimization lever.
5. Rebuild, rerun, and capture another profile.
6. Compare before/after.
7. Keep the change only if evidence improves the target metric without breaking correctness.

Do not optimize with debug `printf`/`DumpTensor` enabled unless the purpose is to measure debug overhead.

## msProf / msopprof On-Board Profiling

`msProf` profiling is exposed through CANN tooling. The course notes that the practical executable is `msopprof`, provided by CANN and used equivalently to `msprof op` for operator profiling.

It can collect:

- On-board performance data from executable tests.
- Simulator data for specified products.
- Operator binary profiling in supported modes.

Command shape:

```bash
msopprof [msProf options] ./execute_op [program args]
```

If the executable has arguments, put executable arguments after the executable name. Arguments before the executable are profiling-tool arguments.

Typical output directory contains:

```text
OPPROF_<timestamp>/
  visualize_data.bin
  OpBasicInfo.csv
  ResourceConflictRatio.csv
  PipeUtilization.csv
  Memory.csv
  L2Cache.csv
  MemoryL0.csv
  ...
```

Key files:

- `OpBasicInfo.csv`: op name, block dim, duration.
- `PipeUtilization.csv`: compute/move unit time and ratios.
- `Memory.csv`: UB/L1/L2/GM bandwidth and read/write stats.
- `ResourceConflictRatio.csv`: UB bank conflict/resource conflict ratios.
- `L2Cache.csv`: L2 cache hit rate.
- `visualize_data.bin`: visualization payload for tooling.

## Profiling Data Interpretation

Start with the largest time contributors:

- Total op duration.
- AIV time per block.
- Vector time and ratio.
- Scalar time and ratio.
- MTE move time and ratio.
- Memory bandwidth utilization.
- Resource conflict ratios.

The course gives two useful theoretical baselines:

- GM move time roughly equals bytes moved divided by peak bandwidth. Example: `sizeof(float) * 4096 * 4096 / 1.8 TB/s`.
- Vector compute time roughly equals operation count divided by theoretical throughput.

Use theoretical numbers as a target, not as an exact expected runtime. Small transfers often cannot saturate bandwidth; simultaneous transfers share bandwidth.

### Bottleneck Heuristics

- High scalar time:
  - too much scalar loop work in Kernel
  - debug `printf`
  - address arithmetic inside inner loops
  - shape/tile computation that could be Host-side
- High vector time:
  - compute-heavy vector op
  - no overlap with CopyIn/CopyOut
  - unnecessary repeated vector passes
- High MTE/Copy time:
  - too many small transfers
  - poor alignment
  - repeated GM round trips
  - no fusion where intermediates could remain local
- Uneven block times:
  - unbalanced Tiling
  - tail cores doing more work than others
  - overuse of cores for tiny data
- Resource conflicts:
  - UB bank conflicts
  - poor local memory layout

## Simulator Profiling

Simulator mode is useful when hardware access is limited or when analyzing instruction timelines.

The course notes simulator profiling has extra requirements:

1. Modify `op_kernel/CMakeLists.txt` to enable debug/simulator-friendly build options.
2. Prepare a callable test program.
3. Collect simulator data with product model specified.

Typical simulator output:

```text
OPPROF_<timestamp>/
  simulator/
    core0.veccore0/
    trace.json
    visualize_data.bin
  dump/
    aicore_binary.o
    object_dump.txt
    pc_start_addr.txt
```

Analyze `trace.json` with:

- Chrome `chrome://tracing`
- MindStudio Insight
- A simple generated chart if only high-level timeline inspection is needed

Simulator trace helps map time to:

- MTE stages
- Vector stages
- Scalar stages
- idle/wait spans
- source lines when tooling/source mapping is configured

Course example: high scalar time was dominated by debug printing; removing prints greatly reduced scalar overhead.

## Tiling Strategy Optimization

Tiling should distribute work evenly and avoid unnecessary tails.

Problem pattern:

- Physical core count is fixed.
- Data splitting creates tail-core work.
- Some cores finish early and stay idle.
- Total runtime becomes dominated by the slowest core.

Optimization pattern from the course:

- For input shape like `[45, 20480]`, a naive split can leave uneven work.
- If 40 Vector cores are available, setting `blockDim` to 40 and distributing data evenly can reduce idle time.
- Move tail work so all cores process nearly equal amounts rather than letting a small number of tail cores do an extra full tile.

Checklist:

- Does `blockDim` match meaningful parallelism, not just maximum available cores?
- Is per-core data length balanced?
- Are tails handled explicitly?
- Is tile count too high for small data?
- Is Host tiling doing work that Kernel otherwise repeats scalarly?

## Pipeline and Double Buffer Optimization

Without overlap, a vector operator may run:

```text
CopyIn -> Compute -> CopyOut
CopyIn -> Compute -> CopyOut
...
```

If CopyIn, Compute, and CopyOut each cost about `t`, Vector utilization can be poor because compute waits for moves.

Double buffering splits data so while one tile computes, the next tile can copy in:

```text
Tile 0 CopyIn
Tile 0 Compute + Tile 1 CopyIn
Tile 0 CopyOut + Tile 1 Compute + Tile 2 CopyIn
...
```

Implementation pattern:

- Set `BUFFER_NUM = 2`.
- Initialize queues with buffer depth 2.
- Structure the Process loop to actually pipeline CopyIn/Compute/CopyOut; setting buffer depth alone is not enough.

Use double buffering when:

- Data size is large enough to amortize queue overhead.
- Copy and compute can overlap.
- UB has enough space for two buffers plus temporaries.

Avoid double buffering when:

- Data is too small.
- UB pressure causes smaller transfers and worse bandwidth.
- The loop structure remains effectively serial.

## Memory-Access Optimization

The course notes that single transfers around or above 16 KB often use bandwidth better than very small transfers. Treat this as a practical heuristic, not a universal law.

Optimize memory access by:

- Increasing per-copy contiguous block size when UB allows.
- Keeping GM/UB addresses aligned.
- Avoiding many tiny `DataCopy` calls.
- Using `DataCopyPad` for tails when appropriate.
- Avoiding unnecessary GM materialization of intermediates; consider fusion.
- Matching tile size with both compute utilization and memory bandwidth needs.

For vector Add-style operators, compare:

- smaller tiles: more overhead, lower bandwidth
- larger tiles: better bandwidth, but more UB use and fewer overlap opportunities

## Removing Debug and Scalar Overhead

Debug prints are expensive. Before profiling:

- Remove or guard `printf`.
- Remove `DumpTensor`.
- Avoid per-element scalar `GetValue` loops.
- Move static shape/tile calculations to Host tiling where possible.
- Keep Kernel scalar arithmetic out of inner loops.

Course example:

- `Init` contained tile-length and GM offset scalar arithmetic.
- `Process` contained loop control and `printf`.
- `CopyIn`/`CopyOut` contained address arithmetic.
- The biggest practical win was removing `printf`; minor tile calculations alone were not expected to dominate.

## Practice Target

Chapter 8 practice:

- Base operator: `AddCustomTemplate`.
- Input shape: `[45, 20480]`.
- Dtype: `float`.
- Goal: correct result and runtime under `80 us`.

Expected optimization levers:

- Balanced Tiling / `SetBlockDim`.
- Double Buffer when data size and UB allow.
- Larger copy chunks for bandwidth.
- Remove debug prints.
- Verify with msProf after each change.

## Common Optimization Traps

- Increasing core count blindly for small workloads.
- Creating many tiny tiles and increasing overhead.
- Enabling `BUFFER_NUM=2` without loop-level pipelining.
- Measuring debug builds or debug prints as production performance.
- Optimizing before correctness.
- Ignoring worst-core time and only looking at average time.
- Changing tiling without updating Kernel tail logic.
- Assuming simulator data exactly equals board data; use it for bottleneck direction and source-line insight.
