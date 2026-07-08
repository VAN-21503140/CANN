# FastGelu CANNJudge Optimization Log

This repository tracks the FastGelu custom CANN operator solution and every important optimization version. Good online results are saved as git commits instead of overwriting history.

## Current Best

Current best version:

- Commit: current commit / `manual_v9_aiv_core_cap24`
- CANNJudge submission: `86658`
- Result: Pass `5/5`
- Times: `3.26 / 4.36 / 6.22 / 6.32 / 7.88 us`
- Sum: `28.04 us`

## Version History

| Commit | Version | Main Change | Submission | Times (us) | Sum (us) | Result |
| --- | --- | --- | --- | --- | ---: | --- |
| `7a93dcb` | Baseline accepted solution | Initial working FastGelu CANN operator from handoff | `83644` | `4.72 / 4.72 / 9.08 / 8.56 / 12.80` | `39.88` | Pass 5/5 |
| `4bee58e` | Numerator buffer reuse | Removed a numerator temporary buffer and computed numerator into output local tensor | - | `4.80 / 4.84 / 9.18 / 8.18 / 13.20` | `40.20` | Pass 5/5 |
| `456c040` | Sigmoid formula | Replaced stable FastGelu formula with `y = x * sigmoid(1.702x)` | - | `4.68 / 4.72 / 8.88 / 8.02 / 11.70` | `38.00` | Pass 5/5 |
| `53ac24a` | V4 generalized tiling | Used 32B block-based generalized tiling with big/small core fields, original stable formula | `85000` | `4.84 / 4.80 / 7.84 / 7.22 / 10.08` | `34.78` | Pass 5/5 |
| `47e8e5a` | V3 simple tiling | Restored simple V3 tiling: small inputs use fewer cores, large inputs use max AIV cores | `85014` | `4.18 / 3.76 / 8.06 / 7.32 / 9.62` | `32.94` | Pass 5/5 |
| `1f41134` | V3 + sigmoid | Combined V3 tiling with `x * sigmoid(1.702x)` | `85028` | `3.92 / 3.74 / 6.98 / 6.46 / 8.70` | `29.80` | Pass 5/5 |
| `12525cc` | V5 hybrid generalized sigmoid | V3 small-input core policy plus 32B generalized big/small core distribution and 4096-element sigmoid tiles | `85101` | `3.52 / 3.90 / 6.72 / 7.38 / 8.18` | `29.70` | Pass 5/5 |
| `a31de5d` | V6 dtype-aware tile | Float32 keeps 4096 tile; float16 uses 8192 tile to reach about 16KB copy chunks | `85216` | `4.14 / 3.38 / 6.40 / 6.92 / 7.92` | `28.76` | Pass 5/5 |
| current commit | V7 lower large-core threshold | Kept V6 dtype-aware tiles and lowered `LARGE_CORE_THRESHOLD` from `CORE_SPLIT_ELEM_NUM * 4` to `* 3` | `85408` | `4.22 / 3.38 / 6.32 / 6.66 / 7.98` | `28.56` | Pass 5/5 |
| `e6b2639` (`experiment/fastgelu-thr2p5-85442`) | V7 threshold interpolation | Same V7 dtype-aware tiles, but interpolated `LARGE_CORE_THRESHOLD` to `CORE_SPLIT_ELEM_NUM * 5 / 2`; kept on an experiment branch while `main` stays on the best `* 3` version | `85442` | `3.96 / 3.22 / 6.52 / 6.64 / 8.32` | `28.66` | Pass 5/5 |
| current commit | V8 remove final vector barrier | Kept V7 tiling and removed the final `PipeBarrier<PIPE_V>()` after `Mul` before `EnQue` | `86226` | `3.08 / 4.10 / 6.26 / 6.82 / 7.90` | `28.16` | Pass 5/5 |
| current commit | V9 cap AIV cores at 24 | Kept the V8 kernel and capped host-side `max_core_num` at the physical AI Core count after confirming the judge machine has 24 AI Cores and 48 Vector units | `86658` | `3.26 / 4.36 / 6.22 / 6.32 / 7.88` | `28.04` | Pass 5/5 |

Documentation and automation commits:

- `981229c`: saved full problem requirements in `FASTGELU_REQUIREMENTS.md`
- `7812e3b`: added the first allnight optimization pipeline
- `8a3c728`: refined candidate ordering using CANN performance heuristics
- `414a997`: added supervised review gate and history-aware adaptive iteration
- `182c0f8`: updated allnight baseline after the V6 win

## Optimization Notes

Main lessons so far:

- Formula simplification to `x * sigmoid(1.702x)` is a large win.
- V3's small-input core policy is better for some small/mid tests than blindly opening many cores.
- Generalized 32B big/small core tiling helps larger cases, but can hurt some tests if the tile size is not matched to dtype.
- Lowering the large-core threshold from `CORE_SPLIT_ELEM_NUM * 4` to `* 3` gave the best threshold-only total. The `* 5 / 2` interpolation improved tests 1, 2, and 4 versus `* 3`, but regressed tests 3 and 5 enough to land slower overall (`28.66 us` vs `28.56 us`).
- The final vector barrier after `Mul` was not needed before `EnQue` for correctness on CANNJudge and removing it improved total runtime.
- The judge hardware reports more AIV capacity than this operator should shard across. Capping host-side AIV cores at `24`, matching the physical AI Core count, beat uncapped V8 even though tests 1 and 2 became slower; tests 3, 4, and 5 improved enough to make the total best so far.
- Based on the CANN performance skill, single copy chunks around `16KB` are a useful target:
  - float32 `4096` elements = `16KB`
  - float16 `8192` elements = `16KB`
- `BUFFER_NUM=2` only helps if the loop is actually pipelined. The current kernel is still mostly serial `CopyIn -> Compute -> CopyOut`, so buffer-depth experiments should be supervised.

## Supervised Pipeline Workflow

Inspect the next candidate plan without touching the web page:

```powershell
d:\py\Anaconda3\python.exe code\tools\allnight_fastgelu_pipeline.py --explain-plan --max-candidates 3
```

Generate one candidate, run static checks, and stop before online submission:

```powershell
d:\py\Anaconda3\python.exe code\tools\allnight_fastgelu_pipeline.py --review-gate --max-candidates 1 --run-name review_next
```

After reviewing the generated diff, submit one candidate at a controlled pace:

```powershell
d:\py\Anaconda3\python.exe code\tools\allnight_fastgelu_pipeline.py --submit --commit-improvements --stop-after-improvement --max-candidates 1 --run-name submit_next
```

The pipeline defaults to a `900s` cooldown between online submissions when more than one candidate is run, to avoid submitting too frequently. While polling a Running submission, it can periodically refresh the result page so completed results show up sooner.

## Repository Policy

- Keep every meaningful optimization as a separate git commit.
- Do not overwrite historical versions.
- Push good versions to GitHub so results are visible from multiple machines.
- Record online submission ID, times, and the optimization hypothesis whenever a version improves.
