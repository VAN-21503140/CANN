# FastGelu threshold 2.5 experiment

This branch records the CANNJudge submission shown on the web page after the V7 threshold search.

## Candidate

- Name: `manual_dtype_f4096_h8192_thr2p5_buf2`
- Code change: keep V6/V7 dtype-aware tiles and set `LARGE_CORE_THRESHOLD` to `CORE_SPLIT_ELEM_NUM * 5 / 2`.
- Compared baseline: V7 best on `main`, `manual_dtype_f4096_h8192_thr3_buf2`.

## CANNJudge result

- Submission ID: `85442`
- Status: Pass `5/5`
- Times: `3.96 / 3.22 / 6.52 / 6.64 / 8.32 us`
- Sum: `28.66 us`

## Interpretation

This candidate improved test 2 relative to V7, but regressed tests 3 and 5 enough that the total was worse.

Current best remains V7 on `main`:

- Submission ID: `85408`
- Times: `4.22 / 3.38 / 6.32 / 6.66 / 7.98 us`
- Sum: `28.56 us`
