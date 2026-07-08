# FastGelu remove Sigmoid-to-Mul barrier experiment

This branch records the CANNJudge submission that removes the vector barrier between `Sigmoid` and `Mul`, in addition to the V8 final-barrier removal.

## Candidate

- Name: `manual_v8_remove_sigmoid_mul_barrier`
- Baseline: V8 `manual_v7_remove_final_pipebarrier`
- Code change: remove `PipeBarrier<PIPE_V>()` between `Sigmoid(sigmoidLocal, sigmoidLocal, count)` and `Mul(yLocal, xLocal, sigmoidLocal, count)`.

## CANNJudge result

- Submission ID: `86290`
- Status: Pass `5/5`
- Times: `3.28 / 4.26 / 6.66 / 6.80 / 7.88 us`
- Sum: `28.88 us`

## Interpretation

The candidate passed correctness, but it was slower than V8. This suggests the `Sigmoid` to `Mul` dependency barrier remains useful for scheduling or correctness-adjacent pipeline behavior, while the final barrier after `Mul` before `EnQue` can be removed.

Current best remains V8 on `main`:

- Submission ID: `86226`
- Times: `3.08 / 4.10 / 6.26 / 6.82 / 7.90 us`
- Sum: `28.16 us`
