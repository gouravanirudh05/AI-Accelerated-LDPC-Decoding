# BP-SGT schedule study (team tasks 1, 2, 4, 5)

Experiments on the interleaved BP+SGT pipeline of `SparseGT_BP/`
(`llr -> BP(T) -> SGT -> BP(T) -> ... -> BP(T)`, final models trained as 5x10,
k = 0.5 n, weights shared across SGT stages). One folder per task:

| folder | task | needs a GPU run? |
|---|---|---|
| `task1_bp_schedules/` | 1. different BP iteration schedules (10, 5, 2 iterations per run, and equal-budget variants) | yes (inference only, no training) |
| `task2_stopping_points/` | 2. distribution of the stopping points across the pipeline stages | no, plots from task 1 / task 5 traces |
| `task4_avg_bp_iterations/` | 4. average total number of BP iterations | no, tables from task 1 / task 5 traces |
| `task5_single_vs_interleaved/` | 5. single-stage BP+SGT vs interleaved BP+SGT | yes (optional 80 min training per code) |

Tasks 3 (latency / bottlenecks) and 6 (SGT block diagram) are handled separately.

## How it works

`core.py` re-implements `common.bp.staged_decode` with **a separate number of BP
iterations per run** and **a per-frame trace** of where the frame stopped (BP run,
iteration inside the run, total BP iterations, number of SGT calls). For the 5x10
schedule it is bit-identical to `SparseGTBPDecoder.decode` (same LLRs and iteration
counts on 2000 frames of MacKay (204,102) and 5G BG2 (128,64)); for a single run
with no refinement it equals `common.bp.bp_decode`, so BP baselines go through the
same code. Early stopping is the project's usual rule: a frame leaves as soon as its
hard decision satisfies every check, so SGT stages only run on frames BP failed.

Every Monte Carlo script writes `traces_<code>.json` (BLER, BER, mean/std BP
iterations, mean SGT calls, frames stopping in each run split into correct and
wrong codeword, never converged, histogram of total iterations at the stop). The
task 2 and task 4 scripts only read those files (numpy + matplotlib, no torch).

The trained checkpoints are the final BP-SGT models already in the repo
(`SparseGT_BP/results/kaggle_final_*/results/*/*_sgt.pt`); their kwargs are stored
in the checkpoint, so the missing `final/sgt_config.json` is not needed.

Schedule notation: `RxT` = R BP runs of T iterations (R-1 SGT stages); `a-b-c` =
runs of a, b and c iterations. Eb/N0 points: 4 to 7 points across each code's
training band (`final/code_bands.json`); override with `--ebno`.

## Running on Kaggle

```
python SparseGT_BP/schedule_study/kaggle/build_kernel.py            # writes kaggle/<name>/
KAGGLE_API_TOKEN=... kaggle kernels push -p SparseGT_BP/schedule_study/kaggle/smoke      # CPU, ~1 min
KAGGLE_API_TOKEN=... kaggle kernels push -p SparseGT_BP/schedule_study/kaggle/sched_a    # tasks 1, 2, 4
KAGGLE_API_TOKEN=... kaggle kernels push -p SparseGT_BP/schedule_study/kaggle/single_a   # task 5 (+ 2, 4)
```

Kernels `sched_a..d` and `single_a..d` use the same code split and accounts as the
BP-SGT final runs (A: MacKay 96, CCSDS, 5G 1408; B: MacKay 204, 5G 128, 5G 256;
C: MacKay 408, 5G 512; D: WiMAX, 5G 704), two GPU queues each. The checkpoints of
the kernel's codes are embedded in the script (no Kaggle input needed). After the
queues, the plot / table scripts run once; outputs are in
`/kaggle/working/results/schedule_study/{task1_bp_schedules, task2_stopping_points/task1,
task4_avg_bp_iterations/task1, task5_single_vs_interleaved, task2_stopping_points/task5,
task4_avg_bp_iterations/task5}`.

Budget per (decoder, Eb/N0 point): >= 100 frame errors, or the code's frame budget
(0.5 M to 3 M), or 90 s, whichever comes first (`--min-errors`, `--max-frames`,
`--max-seconds`). Upper bound per kernel queue for `sched_*`: 10 decoders x up to 7
points x 90 s = 1.75 h per code, so the longest queue (MacKay 96 + CCSDS, or 5G 128 +
5G 256) is at most about 3.5 h; most high-SNR points of BP-SGT hit the 90 s cap first.
`single_*`: 80 min training + 8 decoders per code, at most about 4.5 h per queue.

## Running locally (smoke test, seconds)

```
cd SparseGT_BP/schedule_study
python task1_bp_schedules/run_schedules.py --quick --codes mackay_204_102
python task1_bp_schedules/plot_schedules.py --traces out/task1_bp_schedules
python task2_stopping_points/plot_stopping_points.py --traces out/task1_bp_schedules
python task4_avg_bp_iterations/avg_iterations.py --traces out/task1_bp_schedules
python task5_single_vs_interleaved/run_single_vs_interleaved.py --quick --codes mackay_204_102 --train-minutes 0.3
python task5_single_vs_interleaved/compare.py --traces out/task5_single_vs_interleaved
```

## Caveats to state with the results

* Task 1 is inference-only: the SGT was trained on BP states after 10 iterations.
  A schedule with 2 or 5 iterations per run feeds it states it never saw, so a loss
  against 5x10 measures "schedule + distribution shift", not the schedule alone.
  In the CPU smoke test (a few hundred frames, not a result) 5x2 and 25x2 were much
  worse than BP(10) / BP(50), which is what this shift would look like; retraining
  with `bp_iters=2` (via `sgt.specs.sgt_spec` overrides) would separate the two.
* Task 5 "single-stage, interleaved weights" has the same caveat (BP(25) states);
  the trained single-stage model (`--train-minutes 80`) is the like-for-like
  comparison. It mines hard examples on BP(25) failures (the first BP run), as the
  interleaved model mined on BP(10) failures.
* Average BP iterations count BP iterations only; SGT calls are reported
  separately (they are not free: 11.5 M MACs per call on MacKay 204, 768 M on 5G 1408).
* Noise rule of the project: about +-10 % on a BLER point with 100 errors, +-22 % with
  20; ratios are only computed where both decoders have >= 20 errors.
