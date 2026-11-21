# Latency analysis of BP and BP-SGT (task 3)

Goal: find exactly where BP (`Nvidia_GNN+BP(lLatest)/common/bp.py`) and the BP-SGT hybrid
(`SparseGT_BP/sgt/sgt_decoder.py`) spend their time on a GPU, component by component, and
test which changes remove each bottleneck. Everything is measured in **one process on one
T4** so that all rows are comparable (the earlier profile rows came from four different
Kaggle sessions).

Nothing here changes the decoders used for the published results. The instrumented copies
run the same tensor operations in the same order (checked by `tests/test_equivalence.py`),
and the faster variants are experiments that live only in this folder.

## Layout

| file | what it does |
|---|---|
| `run_latency.py` | the whole analysis; writes `<code>.json`, `REPORT.md`, chrome traces |
| `lat/regions.py` | named timing regions: `sync` mode (wall time per region) and `profiler` mode (GPU kernel time, kernel launches, torch ops, host syncs per region from torch.profiler) |
| `lat/bp_inst.py` | instrumented `bp_decode`, `bp_stage`, `staged_decode` |
| `lat/sgt_inst.py` | instrumented SGT `refine`, plus cheaper but equivalent ways to build the attention bias |
| `lat/bp_fast.py` | BP ablations: early stopping without a host sync, CUDA graphs, scatter backends, fused sign, min-sum, fp16, torch.compile |
| `cost_model.py` | analytic MACs / memory traffic of one SGT stage per component (CPU, seconds) |
| `tests/test_equivalence.py` | CPU check that the instrumented and variant decoders give the original outputs |
| `kaggle/build_kernel.py` | builds the self-contained Kaggle kernels (`latency_t4`, `latency_smoke`) |
| `results/cost_model.md` | output of `cost_model.py` |

## How to run

```bash
# CPU, seconds: equivalence tests and a smoke run (numbers meaningless on CPU)
python Latency_Analysis/tests/test_equivalence.py
python Latency_Analysis/run_latency.py --quick --codes 5G_BG2_k64_n128

# Kaggle T4 (account gouravanirudh; reads the trained checkpoints of "SGT LDPC Final A")
python Latency_Analysis/kaggle/build_kernel.py
kaggle kernels push -p Latency_Analysis/kaggle/latency_smoke      # optional, ~5 min
kaggle kernels push -p Latency_Analysis/kaggle/latency_t4         # full run, roughly 45-60 min
kaggle kernels output gouravanirudh/ldpc-latency-t4 -p Latency_Analysis/results/kaggle_latency_t4
```

The full run covers ccsds_128_64 and 5G_BG1_k1056_n1408 (trained checkpoints from Final A)
and mackay_204_102 and wimax_576_288 (untrained). Fixed-stage timing does not depend on the
weights. Early-stop SGT timing does, so for the untrained codes it only shows BP with restarts.
To pick other codes, change `CODES` in `kaggle/build_kernel.py` and rebuild. Use any code name
from the final comparison.

## What each section of the report answers

| section | question |
|---|---|
| `bp_iters` | Fixed-iteration BP with T in {1, 10, 50, 200}, eager and CUDA graph, B in {1, 64, 1024, 4096}. Fits latency = fixed cost + T x per-iteration cost. Eager minus graph is the kernel-launch overhead. |
| `bp_regions` | One BP decode split into components: check-node update (tanh/log, magnitude scatter, sign scatter, exp/atanh), variable-node scatter, v2c gather, write-back, syndrome, **host sync** (`ok.any()`), batch compaction. It reports sync-mode wall time, GPU kernel time, kernel launches, torch ops and host syncs (from torch's sync debug mode). |
| `bp_variants` | One change at a time, on the same frames, with FER and mean iterations so that no speed-up comes from decoding worse. The changes are: masked early stop (no `.any()` per iteration), host check every 1/5/10/T iterations, CUDA graph per block of iterations, fused sign+magnitude aggregation, scatter backend (index_add / padded gather-sum / dense matmul / CSR), torch.compile (fusion), min-sum fp32 and fp16. |
| `agg_micro` | The scatter-sum alone, per backend, side and batch size. |
| `sgt_refine` | One SGT stage on hard frames, split into features, top-k, tokens, **bias build** (`Hk` gather, `shared` bmm, `shared_unsat` bmm, embedding), attention, FFN and head. It runs four variants: `orig`, `gather` (shared-check counts gathered from a precomputed H Hᵀ, no k·k·m bmm), `gather16` (also the unsatisfied-count bmm in fp16 on tensor cores, exact for 0/1 operands), and `nobias` (timing only, because it changes the output). |
| `sgt_k` | One SGT stage versus k_frac in {0.05 ... 0.5}: how the bias build and attention scale with k. |
| `sgt_decode` | The full 5x10 BP-SGT decode split into BP stages, SGT stages (with the bias share) and compaction/sync/write-back. It runs fixed and early-stop, on random and on hard frames, with the active frames per stage. |

How to read the numbers: **plain** is the uninstrumented decoder. **Sync-mode** synchronises at
every region boundary, so each region's time includes its own launch overhead. The sum of the
regions is therefore larger than plain. **GPU kernel ms** is pure kernel time. When it is much
smaller than the sync-mode time, the region is bound by launches and host syncs rather than by
arithmetic.

## What the existing T4 numbers already show (before this run)

These rows were derived from the clean single-process profiles in
`SparseGT_BP/results/kaggle_profile_{a,b,c,d}` (four Kaggle sessions, so differences under about
20% between codes are not reliable). One SGT stage = (SGT 5x10 minus BP(50)) / 4, because both
run 50 BP iterations. "Per frame at max B" uses the throughput rows.

| code | BP(50) eager B=1 | BP(50) CUDA graph B=1 | launch overhead | BP(50) early-stop B=1 | SGT stage, B=1 graphed | SGT stage per frame at max B | modelled MMAC per stage | effective TMAC/s | SGT/BP throughput, fixed | same, early-stop |
|---|---|---|---|---|---|---|---|---|---|---|
| mackay_96_48 | 24.5 ms | 3.56 ms | 85% | 0.99 ms | 0.29 ms | 18 us (B=4096) | 4.0 | 0.22 | 0.158 | 0.71 |
| ccsds_128_64 | 23.8 ms | 3.19 ms | 87% | 2.92 ms | 0.50 ms | 30 us (B=4096) | 5.8 | 0.19 | 0.163 | 0.73 |
| 5G_BG2_k64_n128 | 21.0 ms | 3.27 ms | 84% | 3.93 ms | 0.53 ms | 43 us (B=4096) | 7.5 | 0.17 | 0.113 | 0.71 |
| mackay_204_102 | 21.4 ms | 3.28 ms | 85% | 3.33 ms | 0.63 ms | 58 us (B=4096) | 11.5 | 0.20 | 0.110 | 0.66 |
| 5G_BG2_k128_n256 | 21.3 ms | 3.41 ms | 84% | 11.34 ms | 0.79 ms | 133 us (B=2048) | 24.1 | 0.18 | 0.077 | 0.52 |
| mackay_408_204 | 22.3 ms | 3.29 ms | 85% | 2.76 ms | 0.85 ms | 208 us (B=2048) | 41.1 | 0.20 | 0.067 | 0.66 |
| wimax_576_288 | 23.3 ms | 3.36 ms | 86% | 7.10 ms | 1.00 ms | 356 us (B=1024) | 88.1 | 0.25 | 0.056 | 0.59 |
| 5G_BG2_k256_n512 | 22.2 ms | 3.42 ms | 85% | 5.26 ms | 0.99 ms | 427 us (B=1024) | 93.4 | 0.22 | 0.051 | 0.29 |
| 5G_BG1_k528_n704 | 22.3 ms | 3.49 ms | 84% | 3.57 ms | 1.25 ms | 621 us (B=1024) | 133.9 | 0.22 | 0.054 | 0.60 |
| 5G_BG1_k1056_n1408 | 23.4 ms | 3.50 ms | 85% | 5.11 ms | 3.17 ms | 2617 us (B=512) | 768.0 | 0.29 | 0.027 | 0.21 |

Facts from these rows, followed by the hypotheses this run tests:

1. **BP at B=1 is launch-bound, not compute-bound.** Eager BP(50) costs 21 to 24 ms on every code,
   from n=96 to n=1536, and a CUDA graph of the same work costs 3.2 to 3.5 ms. So about 85% of the
   eager time is kernel-launch and Python overhead. Code size barely matters at B=1. Counted here on
   CPU, one SPA iteration issues **31 torch ops** (9 of them only for the sign product), which gives
   about 0.45 to 0.5 ms per iteration eager against about 0.07 ms graphed.
2. **Early stopping is paid with a host sync per iteration.** `bp_decode` calls `ok.any()` (and
   `keep.any()`) every iteration, and each call blocks the CPU until the GPU catches up. That defeats
   the asynchronous launch queue, and it is why BP(50) and BP(200) have the same B=1 latency: both
   stop after a few iterations, and each iteration costs overhead, not arithmetic. On 5G n=256,
   early-stop BP(50) at B=1 (11.3 ms) is slower than the fixed graphed BP(50) (3.4 ms). Test:
   `bp_variants` (masked early stop with a check every K iterations, and CUDA-graphed blocks).
3. **At large batch BP should be bound by memory passes.** The 31 ops per iteration each read and
   write E-sized tensors. Test: the fused-sign, scatter-backend and torch.compile variants, plus the
   GPU kernel time per region.
4. **The SGT stage time per frame tracks the modelled MACs** at a nearly constant 0.17 to 0.29 TMAC/s
   over a 200x range of cost (about 5% of the T4's 8 TFLOPS fp32 peak). In the model, the dense
   Tanner-graph bias build (two `[B, k, m] x [B, m, k]` bmm with k = 0.5n) is 74% of the MACs on
   5G n=1408 and 53 to 57% on WiMAX, 5G n=512 and 5G n=704 (`results/cost_model.md`). The
   prediction is that the bias build dominates SGT time on long codes and attention dominates on
   short ones. Test: `sgt_refine` and `sgt_k` with the `gather`/`gather16` variants.
5. **With early stopping the SGT costs much less**, because only BP failures reach it (throughput
   ratio 0.5 to 0.7 of BP(50) instead of 0.03 to 0.16). The exception is long codes at low SNR
   (5G n=1408: 0.21), where many frames reach the SGT.

## Caveats

* The profiler attribution (kernel time and launches per region) could only be exercised on CPU
  in this container. If it shows zeros on the GPU, the sync-mode table, the `top_ops` table in the
  JSON and the chrome traces in `traces/` still hold the data.
* `bp_fast.py` variants are experiments for locating the cost. They are not a drop-in decoder yet.
  The padded and dense backends change the summation order (LLR differences up to about 0.03, with
  the same hard decisions in the tests). Min-sum changes the algorithm, and its FER is reported
  next to its time.
* `--quick` numbers on CPU are meaningless. They only show that the pipeline runs.
