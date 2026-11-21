# Fast BP-DAT: a latency-optimised BP and BP-DAT for large 5G codes

This folder is the GPU-latency version of BP-DAT (`DualAttention_BP/`) and of the BP it runs
inside. It targets 5G NR BG1 (704,528) and (1408,1056). **It does not change any model.** It
runs the same trained weights through a faster schedule, and it decodes the same frames to the
same decisions (see *Equivalence*). The engine works for any `BP -> module -> BP` hybrid, so
BP-GNN and BP-SGT can use it too, and the benchmark measures all of them in one T4 session.

**Status: CPU-tested only.** No GPU run has happened yet, so none of the speed-ups below is
measured. They are what the design is meant to remove, based on the T4 profiles in
`Latency_Analysis/`. The `fastdat_t4` Kaggle kernel produces the numbers.

## What was slow (from `Latency_Analysis/`)

* At B = 1 on a T4, eager BP(50) takes 21-24 ms and the CUDA-graphed version 3.2-3.5 ms, so
  about 85% of the time is kernel-launch overhead. One SPA iteration issues ~31 torch ops.
* Every BP iteration of `bp_stage` / `bp_decode` reads `ok.any()` back to the host, then
  compacts the batch (index_select on every state tensor). That costs one host sync per
  iteration, and the changing shapes rule out CUDA graphs.
* Every scatter-sum is an `index_add_` (atomics).
* The learned module runs eagerly, with a few hundred small launches per call.

## What this folder does

| | original (`common/bp.py`, `staged_decode`) | fast (`fastdat/engine.py`) |
|---|---|---|
| early stop inside a BP run | compact the batch after every iteration (1 host sync / iteration) | converged frames are frozen with `torch.where`; the shape never changes |
| host syncs | 1 per BP iteration | 1 per block of `check_every` iterations (default 5) + 1 per stage boundary. `check_every=0, compact="none"`: **zero** |
| frames reaching the module | only unsolved frames (compacted) | only unsolved frames: compacted **once per stage boundary** into a power-of-two bucket (padding rows marked done), so a handful of fixed shapes |
| launches | ~31 per BP iteration + the module's ops | one CUDA graph replay per block (all graphs share one memory pool) |
| scatter-sums | `index_add_` | padded neighbour tables: gather + sum (`fastdat/layout.py`), deterministic, no atomics |
| DAT cross-attention softmax | `scatter_reduce(amax)` + exp + scatter_sum + gather | one `torch.softmax` over the padded [B, nodes, width, heads] scores (`fastdat/dat_fast.py`) |
| DAT module precision | fp32 | fp32 (equivalent), or fp16 autocast for the module only (BP stays fp32, since SPA underflows in fp16) |
| constant work | recomputed per call | node/edge type embeddings and RWSE computed once |

Optional: `compile=True` runs `torch.compile` on every block before it is captured.

## Equivalence (`tests/test_equivalence.py`, CPU)

On 5G (128,64), (704,528) and (1408,1056), with trained BP-GNN and BP-SGT checkpoints (from
the repo) and BP-DAT / DAT-NB with a randomised head (an untrained DAT is just BP with
restarts, which would make the test trivial). The test uses 64 frames at two Eb/N0.

1. **The engine with the original module and `agg="index_add"`** matches `staged_decode` /
   `bp_stage` for BP(50), BP-DAT, DAT-NB, BP-GNN and BP-SGT. It gives the same hard decision on
   every frame and the same iteration count. This holds with host checks every 1, 5 or 10
   iterations or none, and with compaction in power-of-two buckets, exact compaction or none.
   The LLRs differ only by float rounding (≤ 5e-4 relative). The original decoder is not
   bit-reproducible across batch sizes either: one frame decoded alone and the same frame in a
   batch of 64 differ by ~3e-3 after 50 iterations on CPU.
2. **The padded layout and fast DAT module** behave the same for FER:
   * the same frames converge, after the same number of iterations;
   * converged frames get the same decisions;
   * the same frames are in error.

   The module alone differs by < 2e-6 in the prior it outputs. The only differences are in
   frames BP never solves, whose messages oscillate (in BP(50) on 5G (128,64), 1 of 64 frames had
   other wrong bits; it was a frame error either way).
3. **Mixed precision is not equivalent.** bf16 on CPU changes the decision on some of the
   (random-head) frames. The T4 benchmark reports fp16's FER and per-frame differences next to
   fp32 on 40k frames, so the decision to use it can rest on measured FER.

## Benchmark (`bench_t4.py`, kernel `fastdat_t4`)

The benchmark runs as one process on one T4, with the rows interleaved over 2 rounds. It covers
5G (704,528) and (1408,1056):

| row | what it is |
|---|---|
| BP(50) | `common.bp.bp_decode`, as in every earlier table |
| BP(50) fast / fast fixed | the engine with no module; "fixed" = no host reads at all |
| BP-GNN, SGT | trained final checkpoints, original `decode` |
| BP-GNN fast engine, SGT fast engine | the same trained module inside the engine (the effect of the engine alone) |
| DAT | original `decode` |
| DAT fast fp32 | engine + fast module, the equivalent path |
| DAT fast fp16, fp16 fixed | module under fp16 autocast; "fixed" = no host reads |

For each row the benchmark reports:
* B = 1 mean, median and p95 over 200 distinct frames, with early stopping included, so this is
  what a codeword actually waits;
* throughput at B = 64, 1024 and 4096;
* FER and mean iterations on the same 40,960 frames at the profiling Eb/N0 and 0.5 dB below;
* for every fast row, the frames whose decision differs from the original and whether the same
  frames are in error;
* host reads per decode.

```
python FastDAT_BP/kaggle/build_kernel.py            # fastdat_smoke (CPU) and fastdat_t4 (T4)
kaggle kernels push -p FastDAT_BP/kaggle/fastdat_t4
kaggle kernels output gouravanirudh/fastdat-ldpc-t4 -p FastDAT_BP/results/kaggle_fastdat_t4
# -> REPORT.md (tables) and fastdat_bench.json
```

The BP-DAT rows use an **untrained** model (zero head = BP with restarts) until BP-DAT has been
trained by the `DualAttention_BP` final kernels. Fixed-schedule timing does not depend on the
weights, but early-stop timing and FER do. After those runs, rebuild with
`--dat-source gouravanirudh/dat-ldpc-final-a` (it trains 5G 1408; 5G 704 is in final_d on
another account). The checkpoint is then read from `/kaggle/input`.

Local runs: `python FastDAT_BP/tests/test_equivalence.py` (about 3 min) and
`python FastDAT_BP/bench_t4.py --quick --codes 5G_BG2_k64_n128` (CPU numbers mean nothing for
speed: a CPU has no launch overhead to remove, and the masked engine does more arithmetic).

## What to expect, and the trade-offs

* **B = 1, high SNR.** Most frames converge in the first BP run. The original pays one host
  round trip plus ~31 launches per iteration. The fast path pays one graph replay per 5
  iterations and one host read per block. This is where the 85% launch overhead goes.
* **B = 1, frame that needs the module.** Each module call becomes one replay instead of a few
  hundred launches. fp16 additionally halves the module's memory traffic.
* **Large batch.** Inside a BP run, converged frames keep being computed (masked) until the
  next stage boundary, where they are dropped. This is extra arithmetic compared with per-
  iteration compaction, bounded by the power-of-two rounding. If large-batch throughput comes
  out below the original, `check_every=1` or `compact="exact"` (eager) are the knobs to try.
* **Zero-sync mode** ("fixed") always runs 5x10 iterations on every frame. It is the right
  choice when the GPU is shared by a pipeline that must never block on the host, not when the
  goal is the lowest average latency.

## Files

| file | content |
|---|---|
| `fastdat/layout.py` | padded neighbour tables (checks -> edges, bits -> edges, checks -> bits) and the gather-sum / syndrome helpers |
| `fastdat/engine.py` | `StagedEngine`: masked early stop, block-wise CUDA graphs, stage-boundary compaction, optional torch.compile |
| `fastdat/dat_fast.py` | `fast_dat_refine(model, layout, dtype)`: the BP-DAT module (DAT and DAT-NB) with padded ops and optional fp16 |
| `fastdat/models.py` | loads the trained BP-GNN / BP-SGT / BP-DAT checkpoints |
| `tests/test_equivalence.py` | CPU equivalence tests above |
| `bench_t4.py` | the one-session benchmark |
| `kaggle/build_kernel.py` | builds `fastdat_smoke` and `fastdat_t4`; bundles the 4 trained checkpoints |
