# Trade-off study: unrolled neural BP vs chained BP->GNN vs end-to-end BP-GNN

Kaggle Tesla T4, account `gourasri`, kernels `gnn-ldpc-tradeoffs-normal` / `-5g`.
Raw data: `kaggle_tradeoffs_normal/`, `kaggle_tradeoffs_5g/` (JSON, plots, logs, checkpoints).
Every model trained for 15 min on one code: MacKay (204,102), or 5G BG2 (k=64, n=128).
BLER from up to 3M frames per point (stop at 100 frame errors).

Decoders: BP-GNN = paper's method, BP(10) -> GNN -> ... -> BP(10), 4 GNN stages, gradients through
all BP runs. BP-GNN detached-BP = "chained" variant (BP without autograd, each GNN trained on its own
output). NBP = unrolled weighted BP, 10 iterations, per-edge or shared weights.

## Claim 1 - "chained hybrid has low GPU memory, unrolled neural BP high": FALSE (reversed)

Training step at batch 256, peak allocated memory:

| decoder | MacKay (204,102) | 5G BG2 (64,128) | max trainable batch (MacKay / 5G) |
|---|---|---|---|
| NBP per-edge / shared | 68 MB | 52 MB | 32768 / >=65536 |
| BP-GNN end-to-end | 1826 MB | 1399 MB | 2048 / 2048 |
| BP-GNN detached-BP | 1574 MB | 1208 MB | 2048 / 2048 |
| GNN (pure) | 2035 MB | 1554 MB | 1024 / 2048 |

Unrolled BP stores one value per edge per iteration and is about 27x lighter. GNN layers store 48
hidden values per edge, and they dominate. Detaching BP saves only about 14% and does not raise the
largest batch that fits.

## Claim 2 - "hybrid adapts across code shapes, neural BP is rigid": PARTLY TRUE

* Per-edge NBP cannot be applied to another code at all (by construction).
* Shared NBP (10 scalar weights) transfers fine: on MacKay (408,204) at 4.0 dB BLER is 1.2e-4,
  versus 2.3e-4 for BP(10).
* BP-GNN trained on (204,102) transfers to the same (3,6)-regular family:
  * (408,204) at 4.0 dB: 1.2e-5, versus 1.5e-5 for BP(50)
  * (96,48) at 5.0 dB: 1.2e-4, versus 1.5e-4 for BP(50)
  * so it works, but the gain is small.
* 5G, lifting size Z changed: BP-GNN trained on (64,128) is slightly worse than BP(50) on the
  longer codes:
  * (128,256) at 3.0 dB: 1.6e-3 versus 1.2e-3
  * (256,512) at 3.0 dB: 4.4e-5 versus 2.0e-5
* Transfer works for the GNN-based decoders and for shared NBP alike. It depends on how parameters
  are indexed, not on BP versus GNN.

## Claim 3 - "hybrid resolves the error floor better": MODEST GAIN, CODE DEPENDENT

At the same budget of 50 BP iterations, on the training code MacKay (204,102):

| Eb/N0 | BP(50) | BP-GNN | BP-GNN detached | BP + random perturb. | NBP (10) | BP(50)+OSD-0 |
|---|---|---|---|---|---|---|
| 4.0 dB | 3.8e-4 | 2.5e-4 | 3.8e-4 | 7.7e-4 | 1.2e-3 | 1.9e-4 |
| 4.5 dB | 6.0e-5 | **3.1e-5** | 6.9e-5 | 1.3e-4 | 1.9e-4 | 1.6e-5 (8 err) |
| 5.5 dB | 3.7e-6 (11 err) | 1.0e-6 (3 err) | 5.7e-6 | 2.0e-6 | 8.7e-6 | 2.0e-6 (1 err) |

* BP-GNN lowers BLER by about 2x versus BP(50) at high SNR (4.5 dB, ~100 errors each). It also beats
  random perturbation, matching the paper's comparison.
* End-to-end training through BP matters. The detached ("chained") version is no better than BP(50)
  at high SNR, the opposite of the table's claim.
* OSD-0 post-processing is still better at high SNR, but much more expensive per failed frame.
* 5G BG2 (64,128): no gain over BP(50); for example 4.5 dB is 9.0e-5 versus 6.3e-5.
* No distinct error floor is visible down to about 1e-6 for these codes, so deeper simulation or
  importance sampling would be needed to judge the floor itself.

## Inference cost (MacKay (204,102), 3.0 dB, T4)

| decoder | B=1 latency | throughput B=4096 | energy/codeword (total / above idle) |
|---|---|---|---|
| BP(50), early stop | 2.1 ms | 83k cw/s | 1361 / 124 uJ |
| BP-GNN, early stop | 2.4 ms | 58k cw/s | 2523 / 766 uJ |
| BP-GNN, fixed 50 it + 4 GNN | 6.1 ms (CUDA graph) | 3.8k cw/s | 16143 / 8027 uJ |
| NBP (10), fixed | 0.8 ms (CUDA graph) | 163k cw/s | 319 / 160 uJ |
| BP(10), fixed | 0.7 ms (CUDA graph) | 169k cw/s | 303 / 152 uJ |

With early stopping the GNN only runs on frames BP could not solve, so BP-GNN costs about 1.9x the
total energy of BP(50); without early stopping it costs about 11x.

## Caveat: pure GNN on 5G

The pure GNN (10 it) failed to train on 5G BG2: BLER 1.0 and BER about 0.2 at all SNRs. The 2Z
punctured information bits have no channel observation, and the GNN never learned to recover them.
BP and BP-GNN recover them through the parity checks. This is a training failure of this pure-GNN
setup, not a general property of GNN decoders: Cammerer et al. 2022 report working 5G GNN decoders.
