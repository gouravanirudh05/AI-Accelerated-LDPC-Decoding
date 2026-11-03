# GNN-enhanced decoding for classical and 5G NR LDPC codes

Adaptation of *A. Gong, S. Cammerer, J. M. Renes, "Graph Neural Networks for
Enhanced Decoding of Quantum LDPC Codes" (NVIDIA Research, 2023)* to classical
binary LDPC codes over the BPSK/AWGN channel.

## Layout

```
GNN_LDPC_Decoding/
  common/            shared implementation
    bp_gnn_decoder.py  BP-GNN hybrid decoder (the paper's method)
    gnn_decoder.py   pure GNN decoder (predecessor baseline) + shared GNN blocks
    bp.py            BP (differentiable + inference), staged decoding, random perturbation
    neural_bp.py     unrolled weighted BP (per-edge or shared weights), trade-off baseline
    osd.py           batched GPU OSD-0 post-processing (GNN+OSD, BP+OSD)
    gf2.py graph.py channel.py code.py   GF(2), Tanner graph, AWGN, code objects
    train.py evaluate.py profiling.py experiment.py report.py
  normal_ldpc/       standard benchmark LDPC codes (alist files in codes/)
    run_normal_ldpc.py
  5g_ldpc/           3GPP TS 38.212 NR LDPC (BG1/BG2 tables in codes/)
    nr_ldpc.py       base graph, lifting, filler bits, puncturing, rate matching
    run_5g_ldpc.py
  tradeoffs/         run_tradeoffs.py: neural BP vs chained (detached-BP) vs end-to-end BP-GNN;
                     GPU memory, cross-code / cross-Z transfer, deep Monte Carlo at high SNR
  kaggle/            build_kernel.py -> self-contained Kaggle kernels
  results/           outputs downloaded from Kaggle (JSON, plots, SUMMARY.md)
```

## Method: BP-GNN hybrid (the paper)

```
llr -> BP(10) -> GNN -> BP(10) -> GNN -> BP(10) -> GNN -> BP(10) -> GNN -> BP(10)
```

* Classical BP stages with GNN layers in between, both on the same Tanner graph.
  When a BP run is stuck (trapping sets, short cycles), the GNN reads its state
  and produces a better initialisation (prior LLRs) for the next BP run.
* GNN layers: d-dimensional VN / CN embeddings, MLP messages of (sender,
  receiver, edge), degree-normalised sum aggregation, residual node updates;
  weights shared across the BP-GNN stages.
* Fully differentiable (gradients flow through all BP runs), trained end to end
  with a loss after every BP stage.
* Compared, as in the paper, against BP restarts with random perturbation and
  against OSD post-processing (and BP-GNN + OSD).

Baseline from the predecessor paper (Cammerer et al. 2022, "GNN for channel
decoding"): a pure GNN decoder (10 learned message-passing iterations, no BP).

## Adaptations to classical / 5G LDPC

1. **Soft channel input.** The quantum decoder only has a syndrome under
   depolarizing noise; here the GNN gets the channel LLRs, the prior of the
   stuck BP run, its marginals, the BP messages (v2c, c2v) as edge features
   and the soft/hard syndrome on the check nodes.
2. **Loss.** BCE on the marginals of every BP stage that follows a GNN, plus a
   soft-syndrome term -log P(check satisfied) (classical counterpart of the
   paper's syndrome/degeneracy-aware loss).
3. **Hard-example mining.** Training batches are filled with frames on which
   plain BP(10) fails, the only frames on which the GNN acts at inference.
4. **Residual prior** prior = llr + f(h_v), zero-initialised, so an untrained
   hybrid is exactly BP with restarts.
5. **Type embeddings.** Classical codes: VN/CN degrees. 5G: protograph-aware
   embeddings indexed by base-graph column / row / edge, which do not depend on
   the lifting size Z, so one model decodes several block lengths.
6. **Early stopping with batch compaction** after every BP iteration: decoded
   frames leave the batch, so the GNN only costs time/energy on hard frames.
   OSD-0 runs batched on the GPU only for frames that still fail.

## Codes

| folder | code | n | k | source |
|---|---|---|---|---|
| normal_ldpc | MacKay 96.33.964 | 96 | 48 | MacKay's encyclopedia |
| normal_ldpc | CCSDS | 128 | 64 | RPTU channel-codes DB |
| normal_ldpc | MacKay 204.33.484 | 204 | 102 | MacKay's encyclopedia |
| normal_ldpc | MacKay 408.33.844 | 408 | 204 | MacKay's encyclopedia |
| normal_ldpc | WiMAX 802.16e R=1/2 | 576 | 288 | RPTU channel-codes DB |
| 5g_ldpc | NR BG2, R=1/2 (train) | 128 | 64 | 38.212 |
| 5g_ldpc | NR BG2, R=1/2 | 256 / 512 | 128 / 256 | 38.212 (unseen during training) |
| 5g_ldpc | NR BG1, R=3/4 (train) | 704 | 528 | 38.212 |
| 5g_ldpc | NR BG1, R=3/4 | 1408 | 1056 | 38.212 (unseen during training) |

The 5G base-graph CSVs come from NVIDIA Sionna (Apache-2.0).

## Baselines and measurements

* BP-GNN (5 x 10 BP iterations + 4 GNN stages), BP-GNN + OSD-0, pure GNN (10 it) and GNN + OSD-0.
* BP sum-product with 10 and 50 iterations, BP(50) + OSD-0, and BP + random
  perturbation with the same 5 x 10 budget as BP-GNN.
* BER (information bits) and BLER vs Eb/N0, Monte Carlo until 150 frame errors or a frame cap.
* **Inference time**: GPU latency (median over repeats, CUDA-synchronised) at
  batch 1/64/1024/4096 and throughput; CPU latency at batch 1/64. Fixed
  iterations and early-stop variants.
* **Power / energy**: NVML board energy counter on the GPU (falls back to
  integrating power samples), with average and peak power, energy per codeword,
  and dynamic energy (above measured idle power). RAPL CPU energy if the
  container exposes it. GNN MACs per codeword and iteration are also reported.

## Trade-off study (tradeoffs/)

Checks three common claims about "unrolled neural BP vs. chained BP -> GNN":

| claim | how it is tested |
|---|---|
| memory | peak GPU memory of a training step (B=256), largest power-of-two trainable batch, inference peak (B=4096) for NBP, GNN, BP-GNN end-to-end and BP-GNN with detached BP |
| adaptability | train on MacKay (204,102), evaluate unchanged on (96,48) and (408,204); 5G BG2 trained on (64,128), evaluated on (128,256), (256,512). Per-edge NBP cannot be transferred |
| error floor | up to 3M frames per Eb/N0 point, 100 frame errors |

## Running

```
python normal_ldpc/run_normal_ldpc.py --quick    # CPU smoke test
python 5g_ldpc/run_5g_ldpc.py --quick
python kaggle/build_kernel.py                    # build Kaggle kernels
kaggle kernels push -p kaggle/normal_ldpc
kaggle kernels push -p kaggle/5g_ldpc
kaggle kernels output gouravanirudh/gnn-ldpc-normal -p results/
```
