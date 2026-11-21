# BP + pooled bipartite GNN (BP-PGNN) for LDPC decoding

The third hybrid of this repository, next to BP-GNN (`Nvidia_GNN+BP(lLatest)`,
adapted from NVIDIA's idea) and BP-SGT (`SparseGT_BP`). The pipeline is the same
as BP-SGT. Only the module between the BP runs changes:

```
llr -> BP(10) -> PGNN -> BP(10) -> PGNN -> BP(10) -> PGNN -> BP(10) -> PGNN -> BP(10)     (5x10)
                                    same weights, 40 iterations per run                     (5x40)
```

**Status: code written and smoke-tested on CPU only. Nothing here has been trained on a GPU yet, so
every "why it should be better" below is a hypothesis to be tested by the Kaggle runs.**

## Why replace the sparse attention

BP-SGT ranks the bits by |L| and lets only the k least-confident ones attend to
each other. The objection: the bits that would fix a stuck pattern are often
*confident and wrong*, or confident and right but needed as evidence. These bits
are not in the weak top-k, so attention only reshuffles information among bits
that are all unreliable. The tuning confirms this: BP-SGT only worked once
k = 0.5 n, where it is almost dense and costs k^2 m (768 M MACs per stage on
5G 1408, 17x BP-GNN). The EEC ablation showed the k x k score matrix was not
where the gain came from.

PGNN keeps *every* bit and check and gets long-range context by **pooling**
instead of by selecting nodes.

## Method (`pgnn/pgnn_decoder.py`)

After every BP run that has not converged:

1. **Inputs**, as in BP-GNN plus one feature:
   * bits: channel LLR, prior and BP marginal (clip/8 and tanh/2 each), and the
     fraction of the bit's checks that are unsatisfied;
   * checks: soft syndrome, 1 - 2 x hard syndrome;
   * edges: tanh(v2c/2), tanh(c2v/2), plus a base-graph edge embedding;
   * node-type embeddings: degree, or the 5G base-graph column and row;
   * **random-walk structural encoding** (see below).
2. **Fine encoder.** One bipartite message-passing layer on the full Tanner graph
   (bit to check, then check to bit).
3. **Bipartite DiffPool (BiDiffPool).** Bits and checks are pooled separately,
   so the coarse graph is still bipartite:
   * pooled features: X_v = S_v^T h_v and X_c = S_c^T h_c (membership-weighted means);
   * pooled adjacency: A' = S_c^T H S_v [K_c x K_v];
   * coarse edge feature: the same product over *unsatisfied* checks only,
     A'_u = S_c^T diag(u) H S_v. The coarse edge feature is A'_u / A', the
     fraction of violated edges between two clusters.
   * **5G NR (`assign="proto"`):** S is the protograph lifting itself. Bit
     cluster = base-graph column, check cluster = base-graph row. The coarse
     graph is the base graph (46x68 for BG1, 42x52 for BG2), exact,
     parameter-free and independent of Z. Pooling is an index-add, so it costs
     almost nothing.
   * **Other codes (`assign="learned"`):** S = softmax(Linear(h)) per frame,
     K_v = n/16 and K_c = m/16. The clusters depend on the frame's BP state, so
     the model can group, for example, "bits around violated checks".
4. **Concatenated readout.** g = MLP([mean h_v || max h_v || mean h_c || max h_c]),
   with bits and checks kept apart. This gives one context vector per frame,
   fed to every coarse node and added back to every fine node.
5. **Two dense bipartite layers on the coarse graph.** Each node aggregates its
   neighbours twice: once weighted by A' and once weighted by A' x the violated
   fraction.
6. **Unpool** (U-Net skip): h_v += Linear(S_v X'_v + g), h_c += Linear(S_c X'_c + g).
   *Every* bit receives the coarse context. This is what lets pooling work for
   a per-bit output: nothing is dropped.
7. **Fine decoder.** One more bipartite layer on the full graph, then a
   zero-initialised head gives one correction per bit. The next prior is
   prior + delta (the base of BP-SGT's tuned config; `base="llr"` gives the
   BP-GNN form). An untrained model is exactly BP with restarts. The smoke
   test checks this bit-exactly.

### The four pooling families: what was used and why

| Method | Used? | Reason |
|---|---|---|
| BiDiffPool | **Main model** | Keeps the bipartite structure. Soft assignment, so no node is dropped and unpooling gives every bit a correction. For 5G the assignment is the protograph, so the memory problem of dense assignments disappears (index pooling). For the other codes K = n/16, so S is n x n/16 (at most 36 clusters on WiMAX). |
| Dual Top-K / SAGPool | **Ablation** (`pool="topk"`) | Separate ratios: 12.5 % of bits, 25 % of checks. Check scores also see the hard syndrome, so violated checks are preferred, which addresses the "weak k" objection. Dropped nodes get only the global vector, and bits can end up isolated from their checks: the trade-off from your table. |
| One-mode projection + pool | **Not used** | Projecting checks onto bits makes every check a dense clique of d_c bits and loses the check nodes, which hold the syndrome. This is what BP-SGT's shared-check bias already did, at k^2 m cost. |
| Concatenated readout | **Used inside the main model** | On its own it gives one vector per frame and cannot produce per-bit outputs. As a context vector broadcast back to every node it is a cheap global channel. `pool="none"` is the ablation with *only* this plus the GNN+ tricks. |

### Optimisation methods from the GNN17 lecture (GNN+, Luo, Shi & Wu, ICML 2025)

| Trick | Applied as |
|---|---|
| Edge features in every message | The BP messages v2c and c2v, plus a base-graph edge embedding, enter every fine message. The violated-edge fraction is the coarse edge feature. |
| Normalisation | **LayerNorm, pre-norm** (your choice). Not BatchNorm: at inference only frames BP failed reach the module, and early stopping shrinks the batch to as little as one frame, so batch statistics would not match training. |
| Dropout, low rate | 0.1 on every residual branch (the lecture's finding: at most 0.2). Off at inference. |
| Residual connections | On every layer, fine and coarse. |
| Feed-forward network | An FFN (d to 2d to d) after every message-passing layer, fine and coarse. |
| Positional encoding | RWSE: return probability of a 2t-step random walk, t = 1..8, for every bit and check. The return probability at 4, 6 and 8 steps rises with the number of 4-, 6- and 8-cycles through the node. Short cycles are what trapping sets are built from, so the module can tell which bits sit in cycle-rich regions. Computed once per code and cached. |

## How it should beat each baseline (hypotheses, to be tested)

**Against BP(50) / BP(200).** Same argument as for every BP hybrid: when BP is
stuck in a trapping set, a learned restart prior changes the fixed point, while
more iterations of the same BP do not. On the short codes, BP-GNN and BP-SGT at
5x10 are already below BP(200), and PGNN starts from the same mechanism. The new
part matters on the long codes. There, both existing hybrids at 5x10 are *worse*
than BP(50): on 5G 1408, 1.66e-2 (GNN) and 2.24e-2 (SGT) against 1.02e-2. That
suggests the restart prior is actually wrong on long codes. A plausible cause is
that each bit's correction is decided from a local or weak-bit-only view.

**Against BP-GNN.** BP-GNN has 2 layers, so each bit sees only its 2-hop
neighbourhood (its checks and their bits). On a 1408-bit code a stuck pattern
spans more than that, and the evidence for which of a check's bits is wrong
comes from other checks further away. Mean aggregation also dilutes the one
violated message among d_c - 1 satisfied ones; this is the CCSDS failure of the
GNN. PGNN keeps the same 2 fine layers (1 before, 1 after pooling) and adds a
coarse graph:
* on 5G the coarse graph is the base graph, whose core is nearly dense, so 2
  coarse layers connect any two bits of the code through about 3 hops;
* the coarse edge feature tells every cluster *where in the code* the violated
  checks sit (which base-graph rows, and on which columns);
* the global readout gives the overall syndrome state.

All of this costs **1.15x to 1.43x BP-GNN MACs** (table below). A wider BP-GNN
with more layers would cost far more per layer.

**Against BP-SGT.** BP-SGT also has long range, but (a) only among the weak bits,
and (b) at 768 M MACs per stage on 5G 1408, so it gets very few training steps
in the same time (2.6k on 5G 1408 against 14.5k on MacKay 204). PGNN reaches
every bit at 52 M MACs on the same code, about 15x cheaper, so it fits a full
batch and many more steps. The "unsatisfied-check" signal enters through the
checks, which are where the violation actually is, not through a |L| ranking of
bits.

**Ablations that separate the claims.** `PGNN-NoPool` (GNN+ tricks + readout,
no coarse graph) against `PGNN` measures whether the *pooling* helps, rather
than only the GNN+ tricks. `PGNN-TopK` against `PGNN` measures whether the soft,
no-drop pooling beats node selection.

## Cost of one refinement stage (MACs per codeword, from `macs_per_refinement`)

| code | n | assignment (K_v x K_c) | BP-GNN [M] | BP-SGT [M] | PGNN [M] | PGNN / BP-GNN | PGNN training |
|---|---|---|---|---|---|---|---|
| MacKay (96,48) | 96 | learned 6x3 | 2.6 | 4.0 | 3.1 | 1.19 | 80 min |
| CCSDS (128,64) | 128 | learned 8x4 | 4.0 | 5.8 | 4.7 | 1.17 | 80 min |
| MacKay (204,102) | 204 | learned 13x7 | 5.5 | 11.5 | 6.7 | 1.21 | 80 min |
| MacKay (408,204) | 408 | learned 26x13 | 11.0 | 41.1 | 13.8 | 1.25 | 80 min |
| WiMAX (576,288) | 576 | learned 36x18 | 16.0 | 88.1 | 20.5 | 1.28 | 160 min |
| 5G BG2 (128,64) | 128 | protograph 52x42 | 4.2 | 7.5 | 6.0 | 1.43 | 80 min |
| 5G BG2 (256,128) | 256 | protograph 52x42 | 8.5 | 24.1 | 10.9 | 1.29 | 80 min |
| 5G BG2 (512,256) | 512 | protograph 52x42 | 16.5 | 93.4 | 20.0 | 1.21 | 160 min |
| 5G BG1 (704,528) | 704 | protograph 68x46 | 22.5 | 134.0 | 26.6 | 1.18 | 160 min |
| 5G BG1 (1408,1056) | 1408 | protograph 68x46 | 45.0 | 768.1 | 51.6 | 1.15 | 160 min |

Budget chosen by Gourav: at most 1.5x BP-GNN per stage. `tests/smoke_test.py`
asserts this for every code. Parameters: about 52k (learned) and 54-56k
(5G), against 22-27k for BP-GNN and 68-72k for BP-SGT. Width d = 16, d_e = 8
and hidden = 48, as in BP-GNN.

## Protocol (choices made on 2026-11-21)

* **Codes:** all 10 codes of the BP-GNN / BP-SGT comparison from the start.
* **Training recipe:** unchanged (BCE after every BP stage + 0.1 x soft-syndrome
  loss, hard-example mining of BP(10) failures in 4 Eb/N0 pools, plateau LR,
  model selection on a fixed set of BP failures, same Eb/N0 bands and batch caps).
* **Training time:** 80 min, **160 min on codes with n >= 512** (WiMAX 576, 5G 512,
  704, 1408). BP-GNN and BP-SGT had 80 min everywhere, so on those four codes the
  comparison is **not like-for-like in training budget**. The merged table marks
  those rows with `*` and prints the training minutes of all three models. If a
  PGNN gain appears only on starred rows, the next step is to retrain BP-GNN for
  160 min on the same codes before claiming anything.
* **Comparison:** equal BP iterations, 5x10 against BP(50) and 5x40 against
  BP(200). The final runs still record latency, throughput and energy, with
  BP(50) re-profiled in the same session as a reference, but speed is not the
  claim here.

## Running

```
# CPU smoke test (about 20 s): forward/backward for diff/topk/none on all 10 codes,
# untrained == BP with restarts, MACs <= 1.5x BP-GNN
python PooledGNN_BP/tests/smoke_test.py

# Kaggle (T4 x2), same accounts and code split as the BP-SGT final runs
python PooledGNN_BP/kaggle/build_kernel.py          # builds smoke, final_a..d, abl_topk_a/b, abl_nopool_a/b
kaggle kernels push -p PooledGNN_BP/kaggle/smoke    # first, CPU, minutes
kaggle kernels push -p PooledGNN_BP/kaggle/final_a  # ... final_b, final_c, final_d

# after copying the kernel outputs to PooledGNN_BP/results/kaggle_final_<x>/
python PooledGNN_BP/final/merge_pgnn.py             # -> results/final_comparison/COMPARISON.md
```

Local single-code run: `python PooledGNN_BP/final/run_pgnn_final.py --group 5g
--codes 5G_BG1_k1056_n1408 [--config pgnn_topk_config.json] [--quick]`.

Kaggle hours: the four final kernels use about 4 x 2 GPU queues. The longest
queue is 5G 1408 (160 min training + deep Monte Carlo + profiling), well under
the 12 h session limit. The ablations are optional (4 more kernels on MacKay
204, WiMAX, 5G 704 and 5G 1408).

## Known limitations

* DiffPool's auxiliary link-prediction and entropy losses are not used. The
  shared training loop (`common/train.py`) only knows the BCE + syndrome loss,
  and the assignment is trained end to end by the decoding loss alone.
* With protograph pooling, all Z lifted copies of a column share one cluster,
  so the coarse graph cannot tell copies apart. Per-copy detail comes back
  through the U-Net skip and the fine layers only.
* For short 5G codes, rate matching keeps only some base-graph rows and columns
  (e.g. 18 of 52 columns for (128,64)). The unused clusters are empty and
  harmless, but they are still counted in the MACs above.
