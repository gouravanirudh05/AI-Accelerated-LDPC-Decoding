# Graph residual correctors on top of truncated BP

A family of graph neural networks that try to **repair the output of a cheap,
truncated belief-propagation decoder** for the `LDPC_N49_K24` code, plus the
experiments that measure whether they actually do.

**Headline result: they do not pay off. Running five more BP iterations beats
every learned corrector here, on both accuracy and wall-clock cost.** The
detailed numbers, and the diagnostics explaining *why*, are below.

---

## 1. The idea being tested

Belief propagation on an LDPC code improves with iterations but costs linearly
in them. `BP(5)` is cheap and mediocre; `BP(20)` is 4x the cost and much better.
The question is whether a small graph network, run **once** on `BP(5)`'s output,
can buy `BP(20)`-like accuracy for less than `BP(20)`-like cost.

The corrector never replaces BP. It emits an additive residual per bit:

```
L_final = L_BP(5)  +  gate * delta_llr
                       ^^^^^^^^^^^^^^^ what the network predicts
```

Two properties follow from that design and are relied on throughout:

* The residual head is **zero-initialised**, so an untrained corrector is
  *exactly* `BP(5)`. Any change in error rate is attributable to training.
* The combination happens **outside** the model (`models/base_corrector.py:combine`),
  so `delta_llr` can be inspected, masked, or ablated without touching the
  network. Every diagnostic in `analysis/` depends on this.

BP itself is frozen — it runs under `no_grad`, and no gradient reaches it unless
a config flag explicitly enables it.

### Sign convention

A **positive** LLR means **bit 0**. The BCE logit for "this bit is 1" is
therefore `-llr`. This trips people up constantly; it is asserted in the tests.

---

## 2. Setup

| | |
|---|---|
| Code | `LDPC_N49_K24` (n=49, k=24, rate 0.49), from `GAT+BP/Codes_DB/LDPC_N49_K24.alist` |
| Channel | AWGN, BPSK, Eb/N0 2–6 dB |
| Baseline | Sum-product BP, flooding schedule, 5 / 10 / 20 iterations |
| Training | 1000–1500 steps, batch 256, Adam lr 3e-4, mixed Eb/N0 drawn per frame |
| Validation | 5,120 frames (used for checkpoint selection) |
| Final eval | **51,200 frames per SNR point**, on a seed distinct from training and validation |
| Threads | **1** — measured 146 ms/step at 1 thread vs 244 ms/step at 4; BP's per-check loop is many tiny tensor ops, so intra-op threading costs more than it saves |

Everything reuses the proven Phase 0–2 BP code through `_compat.py`, an
`importlib` shim (the `GAT+BP` directory contains a `+`, so it is not
importable by name). **The only change made to pre-existing files** is an
opt-in `return_trace=True` on `SumProductBP.forward` — the original 7 tests in
`GAT+BP/tests` still pass unmodified.

---

## 3. What the corrector sees

`BP(5)` is run with `return_trace=True`, which exposes the per-iteration LLR
history rather than just the final answer. From that trace (`bp_trace.py`):

**Per variable node (8 features):** soft channel value, soft final value, raw
channel LLR, raw final LLR, confidence `|LLR|`, **sign-flip count** across
iterations, node degree, and local parity violation.

The sign-flip count is the one feature BP's final output cannot provide: a bit
that oscillated 0-1-0-1-0 across five iterations looks identical to a settled
bit if you only read the last LLR. Its test asserts an oscillating bit gets a
non-zero count.

**Per check node (6 features)** and **per edge (6 features)**, mirroring the
legacy model's channels.

---

## 4. The architectures

Four families, each with a hand-rolled implementation and (where it makes
sense) a PyTorch Geometric one. The Tanner graph is kept as **two directed
edge-index tensors** (`vn_to_cn`, `cn_to_vn`), never collapsed to one
undirected set — extrinsic message passing needs the directions distinguished.

### 4a. GCN — the floor
`models/gcn_corrector.py` · `PyGGCNCorrector`

Plain neighbourhood averaging. Establishes what any graph structure at all buys.
The PyG version uses `SAGEConv((h,h), h, aggr="mean")` per direction, **not**
`GCNConv` — GCNConv's symmetric normalisation `1/sqrt(d_i d_j)` is derived for
homogeneous graphs and is not meaningful across a bipartite VN/CN split.

### 4b. GAT — the primary candidate
`models/gat_corrector.py` · `PyGGATCorrector`

Attention over neighbours, so the model can weight incoming messages instead of
averaging them. Two orthogonal flags:

* **`attention_variant: v1 | v2`** — PyG's `GATConv` vs `GATv2Conv`.
* **`edge_exclusion: true`** — enforce *extrinsic* information: the message on
  edge `e` excludes `e`'s own contribution, `(sum_j a_j v_j - a_e v_e)/(1 - a_e)`.
  This mirrors what BP itself does and prevents self-reinforcement.

> **GATv1 vs GATv2, precisely.** The difference is whether neighbour *ranking*
> can depend on the query. v1 scores `LeakyReLU(a_t·Wh_t + a_s·Wh_s)`: the
> query term is an additive constant within a target's neighbourhood and
> LeakyReLU is monotonic, so the neighbour ordering is query-independent
> (*static* attention). v2 scores `a·LeakyReLU(Wh_t + Wh_s)` with the
> nonlinearity inside, so ordering can change (*dynamic*).
>
> **Critical testing note:** if all pre-activations happen to be positive,
> LeakyReLU is the identity and *both* variants collapse to static. An early
> version of this test set all weights to one positive constant and "proved"
> v2 was dynamic when the test was vacuous. The real test uses random weights
> and zeroed `edge_bias`, and asserts ranking invariance rather than value
> equality.

### 4c. Local graph transformer
`models/graph_transformer_local.py` · `PyGGraphTransformerLocal`

Transformer block (attention + FFN + LayerNorm + residual) over a local
neighbourhood. `hop_count: 1 | 2` selects direct neighbours or 2-hop. The 2-hop
index is **materialised once at graph-build time and cached**, never recomputed
per forward pass.

### 4d. Sparse global graph transformer
`models/graph_transformer_sparse_global.py`

Select a small subset `S` of the least-confident bits, then run **full,
structurally unmasked attention among the members of `S` only** — letting
uncertain bits far apart on the graph consult each other directly. Two
selection modes:

* `mode: top_k` — the `top_k` least-confident bits, fixed size.
* `mode: threshold` — `{v : |LLR(v)| < threshold}`, variable size, padded with
  a key-padding mask.

Bits outside `S` get `delta_llr = 0`, enforced by **gather/scatter on the
output**, not by feeding them zeroed inputs. `threshold=0` yields an all-zero
residual without erroring (asserted in tests).

> **Naming wart, stated plainly:** `pyg_sparse_global_gt` is *aliased to the
> same hand-rolled class*, not a PyG layer. Selection plus dense attention over
> a node subset has no graph structure left to exploit, so it uses
> `torch.nn.MultiheadAttention` — the standard library implementation for dense
> attention. The `pyg_sparse_*` runs differ from `sparse_global_gt` only in
> `hidden_dim` (64 vs 32). The registry is aliased for symmetry and this is
> documented in `models/__init__.py`.

### 4e. ResNet (CNN) correctors — the non-graph comparison
`models/cnn_corrector.py`

Same inputs, same zero-initialized bounded residual head; only the encoder is
a convolution instead of message passing.

* `resnet1d` — 1-D ResNet over the bit sequence (4 residual blocks, kernel 5,
  dilations 1/2/4/8, receptive field 121 > n). It sees no Tanner-graph
  connectivity beyond the per-bit features every corrector gets; adjacent bit
  indices usually share no check, so convolutional locality is an assumption
  the code does not satisfy.
* `resnet2d` — 2-D ResNet over `H` rendered as an `m x n` image: an `H` mask
  channel plus bit and check features on each edge cell, so the CNN *does*
  see which bits share a check. Columns are pooled back to one hidden state
  per bit.

Both use circular padding (indices have no natural boundary). Comparing the
two separates "CNN vs GNN" from "can the model see the code structure".

### Residual mode — a trap worth documenting

`residual_mode: node | edge` decides whether one bounded residual is emitted
per bit or one per incident edge and then summed.

**In edge mode the effective bound is `max_degree × correction_scale`, not
`correction_scale`.** This code has max VN degree 4, so `correction_scale=8.0`
in edge mode means a bit can move by **±32 LLR**. That mistake was made and
shipped: `edge_local_gt` applied a median |Δ| of 13.3 and **broke 3,336 bits
BP had right while fixing 634**. `effective_correction_range` is now a property,
logged at train start, and guarded by a test against the shipped configs.

---

## 5. Loss terms

Six independently toggleable weighted terms (`losses/losses.py`):

| term | what it does |
|---|---|
| `bce` | per-bit binary cross-entropy on `-llr` |
| `deep_supervision` | BCE at each layer's intermediate residual, discounted by `gamma` |
| `focal` | down-weights easy bits |
| `hard_mining` | restricts loss to bits BP got wrong **or** with `|LLR| < threshold` |
| `block_soft_bler` | `softplus(T · logsumexp(error_logit / T))` — a differentiable soft-min over the block, targeting FER directly |
| `syndrome_aux` | parity-check violation penalty; **auxiliary only** — `LossConfig.validate()` raises if it is the sole enabled term |

Each term has a test asserting it produces a finite, correctly-shaped,
differentiable scalar on its own.

**Weights were calibrated by measuring gradient L2 norms**, not guessed: `bce`
0.039, `hard_mining` 4.231, `block_soft_bler` 0.796. A naive equal-weight config
would have been ~110x dominated by `hard_mining`, so the shipped FER recipe is
`1.0 / 0.02 / 0.1`, giving measured contributions 0.030 / 0.021 / 0.037.

---

## 6. Repository layout

```
ldpc_corrector/
  _compat.py          importlib shim onto the Phase 0-2 BP code in GAT+BP/
  tanner_graph.py     directed edge-index construction, cached 2-hop index
  bp_trace.py         per-iteration LLR trace -> VN / CN / edge features
  pyg_graph.py        bipartite disjoint-union batching for PyG
  decoder.py          BPBaseline, CorrectorDecoder, LegacyGATBPDecoder
  models/             the four architectures, hand-rolled and PyG
  losses/             the six loss terms
  train.py            config-driven training loop
  evaluate.py         BER/FER sweep, paired McNemar, FLOPs + wall-clock, plots
  ablate.py           runs config variants, not new code paths
  report.py           table formatting, precision / seed-spread summaries
  analysis/           mechanism, rescale, frame_concentration, timing
  configs/            one YAML per architecture + ablations.yaml (39 variants)
  tests/              336 tests (+7 pre-existing in GAT+BP/tests)
```

Ablations are **config variants, not new code paths** — `ablations.yaml` holds
39 of them.

---

## 7. Experiments conducted

| phase | what | outcome |
|---|---|---|
| Repo survey | locate alist loader, H representation, BP interface, channel sim, existing GNN | BP exposed only final LLR; added opt-in trace |
| Build | 4 architectures x 2 implementations, 6 losses, training + eval harness | 336 + 7 tests passing |
| Multi-seed | `pyg_{gcn,gat_v1,gat_v2,sparse}` x 3 seeds; `local_gt` hop1/hop2 x 3 seeds | seed spread quantified per family |
| Ablations | `attention_variant`, `hop_count`, `residual_mode`, `top_k`/`threshold`, loss composition, edge-mode scale fix | see §8.3 |
| Final sweep | 51,200 frames/point, 5 SNR points, paired vs BP(5), FLOPs + wall-clock, ECCT/AECCT overlay | see §8.1 |
| Mechanism | mask full-graph residuals to the sparse subset; measure where the correction budget is spent | restriction hypothesis **refuted** (§9.1) |
| Saturation / loss | `correction_scale=1.0`, `hard_mining_weight=1.0`, and both, then swept at 51,200 frames | both interventions **failed** (§8.4) |
| Rescaling | sign agreement, correlation with `bp_llr`, hard-decision flip rate | residual is **anti**-correlated (−0.98) with `bp_llr` (§9.3) |
| Frame concentration | where fixed and broken bits land relative to frame success | **explains BER-vs-FER gap** (§9.4) |
| Cost | 40 batches x 5 repeats per variant, FLOPs + wall-clock | BP(10) dominates; GATv2 costs 1.84x GATv1 (§8.5) |

---

## 8. Results

All numbers below are at **51,200 frames per SNR point**, on frames seeded
independently of training and checkpoint selection. `z` is a paired McNemar
statistic against BP(5) on identical frames; `z > +1.96` favours the variant,
`z < -1.96` favours BP(5).

### 8.1 Headline, at 4 dB

| variant | BER | FER | z vs BP(5) @4dB |
|---|---|---|---|
| **BP(20)** | **2.797e-03** | **2.035e-02** | **+34.5** |
| **BP(10)** | **3.533e-03** | **2.801e-02** | **+28.3** |
| legacy GAT+BP (Phase 0–2) | 3.988e-03 | 3.924e-02 | +13.5 |
| `pyg_sparse_s44` | 4.419e-03 | 4.043e-02 | +11.2 |
| `sparse_global_gt` | 4.478e-03 | 4.184e-02 | +7.1 |
| `fer_gcn` | — | 4.254e-02 | +3.3 |
| `gcn` | — | 4.299e-02 | +2.0 |
| `fer_gat` | — | 4.312e-02 | +1.9 |
| **BP(5)** *(reference)* | **5.031e-03** | **4.359e-02** | — |
| `pyg_gcn_s42` | 4.188e-03 | 4.373e-02 | −0.5 |
| `pyg_gat_v2_s43` | 4.252e-03 | 4.387e-02 | −1.0 |
| `pyg_local_gt_hop2` | 4.318e-03 | 4.449e-02 | −3.2 |
| `pyg_gat_v1_s42` | 4.313e-03 | 4.473e-02 | −3.9 |
| `local_gt` | — | 4.805e-02 | −12.7 |
| `edge_local_gt` | — | 5.477e-02 | −23.4 |
| ECCT (FP32) *(reproduced)* | 3.520e-03 | 4.340e-02 | — |
| AECCT final (ternary/int8) | 3.655e-03 | 4.293e-02 | — |

**Read the BER and FER columns against each other.** `pyg_gcn` has the *best
BER of any corrector* (4.188e-03, an 17% improvement on BP(5)) and a FER
*worse* than BP(5). The models reliably improve the metric BCE is a surrogate
for, and not the metric that matters. The reproduced ECCT shows the same
signature — BER between BP(10) and BP(20), FER at BP(5) level.

### 8.2 The effect changes sign with SNR

Every one of the 21 checkpoints is **negative at 2 dB, without exception**:

| variant | 2 dB | 3 dB | 4 dB | 5 dB |
|---|---|---|---|---|
| `pyg_sparse` x3 | −6.4 … −10.6 | **+6.5 … +11.5** | **+8.3 … +11.2** | **+4.7 … +5.0** |
| `pyg_local_gt_hop2` x3 | −11.7 … −30.3 | −15.0 … +3.6 | −3.2 … +6.6 | −1.8 … +3.5 |
| `pyg_gcn` x3 | −25.7 … −28.8 | −9.6 … −14.0 | −0.5 … −3.1 | −0.3 … −2.7 |
| `pyg_gat_v2` x3 | −25.0 … −29.4 | −10.7 … −15.6 | −1.0 … −4.1 | −2.1 … −3.6 |
| `pyg_gat_v1` x3 | −28.8 … −33.2 | −15.2 … −17.8 | −3.9 … −5.4 | −2.4 … −3.2 |

Selecting checkpoints on **aggregate mixed-SNR validation FER was the wrong
objective**: it averages a 2 dB loss against a 3–5 dB gain and reports the mean.

### 8.3 Ablations

**GATv1 vs GATv2 — v2 wins, but only at 2 dB.**

```
2 dB  v1 FER  4.0662e-01  4.0682e-01  4.1016e-01
      v2 FER  4.0521e-01  4.0127e-01  4.0596e-01   <- disjoint, v2 better
3 dB  v1 FER  1.7236e-01  1.7305e-01  1.7416e-01
      v2 FER  1.7174e-01  1.6967e-01  1.7258e-01   <- interleaved
4 dB  v1 FER  4.4727e-02  4.4941e-02  4.5176e-02
      v2 FER  4.4805e-02  4.3867e-02  4.4707e-02   <- interleaved
```

At 2 dB the seed ranges are strictly disjoint — v2's worst seed beats v1's best.
Elsewhere they interleave. Dynamic attention helps only where the graph is most
confused, by ~0.4% relative. The hand-rolled path reproduces the ordering
independently (`gat` 4.373e-02 vs `gat_attn_v1` 4.402e-02).

**Other axes:**

* **`hop_count` is not a distinguishable factor.** hop2 mean 0.115951 (spread
  0.0072) vs hop1 0.117839 (spread 0.0094) — the gap is a third of the noise.
* **`local_gt` is the least stable family**, hop1 spanning z = −34.3 to −6.5
  across three seeds; `pyg_sparse` is the most stable (FER@4dB spread 1.1e-03
  against `pyg_local_gt_hop1`'s 4.6e-03).
* **`top_k` is flat in K**: 4 / 8 / 16 give 4.186e-02 / 4.100e-02 / 4.164e-02,
  all inside the family's own seed spread.
* **Fixed-size selection beats threshold selection**: `pyg_sparse_thresh` is
  −21.7 at 2 dB and never exceeds +0.6.
* **Edge mode hurts every architecture** even at the corrected scale.
* **The FER-targeted loss helps slightly**: `fer_gcn` +3.3 vs `gcn` +2.0;
  `fer_gat` +1.9 vs `gat` −0.5.
* **The hand-rolled correctors beat their PyG replacements** — `gcn`
  4.299e-02 vs `pyg_gcn` 4.417e-02 — despite the PyG versions having 2x the
  hidden width and 4x the parameters. PyG's `GATConv`/`SAGEConv` carry no FFN
  or output projection and extra width did not compensate.
* **Capacity is not the bottleneck**: `sparse_global_gt` (17,409 params) and
  `pyg_sparse` (67,585 params) both land at ~0.1123 validation FER.

### 8.4 Saturation and hard mining — both interventions failed

Section 9.2 showed the residual pinned at the `correction_scale` bound on 93%
of bits. Two ways to attack that were tried; **neither helped**, and the
51,200-frame sweep is what settles it (validation FER is shown alongside to
make the point about resolution):

| run | val FER | BER@4dB | FER@4dB | z @4dB |
|---|---|---|---|---|
| `pyg_gat_v2_s43` *(scale 8.0 baseline)* | 0.117969 | 4.252e-03 | 4.387e-02 | −1.0 |
| `pyg_gcn_s42` *(scale 8.0 baseline)* | 0.117188 | 4.188e-03 | 4.373e-02 | −0.5 |
| `scale1_gat_v2` | 0.117188 | 4.734e-03 | 4.416e-02 | −2.6 |
| `scale1_gcn` | 0.118555 | 4.744e-03 | 4.436e-02 | −3.5 |
| `both_gat_v2` | 0.119336 | 4.802e-03 | 4.439e-02 | −3.4 |
| `both_gcn` | 0.119727 | 4.790e-03 | 4.443e-02 | −3.5 |
| `hardmine_gat_v2` | 0.155859 | 1.739e-02 | 5.926e-02 | −28.3 |
| `hardmine_gcn` | 0.156055 | 1.718e-02 | 5.990e-02 | −28.9 |

* **`correction_scale=1.0` came out slightly *worse* than scale 8.0**
  (4.416e-02 vs 4.387e-02), and worse on BER too. **The saturation was a
  symptom, not the cause** — capping the bound just produces a smaller
  shrinkage, still sign-consistent with §9.3, still no frame-level gain.
* **`hard_mining_weight=1.0` (50x the calibrated 0.02) is catastrophic** and
  reproduces on both architectures. It did what it was designed to do — the
  residual on BP's wrong bits rose 5x (0.46 → 2.52) — but left the residual on
  correct bits at the bound, and ends up flipping 8.25% of all bits, 17,551
  wrongly against 3,152 rightly (§9.3–9.4). It degrades BER 3.4x as well, so
  it is not a FER-for-BER trade; it is simply broken.
* Capping the scale *rescues* most of the hard-mining damage (`both_*` at
  ~4.44e-02 vs `hardmine_*` at ~5.96e-02), because it physically limits how
  much harm the encouraged corrections can do — but the result is still worse
  than plain BCE.

### 8.5 Cost per decode

Measured with `analysis/timing.py` (40 batches x 5 repeats, warmup discarded,
single thread). The `min`-`max` columns are the spread across repeats.

| variant | FLOPs/frame | µs/frame | min–max | xBP(20) | FER@4dB |
|---|---|---|---|---|---|
| BP(5) | 25,480 | 34.7 | 34.0–36.6 | 0.25 | 4.359e-02 |
| **BP(10)** | **50,960** | **68.2** | 67.8–72.8 | **0.49** | **2.801e-02** |
| BP(20) | 101,920 | 139.8 | 135.4–141.0 | 1.00 | 2.035e-02 |
| `sparse_global_gt` | 332,232 | 76.1 | 72.4–81.9 | 0.54 | 4.184e-02 |
| `pyg_sparse_topk4` | 614,408 | 103.3 | 91.1–104.6 | 0.74 | 4.186e-02 |
| `pyg_sparse_s44` | 1,163,272 | 112.1 | 108.8–112.6 | 0.80 | 4.043e-02 |
| `fer_gcn` | 1,985,480 | 275.2 | 264.9–278.6 | 1.97 | 4.254e-02 |
| `pyg_gcn_s42` | 7,787,528 | 268.7 | 260.3–273.3 | 1.92 | 4.373e-02 |
| `gat` | 1,711,752 | 345.3 | 324.3–350.4 | 2.47 | 4.373e-02 |
| `pyg_gat_v1_s42` | 2,397,192 | 355.9 | 341.9–372.9 | 2.55 | 4.473e-02 |
| `pyg_gat_v2_s42` | 2,397,192 | 654.6 | 634.6–665.0 | 4.68 | 4.446e-02 |
| legacy GAT+BP | 2,563,400 | 396.5 | 385.6–413.9 | 2.84 | 3.924e-02 |
| `pyg_local_gt_hop2_s43` | 7,414,792 | 2,155.5 | 2,095–2,192 | 15.42 | 4.331e-02 |

**BP(10) strictly dominates every corrector — cheaper in wall-clock *and*
better in FER.** It costs 0.49x BP(20) and reaches 2.801e-02; the cheapest
corrector (`sparse_global_gt`) costs 0.54x and reaches only 4.184e-02.

**GATv2 costs 1.84x GATv1 at identical FLOPs** (654.6 vs 355.9 µs, both
2,397,192). `GATv2Conv` transforms source and target separately and scores
after the nonlinearity, materialising more intermediates than the FLOP estimate
captures. With §8.3 that settles the v1-vs-v2 question practically: v2 buys
~0.4% relative FER at 2 dB only, for 84% more runtime.

**Report both cost metrics.** FLOPs and wall-clock disagree by up to 14x here
because BP's check update is a Python loop over many small per-check tensors —
latency-bound, not FLOP-bound. FLOPs alone flatter BP and understate what a
corrector costs to run; wall-clock alone is implementation- and
machine-specific. Note the earlier figures taken from the sweep's own
`timing_batches=5` were unusable (identical architectures measured 477 / 518 /
653 µs); always use `analysis/timing.py` for cross-variant comparison.

### 8.6 ResNet (CNN) correctors

3 seeds each, configs `resnet1d.yaml` / `resnet2d.yaml` (1500 steps, otherwise
the PyG protocol), trained and swept on a Kaggle Tesla T4
(`kaggle/build_kernel.py`). Same 51,200 seeded frames per point as §8.1:
BP(5)'s FER on the GPU matches the CPU headline sweep to every digit at all
five SNRs, so the paired z values are directly comparable with §8.1–8.2.

| variant | 2 dB | 3 dB | 4 dB | 5 dB | 6 dB | FER@4dB |
|---|---|---|---|---|---|---|
| `resnet1d` x3 | −19.3 … −22.3 | −6.8 … −9.0 | −1.5 … +0.8 | +0.6 … +1.6 | −1.0 … +1.0 | 4.34–4.39e-02 |
| `resnet2d` x3 | −7.1 … −16.7 | −3.9 … −0.3 | **+2.1 … +4.3** | +0.2 … +2.0 | +0.6 … +1.4 | 4.28–4.31e-02 |
| `pyg_sparse` x3 (§8.2) | −6.4 … −10.6 | +6.5 … +11.5 | +8.3 … +11.2 | +4.7 … +5.0 | — | 4.04e-02 (s44) |

* **`resnet1d` behaves like `pyg_gcn`**: BER improves (4.30e-03 vs BP(5)'s
  5.03e-03 at 4 dB) while FER does not move — the §9.4 pattern again.
* **Giving the CNN the code structure helps, but not enough.** `resnet2d` is
  the only CNN significantly positive anywhere (4 dB, 2 of 3 seeds past
  z = +1.96) and is less damaging at 2–3 dB, but it is still well behind the
  sparse-global GNN and never positive at 2 dB.
* **BP(10) still dominates.** On the T4, BP(10) runs at 134 µs/frame and
  reaches 2.80e-02; `resnet2d` costs 146–152 µs/frame (51 MFLOPs/frame, 500x
  BP(20)'s FLOPs) for 4.28e-02. `resnet1d` is cheaper (82–84 µs) but gains
  nothing. These timings come from the sweep's own `timing_batches=5`, which
  this section warns against using for fine comparisons; the gap to BP(10)
  is large enough not to depend on it.

### 8.7 Sparse-global follow-ups: gate, FER loss, resumed BP

Three follow-ups to `pyg_sparse` (`kaggle/run_sparse_followup.py`), 3 seeds
each at its original protocol (1000 steps, hidden 64, top-8), trained and swept
on a Kaggle T4 at 51,200 frames per point (BP(5) matches §8.1 to every digit),
paired against BP(5) **and BP(10)**:

* `sparse_gate` — `model.use_gate=true`, a learned per-bit gate that could
  switch the correction off where it hurts (2 dB);
* `sparse_fer` — the FER-facing loss of §8.3 (BCE + hard mining 0.02 + soft
  block error 0.1), never tried on this architecture before;
* `sparse_resume` — **resumed BP**: the correction is added to the channel LLR
  of the corrected bits and BP continues for 5 more iterations from BP(5)'s own
  messages (`bp.resume_iterations=5`, `SumProductBP(initial_messages=...)`),
  trained end to end through those iterations. With a zero correction this is
  BP(10) bit for bit (`tests/test_resume.py`), so it is judged against BP(10).
  The same baseline checkpoints were also swept with resumed BP added and no
  retraining.

BP(6)–BP(8) were swept afterwards on the same frames
(`Results_Corrector/sparse_followup/bp_iterations_5_to_8/`).

| variant (3 seeds) | 2 dB | 3 dB | 4 dB | 5 dB | z vs BP(5) @2 dB | z vs BP(5) @4 dB | µs/frame (T4) |
|---|---|---|---|---|---|---|---|
| BP(5) | 0.3859 | 0.1647 | 0.0436 | 0.0069 | — | — | 72 |
| `sparse` (baseline) | 0.3907 | 0.1608 | 0.0409 | 0.0063 | −11.0 … −8.9 | +9.5 … +10.9 | 92–95 |
| `sparse_gate` | 0.3909 | 0.1609 | 0.0408 | 0.0064 | −11.8 … −6.3 | +9.9 … +11.3 | 94–97 |
| **`sparse_fer`** | **0.3834** | **0.1562** | **0.0395** | **0.0061** | **+2.7 … +6.3** | **+13.6 … +13.8** | 94–104 |
| **BP(6)** | **0.3674** | **0.1494** | **0.0376** | **0.0057** | +29.1 | +17.2 | ~86* |
| BP(7) | 0.3535 | 0.1390 | 0.0339 | 0.0051 | +40.4 | +22.2 | ~101* |
| `sparse` + BP(5) resumed, no retraining | 0.3426 | 0.1260 | 0.0292 | 0.0041 | | | 164–171 |
| `sparse_resume` (trained through BP) | 0.3336 | 0.1238 | 0.0282 | 0.0039 | | | 164–169 |
| `sparse_resume` + gate + FER loss | 0.3353 | 0.1240 | 0.0284 | 0.0039 | | | 163–169 |
| **BP(10)** | **0.3303** | **0.1219** | **0.0280** | **0.0038** | +53.3 | +28.2 | 145 |

\* BP(6)/BP(7) were swept on the CPU; their T4 cost is interpolated from
BP(5)/BP(10)/BP(20), which scale at ~14.4 µs per iteration.

Paired z of the resumed variants **against BP(10)**: `sparse_resume`
−9.4 … −1.5 at 2 dB, −7.0 … −3.9 at 3 dB, −2.3 … 0.0 at 4 dB; without
retraining −20.2 … −18.4 at 2 dB, −5.1 … −4.7 at 4 dB.

**What it says:**

1. **The FER-facing loss is the one real improvement.** `sparse_fer` is the
   first corrector in this repo that is positive against BP(5) at *every*
   SNR, 2 dB included (+2.7 … +6.3, where every other corrector is negative),
   and its 4 dB FER (0.0395) is the best one-shot corrector result here. It
   fixes the 2 dB regression that §8.2 found in all 21 checkpoints.
2. **The gate does nothing.** `sparse_gate` matches `sparse` at every SNR; the
   per-bit gate did not learn to switch off at 2 dB. The loss fixed what the
   gate could not.
3. **But one more BP iteration beats every corrector.** BP(6) beats
   `sparse_fer` at every SNR (0.0376 vs 0.0395 at 4 dB) and is cheaper (~86 vs
   ~95 µs). The whole gain of the best corrector is worth less than one
   additional BP iteration. That is a sharper form of §8.5's "BP(10)
   dominates".
4. **Resumed BP lands on BP(10), not past it.** Trained through the resumed
   iterations, the corrector matches BP(10) at 4–6 dB (paired z −2.3 … 0)
   and is significantly *worse* at 2–3 dB, at ~15% more wall-clock. Since the untrained version *is*
   BP(10), training found no correction that helps BP converge. The 5 extra
   BP iterations do all the work.
5. **Without retraining, the correction actively hurts resumed BP** (z −19 at
   2 dB against BP(10)). That fits §9.3: the learned rule shrinks BP's LLR, and
   folded in as a prior a shrink is weaker evidence for BP to propagate.
   Training through BP mostly teaches the corrector to stop shrinking.

So along the sparse-global line, spending the same compute on BP iterations
wins every comparison. Combining all three changes adds nothing over resumed BP
alone.

## 9. Why it does not work

Four diagnostics, each refuting the previous explanation. This is the most
useful output of the whole exercise, so the wrong turns are kept in.

### 9.1 Restriction is not the mechanism

Masking each full-graph corrector's residual down to the same bottom-8
confidence bits that sparse-global selects changes FER by at most 2e-04
(`analysis/mechanism.py`). Sparse-global's edge does not come from being
structurally unable to disturb confident bits.

### 9.2 The residual is larger on bits BP got right

In *every* configuration -- scale 8.0, scale 1.0, hard-mined, both -- the
median `|residual|` is larger on bits BP decoded correctly than on the bits it
got wrong:

```
variant            |d| right  |d| wrong
pyg_gcn_s42            8.000      0.460   <- 8.000 is exactly correction_scale
pyg_gat_v2_s43         6.193      0.475
pyg_local_gt_hop2_s43  6.548      0.602
scale1_gcn             1.000      0.561   <- saturated at the smaller bound too
hardmine_gcn           8.000      2.522   <- hard mining raised "wrong" 5x but
                                             left "right" pinned at the bound
```

This *looks* like the model failing to identify which bits need help. It is
not — see 9.3.

### 9.3 The correctors shrink BP's LLR; they do not reinforce it

`analysis/rescale.py` measures the residual's relationship to `bp_llr`
directly. The prediction was sign agreement near 1.0 (a residual that only
reinforces). The measurement is the opposite:

```
variant                    agree    corr     flip    ->ok   ->bad    net
pyg_gcn_s42                0.019  -0.979  0.00358     539     360   +179
pyg_gat_v2_s43             0.012  -0.975  0.00391     585     397   +188
pyg_local_gt_hop2_s43      0.028  -0.981  0.00534     728     612   +116
pyg_sparse_s44             0.058  -0.974  0.00342     511     348   +163
scale1_gcn                 0.010  -0.949  0.00410     527     502    +25
hardmine_gcn               0.002  -0.956  0.08252    3152   17551 -14399
gcn                        0.026  -0.979  0.00334     521     316   +205
fer_gcn                    0.006  -0.977  0.00509     733     545   +188
```

Sign agreement is 1–6% and the correlation with `bp_llr` is **−0.98**. The
learned rule is `delta ≈ −α·bp_llr`: the correctors **shrink** BP's LLR toward
zero.

That dissolves 9.2 as a separate phenomenon. `|d|` is larger on already-correct
bits purely because `|delta| ∝ |bp_llr|` and correct bits carry larger `|LLR|`.
It was never a misallocation of the correction budget.

It also shows the correctors are *not* no-ops: they flip ~0.35% of hard
decisions, and come out **ahead at the bit level** (+179, +188, +163, +205).

### 9.4 Bit-level gains do not convert into frame-level gains

`analysis/frame_concentration.py` explains the gap. A failed BP(5) frame
carries **6.30 bit errors on average**, so fixing one bit inside it rescues
nothing — while breaking one bit inside a clean frame reliably destroys it.

```
variant                   fix in bad  fix rescues  break in ok  err/bad frm
pyg_gcn_s42                      525           14           35         6.30
pyg_sparse_s44                   478           33           17         6.30
gcn                              505           16           27         6.30
hardmine_gcn                    3151            1         3400         6.30
```

For `pyg_gcn_s42`: of 539 bits corrected, **525 (97.4%) are wasted** inside
frames that fail regardless, and only **14 rescue a frame**, against **35
breaks** that each destroy a previously-clean frame. Net **+179 bits, −21
frames**. That is precisely why BER improves 17% while FER does not.

**And it explains why sparse-global is the only family with positive z:** it
inverts the ratio to **33 rescues against 17 breaks** (2:1), where `pyg_gcn`
runs 14 against 35 (1:2.5). Restricting the residual to 8 low-confidence bits
concentrates it on frames near the decision boundary instead of scattering it
across confident ones.

### 9.5 What follows for anyone continuing this

Per-bit BCE optimises the wrong object, and no reweighting tried here fixes it
(focal, hard mining, soft-BLER, and gradient-calibrated combinations were all
measured; §8.3–8.4). The structural facts to design against are:

1. **A fix is worth almost nothing unless it completes a frame.** With 6.3
   errors per failed frame, a corrector must fix ~6 bits jointly or gain
   nothing. This argues for a genuinely block-level objective, or for iterating
   the corrector inside BP rather than once after it.
2. **A break is always worth a full frame.** The asymmetry is ~1:0.026 in
   frame terms, so the safe operating point is extreme conservatism — which is
   what restriction to a few uncertain bits achieves, and why it is the only
   thing that worked.
3. **Shrinking the LLR is what BCE actually rewards** on this code. Any future
   loss should be checked against `analysis/rescale.py` for a −0.98
   correlation before training is scaled up.

## 10. Methodology notes

Things that cost real time and are worth not repeating.

**The 5,120-frame validation set cannot resolve what matters.** It reads FER to
±2e-04 at best. Two conclusions were drawn from it and later overturned by the
51,200-frame sweep:

1. "The sparse corrector is statistically tied with BP(5)" — it is not; the
   effect is strongly resolvable but **changes sign with SNR**, so the aggregate
   averaged it to nothing.
2. "The FER-targeted loss buys nothing" — it gives a consistent +1.4 z
   improvement on both architectures.

Rate resolution from `k` frame errors is `sigma/rate ≈ 1/sqrt(k)`. At 51,200
frames the 2–4 dB points collect 2,000–20,000 errors (0.7–2% relative), but
**5 dB reaches only ~350 errors and 6 dB only ~32** — treat the two highest
SNR points as indicative, not decisive.

**`gat` and `local_gt(hop_count=1)` are the same model** in the hand-rolled
path — same 40 parameter tensors, bit-identical outputs when weights are
matched (max abs diff 0.0). Their apparent 0.1150-vs-0.1283 FER gap was pure
seed variance, and it invalidated an earlier claim. Pinned by
`test_local_gt_hop1_is_the_same_model_as_gat`.

**Always check `effective_correction_range`, not `correction_scale`.** See §4.

---

## 11. Reproducing

```bash
python -m venv .venv && ./.venv/bin/pip install -r ldpc_corrector/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1        # single thread is fastest here

# train one architecture
./.venv/bin/python -m ldpc_corrector.train \
    --config ldpc_corrector/configs/pyg_sparse_global_gt.yaml \
    --set name=my_run seed=42 train.steps=1000

# final sweep: BER/FER vs Eb/N0, paired McNemar, cost, ECCT/AECCT overlay
./.venv/bin/python -m ldpc_corrector.evaluate \
    --config ldpc_corrector/configs/pyg_sparse_global_gt.yaml \
    --bp-iterations 5 10 20 \
    --checkpoint Results_Corrector/pyg/pyg_sparse_s44.best.pt \
    --legacy-checkpoint GAT+BP/best_model.pt \
    --set eval.frames=51200 --out-dir Results_Corrector/eval/headline

# diagnostics
./.venv/bin/python -m ldpc_corrector.analysis.mechanism           <ckpt> ...  # §9.1
./.venv/bin/python -m ldpc_corrector.analysis.rescale             <ckpt> ...  # §9.3
./.venv/bin/python -m ldpc_corrector.analysis.frame_concentration <ckpt> ...  # §9.4
./.venv/bin/python -m ldpc_corrector.analysis.timing              <ckpt> ...  # §8.5

# ablation grid (config variants, no new code paths)
./.venv/bin/python -m ldpc_corrector.ablate --sweep ldpc_corrector/configs/ablations.yaml
#   --only <name> ...   run a subset, in the order given
#   --dry-run           list what would run

./.venv/bin/python -m pytest ldpc_corrector/tests GAT+BP/tests -q   # 336 + 7 = 343
```

Outputs land in `Results_Corrector/`: `<name>.best.pt`, `<name>.last.pt`,
`<name>.config.json`, `<name>.history.json`, `<name>.log`; sweeps write
`results.csv`, `ber_vs_ebno.png`, `bler_vs_ebno.png`.

---

## 12. Status and honest assessment

**Done:** all four architectures in both implementations, all six loss terms,
training and evaluation harnesses, a 39-variant ablation grid, 3-seed
replication of the PyG families, a 51,200-frame final sweep with paired
significance testing and cost accounting, and four mechanism diagnostics.
**336 tests pass** in `ldpc_corrector/tests`, plus the original **7** in
`GAT+BP/tests` still passing unmodified (343 combined).

**The approach does not achieve its goal.** The target was to close the
BP(5)→BP(20) FER gap of 2.32e-02 at 4 dB. The best corrector closes 3.2e-03 of
it — about **14%** — while costing more wall-clock than BP(10), which closes
**67%** for less. On the cost/accuracy tradeoff this experiment was built to
measure, **more BP iterations win.** No corrector is positive against BP(5) at
2 dB, and only the sparse-global family is positive at 3–5 dB.

**The valuable output is the diagnosis, not the models.** §9 establishes, with
measurements rather than argument, that per-bit BCE on this code rewards
shrinking BP's LLR (correlation −0.98), that this does produce a net bit-level
gain, and that the gain cannot become a frame-level gain because a failed frame
carries 6.3 bit errors while a broken bit costs a whole frame. That asymmetry —
roughly 1:0.026 in frame terms — is the thing to design against, and it is why
the only architecture that helped is the one restricted to a handful of
uncertain bits. §9.5 lists the three constraints any follow-up should respect.

**Caveats and gaps, explicitly:**

* The 5 dB and 6 dB sweep points collect only ~350 and ~32 frame errors
  respectively (≈5% and ≈18% relative sigma). Treat them as indicative.
* `pyg_local_gt` has 3 seeds against a 4.6e-03 FER spread, so its family mean
  is soft; every other PyG family is tighter than 1.2e-03.
* Wall-clock is machine- and load-specific. The §8.5 ratios reproduce; the
  absolute microseconds do not (BP(5) measured 23.3 µs during the sweep and
  34.7 µs in the careful run).
* `deep_supervision`, `focal`, and `syndrome_aux` are implemented and
  unit-tested but were never included in a trained-and-swept variant.
* `use_gate` (learned per-bit gating) and `edge_exclusion` are implemented and
  tested but not swept at 51,200 frames.
* The ECCT/AECCT overlay is a *pre-existing reproduction* in
  `comparison/data/test_results.csv`, not re-run here. Its FER sits at BP(5)
  level despite a BER between BP(10) and BP(20) — the same misalignment
  signature as §9, which is worth noting but was not investigated.
