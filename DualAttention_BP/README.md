# BP + dual-attention Tanner transformer (BP-DAT) for LDPC decoding

The fourth hybrid of this repository, next to BP-GNN (`Nvidia_GNN+BP(lLatest)`), BP-SGT
(`SparseGT_BP`) and BP-PGNN (`PooledGNN_BP`). The pipeline is unchanged. Only the module
between the BP runs is new:

```
llr -> BP(10) -> DAT -> BP(10) -> DAT -> BP(10) -> DAT -> BP(10) -> DAT -> BP(10)     (5x10)
                                same weights, 40 iterations per run                   (5x40)
```

**Status: code written and smoke-tested on CPU only. Nothing here has been trained on a GPU
yet, so every "why it should be better" claim is a hypothesis for the Kaggle runs.**

The architecture comes from a survey of the *general* GNN literature on bipartite,
heterogeneous and hypergraph learning, graph transformers and long-range GNNs. LDPC and
neural-decoding work was deliberately not searched. The survey, with every candidate and why it
was or was not picked, is in [RESEARCH.md](RESEARCH.md). In short, the closest graph class to a
Tanner graph is the **variable-constraint bipartite graph** of MILP / SAT. The newest backbone
for those graphs, dual attention (Huang et al., ICML 2026), maps one-to-one onto bits and checks.

## Method (`dat/dat_decoder.py`)

Inputs, structural encoding (RWSE), zero-initialised head and `prior + delta` output are BP-PGNN's,
so the only difference to BP-PGNN is the module, and an untrained model is exactly BP with
restarts (checked bit-exactly by the smoke test). Each of the 2 layers runs:

1. **checks <- bits**: cross branch (on the edges of H) || self branch (all checks) || node state || check features -> MLP, then FFN;
2. **bits <- checks**: the same with the roles swapped.

Pre-LayerNorm, residuals, dropout 0.1 and FFNs as in BP-PGNN (GNN+ recipe). Width d = 16,
d_e = 8, hidden = 48 and 2 heads.

| Branch | `DAT` (main) | `DAT-NB` (second candidate) | `DAT-NoSelf` (ablation) |
|---|---|---|---|
| cross (sparse, on H) | **edge-aware softmax attention** over the receiver's neighbours. Score = sum over head dims of (q ⊙ k ⊙ (1 + W e)) / sqrt(d_h), value = v + W' e, with e = BP messages tanh(v2c/2), tanh(c2v/2) + base-graph edge embedding | **non-backtracking** directed edge states: s_{c->v} = MLP(sum_{v' != v} s_{v'->c}, h_c, e) and s_{v->c} likewise (extrinsic, NBA-GNN) | as DAT |
| self (global) | **linear attention, sigmoid kernel**, every bit over all bits and every check over all checks: σ(Q)(σ(K)^T V) / (σ(Q) Σ σ(K)) | as DAT | none |

Differences to the MILP backbone: the two Tanner half-steps run in sequence (bit -> check -> bit
per layer, as BP-GNN) instead of in parallel, and the edge feature is the pair of BP messages
instead of the constraint coefficient A_ij.

## Cost of one refinement stage (MACs per codeword, from `macs_per_refinement`; CPU smoke test)

| code | n | BP-GNN [M] | BP-SGT [M] | BP-PGNN [M] | **DAT [M]** | DAT / BP-GNN | DAT-NB / BP-GNN | DAT-NoSelf / BP-GNN |
|---|---|---|---|---|---|---|---|---|
| MacKay (96,48) | 96 | 2.6 | 4.0 | 3.1 | 2.4 | 0.93 | 1.21 | 0.70 |
| CCSDS (128,64) | 128 | 4.0 | 5.8 | 4.7 | 3.4 | 0.83 | 1.16 | 0.64 |
| MacKay (204,102) | 204 | 5.5 | 11.5 | 6.7 | 5.1 | 0.93 | 1.21 | 0.70 |
| MacKay (408,204) | 408 | 11.0 | 41.1 | 13.8 | 10.2 | 0.93 | 1.21 | 0.70 |
| WiMAX (576,288) | 576 | 16.0 | 88.1 | 20.5 | 14.6 | 0.91 | 1.20 | 0.69 |
| 5G BG2 (128,64) | 128 | 4.2 | 7.5 | 6.0 | 4.0 | 0.94 | 1.21 | 0.71 |
| 5G BG2 (256,128) | 256 | 8.5 | 24.1 | 10.9 | 8.0 | 0.94 | 1.21 | 0.71 |
| 5G BG2 (512,256) | 512 | 16.5 | 93.4 | 20.0 | 15.1 | 0.91 | 1.20 | 0.69 |
| 5G BG1 (704,528) | 704 | 22.5 | 134.0 | 26.6 | 18.2 | 0.81 | 1.14 | 0.62 |
| 5G BG1 (1408,1056) | 1408 | 45.0 | 768.1 | 51.6 | **36.3** | **0.81** | 1.14 | 0.62 |

BP-GNN / BP-SGT / BP-PGNN columns are copied from `PooledGNN_BP/README.md`. All three DAT
variants stay inside the 1.5x BP-GNN budget set for BP-PGNN, which the smoke test asserts.
On 5G 1408, DAT reaches every bit globally for **fewer MACs than BP-GNN** and about 21x fewer
than BP-SGT. Parameters: 29-34k (DAT), 22-26k (DAT-NoSelf), against 22-27k for BP-GNN.

**GPU caveat.** MACs are not latency. On CPU, one DAT stage took 1.4-2.7x BP-GNN's wall time
despite fewer MACs, because it issues more small ops (scatter-softmax, einsums). At small
batch on a T4, the hybrids are launch-bound (see `Latency_Analysis/`), so the op count matters
more than MACs. The final runs record B = 1 latency and throughput next to a same-session
BP(50). Fusing the cross-attention softmax, or capturing the stage in a CUDA graph, is the
obvious next step if the error rates justify it.

## Protocol

Same as BP-PGNN (`pgnn/specs.py` is reused): all 10 codes, the BP-GNN training recipe (BCE after
every BP stage + 0.1 x soft-syndrome loss, hard-example mining, plateau LR, model selection on
BP failures), **80 min of training, 160 min on codes with n >= 512**, and comparison at equal
BP iterations (5x10 vs BP(50), 5x40 vs BP(200)). DAT and BP-PGNN are therefore like-for-like.
Against BP-GNN and BP-SGT (80 min everywhere), the four long codes are not like-for-like, and
the merged table marks them with `*`.

## Running

```
# CPU smoke test (about 2 min): forward/backward of DAT, DAT-NB, DAT-NoSelf on all 10 codes,
# untrained == BP with restarts, MACs <= 1.5x BP-GNN, early-stop decode, CPU stage time
python DualAttention_BP/tests/smoke_test.py

# Kaggle (T4 x2), same accounts and code split as the BP-SGT / BP-PGNN final runs
python DualAttention_BP/kaggle/build_kernel.py          # smoke, final_a..d, abl_nb_a/b, abl_noself_a/b
kaggle kernels push -p DualAttention_BP/kaggle/smoke    # first, CPU, minutes
kaggle kernels push -p DualAttention_BP/kaggle/final_a  # ... final_b, final_c, final_d  (DAT, all 10 codes)
kaggle kernels push -p DualAttention_BP/kaggle/abl_nb_a # optional: DAT-NB and DAT-NoSelf on MacKay 204,
                                                        # WiMAX 576, 5G 704, 5G 1408

# after copying the kernel outputs to DualAttention_BP/results/kaggle_<name>/
python DualAttention_BP/final/merge_dat.py              # -> results/final_comparison/COMPARISON.md
```

The merge also picks up BP-PGNN results from `PooledGNN_BP/results/` if they are there, so one
table compares BP, BP-GNN, BP-SGT, BP-PGNN and BP-DAT.

Local single-code run: `python DualAttention_BP/final/run_dat_final.py --group 5g --codes
5G_BG1_k1056_n1408 [--config dat_nb_config.json] [--quick]`.

## Known limitations

* The linear self-attention sums over all nodes of a frame, so a frame's global context is a
  d_h x d_h matrix per head. That is cheap, but it cannot point to one specific far-away bit the
  way softmax attention can. If DAT-NoSelf matches DAT, this is the likely reason, and a
  Polynormer-style gated or a top-r softmax variant over *checks* (not weak bits) is the next try.
* The sigmoid-kernel denominator sums over n + m nodes in fp32; it is scaled
  by 1/N to stay small under fp16, but no GPU run has tested this.
* The cross branch's scatter-softmax uses `scatter_reduce(amax)` on a detached score, so the
  softmax is exact, but it is an extra pass over E.
* Quasi-cyclic 5G codes: the Z lifted copies of a base-graph column share the type embedding and
  are only told apart by their LLR state and RWSE, as in every other hybrid here.
