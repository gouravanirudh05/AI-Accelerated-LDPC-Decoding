# BP + sparse graph transformer (BP-SGT) for LDPC decoding

The NVIDIA BP-GNN hybrid of `Nvidia_GNN+BP(lLatest)` with the GNN between the
BP runs replaced by a **sparse graph transformer that only acts on the k
least-confident bits** of the stuck BP run.

```
llr -> BP(10) -> SGT -> BP(10) -> SGT -> BP(10) -> SGT -> BP(10) -> SGT -> BP(10)
```

## Method (sgt/sgt_decoder.py)

After every BP run that has not converged:

1. **Select** the k = k_frac * n bits with the smallest |L| (`select="unsat"`:
   |L| / (1 + #unsatisfied checks of the bit)).
2. **Tokens**: one per selected bit, built from the stuck BP state relative to
   the bit's current hard decision s = sign(L): |L|, s*llr, s*prior,
   unsatisfied-check fraction and count, fraction of check messages that
   disagree with s, mean s*tanh(c2v/2), frame syndrome weight, confidence rank,
   plus a VN-type embedding (degree for classical codes, base-graph column for
   5G NR, so one model is lifting-size agnostic).
3. **Sparse graph attention** among the k tokens only, O(k^2) and independent of
   the rest of the graph. Each head gets a learned bias from the Tanner graph:
   number of checks two selected bits share and how many of those are
   unsatisfied. `attn="local"` masks attention to pairs sharing a check (2-hop
   Tanner neighbourhood); `attn="global"` lets all k bits attend to each other.
4. **Fix** the selected bits: the next BP run starts from a prior where only the
   k selected bits change (`fix="residual"`: base + s*delta, `fix="replace"`:
   s*delta). `base="prior"` keeps corrections of earlier stages, `base="llr"`
   restarts unselected bits from the channel (as the BP-GNN does). The head is
   zero-initialised, so the untrained model is exactly BP with restarts.
5. **Warm start** (`warm=True`, added after round 1): the next BP run continues
   from the previous check messages and only the k fixed bits' channel priors
   change, so the untrained model is exactly BP(50) and the SGT only adds
   corrections. `warm=False` restarts BP at every stage, as the BP-GNN does.

Training is end to end through all five BP runs (gradients flow through BP),
with the BP-GNN recipe unchanged: BCE on every BP stage after an SGT + soft
syndrome loss, hard-example mining of BP(10) failures in 4 Eb/N0 pools, model
selection on a fixed set of BP failures. Codes, Eb/N0 bands, frame budgets,
evaluation, OSD and profiling (latency, throughput, NVML energy) are imported
from `Nvidia_GNN+BP(lLatest)` (`common/`, `normal_ldpc/`, `5g_ldpc/`, `final/`),
so every number is directly comparable with the BP-GNN results.

## Datasets (same as the BP-GNN final comparison)

MacKay (96,48), CCSDS (128,64), MacKay (204,102), MacKay (408,204), WiMAX
(576,288); 5G NR BG2 (128,64), (256,128), (512,256), BG1 (704,528),
(1408,1056). Random codewords over BPSK/AWGN.

## Hyper-parameters tuned

| hyper-parameter | values |
|---|---|
| k_frac (k = k_frac * n) | 0.05, 0.1, 0.2, 0.3 |
| transformer layers | 2, 3 |
| width d (FF hidden 2d) | 32, 64 |
| attention | global + structure bias, local (shared-check mask) |
| selection score | \|L\|, \|L\| / (1 + unsat) |
| fix mode | residual, replace |
| base of the next prior | previous prior, channel LLR |
| learning rate | 1e-3, 5e-4 |
| attention heads | 4, 8 |
| structure bias | on, off |
| hard-example fraction | 0.75, 0.5 |
| soft-syndrome loss weight | 0.1, 0 |
| BP between stages (round 2) | restart, warm start |

Round 1 changes one factor at a time around the base config (k_frac 0.1, 2
layers, d 64, 4 heads, global, \|L\|, residual, prior, lr 1e-3; 20 min,
plateau schedule) on MacKay (204,102), WiMAX (576,288) and 5G BG2 (128,64),
plus the main factors on MacKay (408,204) and 5G BG1 (704,528); score = mean
BLER / BLER(BP(50)) at high Eb/N0 on the same seeded frames, combined over
codes by geometric mean (results/tune1_RANKING.md). The round-1 "plateau"
config turned out identical to the base config, so the pair measures run-to-run
noise (2.51 vs 2.64 overall, up to 20 % on one code). Round 1 findings: k is
the dominant factor (k_frac 0.05 / 0.1 / 0.15 / 0.2 / 0.3: 3.23 / 2.51 / 2.09 /
1.95 / 1.38, same trend on the larger codes); global attention with the
Tanner-graph bias beats local-only attention and no bias; other factors are
within noise. Round 2 (tuning/round2.json) extends k to 0.5, combines the
small winners and tests warm start, on the three codes + 5G BG1 (704,528).
Round 2 (results/tune2_RANKING.md): warm start reproduces BP(50) (score
~1.00 for every k) but learns almost no corrections; with BP restarts k_frac
0.5 is best (1.315 overall; MacKay (204,102) 0.62). Round 3
(results/tune3_RANKING.md): warm start trained on BP(50) failures gets worse
(1.20-1.24), and restart k_frac 0.5 improves with training time (1.134 at 30
min; MacKay (204,102) 0.51). Winner (`final/sgt_config.json`): BP restarted at
every stage, k_frac 0.5, 2 layers, d 64, 4 heads, global attention with
Tanner-graph bias, |L| selection, residual fix on the previous prior,
hard-example fraction 0.75, lr 1e-3, trained 80 min (plateau) per code.

## Results (results/final_comparison/COMPARISON.md)

Final model trained 80 min per code (same recipe and budget as the BP-GNN),
deep Monte Carlo (>= 100 frame errors or 0.5-3 M frames per point), Tesla T4.
BLER ratio at the two highest Eb/N0 points per code where BP(200) has >= 20
errors (20 points over 10 codes); < 1 = SGT has fewer frame errors:

| comparison | SGT better at | geometric-mean BLER ratio |
|---|---|---|
| SGT 5x10 vs BP-GNN 5x10 | 5 / 20 | 1.11 |
| SGT 5x40 vs BP-GNN 5x40 | 11 / 19 | 0.99 |
| SGT 5x10 + OSD-2 vs BP-GNN 5x10 + OSD-2 | 4 / 19 | 1.05 |
| SGT 5x40 + OSD-2 vs BP-GNN 5x40 + OSD-2 | 9 / 19 | 0.98 |
| SGT 5x10 vs BP(50) | 7 / 20 | 0.98 |
| SGT 5x40 vs BP(200) | 14 / 20 | 0.76 |

* BP-SGT matches the BP-GNN in error rate overall (within the ~+-20 % Monte
  Carlo uncertainty of 100-error points). Clearly better on CCSDS (128,64)
  (5x10: 2.8e-5 vs 6.0e-5 at 5 dB), slightly better on MacKay (96,48) and 5G
  (1408,1056) at 5x40; worse on WiMAX and 5G (512,256) at 5x10.
* Cost relative to BP(50) of the same profiling session: same as the BP-GNN on
  the codes up to n = 408 and 5G n <= 704 (sometimes cheaper, e.g. CCSDS 5x40
  1.2x vs 2.9x BP(50) energy), clearly more on the three longest codes at 5x10
  (5G n=1408 5.2x vs 1.7x, 5G n=512 3.7x vs 2.3x, WiMAX 1.9x vs 1.1x) because
  k_frac = 0.5 makes the k x k x m shared-check product dominate (768 M MACs per
  stage on 5G n=1408 vs 45 M for the GNN). A smaller k (0.2-0.3) is the cheaper
  operating point at ~1.4-1.5x higher BLER in tuning.
* Energy / latency of SGT come from a clean re-profile (final/profile_sgt.py):
  the final runs used one process per T4 and NVML ignores CUDA_VISIBLE_DEVICES,
  so their power readings came from the wrong GPU (BLER and memory unaffected).

## Layout

```
sgt/            decoder (sgt_decoder.py), specs.py (codes, bands, training recipe)
tuning/         run_tuning.py (sweep on one code), merge_tuning.py (ranking over codes)
final/          run_sgt_final.py (tuned model on all 10 codes), profile_sgt.py (clean re-profile),
                merge_sgt.py (merge with the BP-GNN final comparison), sgt_config.json
kaggle/         build_kernel.py -> self-contained kernels (one run queue per T4 of a T4 x2 session)
results/        Kaggle outputs, rankings, final_comparison/COMPARISON.md
```

## Running

```
python tuning/run_tuning.py --quick --code mackay_204_102 --configs base,local   # CPU smoke test
python final/run_sgt_final.py --quick --group 5g --codes 5G_BG2_k64_n128
python kaggle/build_kernel.py --only tune1_mackay204
KAGGLE_API_TOKEN=~/.kaggle/access_token kaggle kernels push -p kaggle/tune1_mackay204
python tuning/merge_tuning.py --round tune1
python final/merge_sgt.py
```
