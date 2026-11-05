# BP-SGT tuning: tune3

score = mean over Eb/N0 of BLER(config) / BLER(BP(50)) (same seeded frames); lower is better. overall = geometric mean over the codes every config was run on (5G_BG1_k528_n704, 5G_BG2_k64_n128, mackay_204_102, wimax_576_288). hard = best FER on the fixed set of BP(10) failures used for model selection.

config | overrides | overall | 5G_BG1_k528_n704 score (hard) | 5G_BG2_k64_n128 score (hard) | mackay_204_102 score (hard) | wimax_576_288 score (hard)
---|---|---|---|---|---|---|
r3_k50 | `{'k_frac': 0.5}` | 1.134 | 1.787 (0.026) | 1.018 (0.083) | 0.511 (0.075) | 1.779 (0.050)
r3_w20m50 | `{'warm': True, 'k_frac': 0.2, 'mining_bp_iters': 50, 'pool_refresh_every': 50, 'val_hard_max_frames': 4000000}` | 1.203 | 1.492 (0.705) | 1.072 (0.640) | 1.132 (0.640) | 1.155 (0.380)
r3_w50m50 | `{'warm': True, 'k_frac': 0.5, 'mining_bp_iters': 50, 'pool_refresh_every': 50, 'val_hard_max_frames': 4000000}` | 1.236 | 1.402 (0.863) | 1.163 (0.634) | 1.199 (0.684) | 1.192 (0.420)
r3_w50m50lr | `{'warm': True, 'k_frac': 0.5, 'mining_bp_iters': 50, 'pool_refresh_every': 50, 'val_hard_max_frames': 4000000, 'lr': 0.003}` | 2.165 | 4.408 (0.953) | 2.552 (0.672) | 1.318 (0.752) | 1.481 (0.863)

## 5G_BG1_k528_n704: BLER

decoder | 3.25 dB | 3.5 dB | 3.75 dB
---|---|---|---|
BP (50 it) | 3.90e-03 | 1.20e-03 | 1.60e-04
r3_k50 | 7.70e-03 | 1.80e-03 | 3.10e-04
r3_w20m50 | 7.10e-03 | 1.40e-03 | 2.40e-04
r3_w50m50 | 6.20e-03 | 1.50e-03 | 2.30e-04
r3_w50m50lr | 2.00e-02 | 4.10e-03 | 7.70e-04

## 5G_BG2_k64_n128: BLER

decoder | 3.5 dB | 4 dB | 4.5 dB
---|---|---|---|
BP (50 it) | 3.40e-03 | 5.00e-04 | 6.40e-05
r3_k50 | 3.10e-03 | 4.90e-04 | 7.30e-05
r3_w20m50 | 3.20e-03 | 5.20e-04 | 7.60e-05
r3_w50m50 | 3.60e-03 | 5.70e-04 | 8.10e-05
r3_w50m50lr | 7.20e-03 | 1.30e-03 | 1.80e-04

## mackay_204_102: BLER

decoder | 3.5 dB | 4 dB | 4.5 dB
---|---|---|---|
BP (50 it) | 2.20e-03 | 3.80e-04 | 6.00e-05
r3_k50 | 1.60e-03 | 1.80e-04 | 1.90e-05
r3_w20m50 | 3.00e-03 | 4.10e-04 | 5.40e-05
r3_w50m50 | 2.50e-03 | 5.50e-04 | 6.20e-05
r3_w50m50lr | 3.10e-03 | 5.20e-04 | 6.70e-05

## wimax_576_288: BLER

decoder | 2.5 dB | 2.75 dB | 3 dB
---|---|---|---|
BP (50 it) | 1.30e-03 | 3.40e-04 | 6.80e-05
r3_k50 | 2.10e-03 | 6.00e-04 | 1.30e-04
r3_w20m50 | 1.50e-03 | 3.80e-04 | 7.60e-05
r3_w50m50 | 1.60e-03 | 3.60e-04 | 8.20e-05
r3_w50m50lr | 1.90e-03 | 5.00e-04 | 9.60e-05
