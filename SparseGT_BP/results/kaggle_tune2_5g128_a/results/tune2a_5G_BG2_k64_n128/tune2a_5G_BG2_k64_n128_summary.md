## tune2a_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r2_k30: 71169 parameters, trained 20.17 min on 5G_BG2_k64_n128
* SGT r2_k50: 71169 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT r2_k30c: 71233 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_k30 | 4.60e-03 | 8.46e-04 | 1.55e-04
SGT r2_k50 | 4.11e-03 | 7.11e-04 | 7.58e-05
SGT r2_k30c | 4.11e-03 | 7.32e-04 | 9.39e-05
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.68e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_k30 | 8.06e-04 | 1.49e-04 | 2.82e-05
SGT r2_k50 | 6.36e-04 | 1.07e-04 | 1.29e-05
SGT r2_k30c | 7.13e-04 | 1.08e-04 | 1.64e-05
BP (50 it) | 6.21e-04 | 8.26e-05 | 1.03e-05

