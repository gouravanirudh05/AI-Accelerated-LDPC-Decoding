## tune2a_5G_BG1_k528_n704

GPU: Tesla T4

* SGT r2_k30: 72193 parameters, trained 20.16 min on 5G_BG1_k528_n704
* SGT r2_k50: 72193 parameters, trained 20.09 min on 5G_BG1_k528_n704
* SGT r2_k30c: 72257 parameters, trained 20.07 min on 5G_BG1_k528_n704

### 5G_BG1_k528_n704  (n=704, k=528, R=0.75, edges=2928)

BLER | 3.2 dB | 3.5 dB | 3.8 dB
---|---|---|---|
SGT r2_k30 | 8.30e-03 | 1.71e-03 | 3.16e-04
SGT r2_k50 | 8.46e-03 | 1.81e-03 | 3.38e-04
SGT r2_k30c | 1.09e-02 | 1.97e-03 | 3.51e-04
BP (50 it) | 3.95e-03 | 1.17e-03 | 1.64e-04

BER | 3.2 dB | 3.5 dB | 3.8 dB
---|---|---|---|
SGT r2_k30 | 4.54e-04 | 9.47e-05 | 1.56e-05
SGT r2_k50 | 4.39e-04 | 8.67e-05 | 1.60e-05
SGT r2_k30c | 5.33e-04 | 7.85e-05 | 1.54e-05
BP (50 it) | 2.61e-04 | 6.89e-05 | 1.00e-05

