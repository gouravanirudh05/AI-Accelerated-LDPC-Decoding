## tune3b_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r3_w20m50: 71169 parameters, trained 30.13 min on 5G_BG2_k64_n128
* SGT r3_w50m50lr: 71169 parameters, trained 30.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_w20m50 | 3.20e-03 | 5.19e-04 | 7.56e-05
SGT r3_w50m50lr | 7.20e-03 | 1.26e-03 | 1.85e-04
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.20e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_w20m50 | 5.39e-04 | 8.93e-05 | 1.32e-05
SGT r3_w50m50lr | 1.71e-03 | 2.65e-04 | 4.13e-05
BP (50 it) | 6.28e-04 | 8.61e-05 | 1.00e-05

