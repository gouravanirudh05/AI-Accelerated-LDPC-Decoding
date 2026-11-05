## tune2b_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r2_w10: 71169 parameters, trained 20.16 min on 5G_BG2_k64_n128
* SGT r2_w30: 71169 parameters, trained 20.05 min on 5G_BG2_k64_n128
* SGT r2_w50: 71169 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w10 | 3.27e-03 | 5.75e-04 | 8.08e-05
SGT r2_w30 | 3.59e-03 | 4.70e-04 | 7.17e-05
SGT r2_w50 | 3.45e-03 | 5.59e-04 | 6.86e-05
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.60e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w10 | 5.62e-04 | 9.78e-05 | 1.46e-05
SGT r2_w30 | 5.89e-04 | 8.05e-05 | 1.33e-05
SGT r2_w50 | 5.92e-04 | 9.28e-05 | 1.17e-05
BP (50 it) | 6.21e-04 | 8.38e-05 | 1.08e-05

