## tune2d_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r2_w30cos: 71169 parameters, trained 20.13 min on 5G_BG2_k64_n128
* SGT r2_w50c: 71233 parameters, trained 20.05 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30cos | 3.42e-03 | 5.03e-04 | 7.40e-05
SGT r2_w50c | 3.87e-03 | 6.66e-04 | 9.69e-05
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.24e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30cos | 5.61e-04 | 8.15e-05 | 1.26e-05
SGT r2_w50c | 7.31e-04 | 1.35e-04 | 1.89e-05
BP (50 it) | 6.20e-04 | 8.38e-05 | 1.04e-05

