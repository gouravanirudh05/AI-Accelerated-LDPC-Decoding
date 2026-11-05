## tune3a_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r3_k50: 71169 parameters, trained 30.14 min on 5G_BG2_k64_n128
* SGT r3_w50m50: 71169 parameters, trained 30.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_k50 | 3.14e-03 | 4.88e-04 | 7.27e-05
SGT r3_w50m50 | 3.63e-03 | 5.68e-04 | 8.11e-05
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.39e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_k50 | 5.34e-04 | 7.32e-05 | 1.30e-05
SGT r3_w50m50 | 6.36e-04 | 9.67e-05 | 1.28e-05
BP (50 it) | 6.14e-04 | 8.32e-05 | 1.06e-05

