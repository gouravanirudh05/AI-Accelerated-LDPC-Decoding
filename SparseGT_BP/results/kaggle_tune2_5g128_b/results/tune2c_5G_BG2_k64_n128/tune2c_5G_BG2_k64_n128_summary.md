## tune2c_5G_BG2_k64_n128

GPU: Tesla T4

* SGT r2_w30c: 71233 parameters, trained 20.14 min on 5G_BG2_k64_n128
* SGT r2_w20: 71169 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30c | 3.23e-03 | 5.03e-04 | 7.10e-05
SGT r2_w20 | 3.45e-03 | 5.36e-04 | 7.68e-05
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.63e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30c | 5.51e-04 | 8.73e-05 | 1.25e-05
SGT r2_w20 | 6.01e-04 | 8.88e-05 | 1.34e-05
BP (50 it) | 6.17e-04 | 8.41e-05 | 1.06e-05

