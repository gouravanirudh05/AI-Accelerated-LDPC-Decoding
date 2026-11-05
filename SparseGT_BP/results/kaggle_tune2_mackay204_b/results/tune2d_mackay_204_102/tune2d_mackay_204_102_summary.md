## tune2d_mackay_204_102

GPU: Tesla T4

* SGT r2_w30cos: 68097 parameters, trained 20.13 min on mackay_204_102
* SGT r2_w50c: 68161 parameters, trained 20.04 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30cos | 2.24e-03 | 3.81e-04 | 5.88e-05
SGT r2_w50c | 1.85e-03 | 3.63e-04 | 5.80e-05
BP (50 it) | 2.14e-03 | 3.83e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30cos | 1.47e-04 | 2.24e-05 | 2.58e-06
SGT r2_w50c | 1.33e-04 | 2.28e-05 | 2.50e-06
BP (50 it) | 1.70e-04 | 2.30e-05 | 2.56e-06

