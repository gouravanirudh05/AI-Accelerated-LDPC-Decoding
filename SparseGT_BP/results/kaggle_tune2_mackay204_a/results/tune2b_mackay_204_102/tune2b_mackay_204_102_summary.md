## tune2b_mackay_204_102

GPU: Tesla T4

* SGT r2_w10: 68097 parameters, trained 20.12 min on mackay_204_102
* SGT r2_w30: 68097 parameters, trained 20.04 min on mackay_204_102
* SGT r2_w50: 68097 parameters, trained 20.04 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w10 | 2.18e-03 | 3.34e-04 | 4.84e-05
SGT r2_w30 | 1.80e-03 | 3.56e-04 | 5.87e-05
SGT r2_w50 | 1.88e-03 | 3.29e-04 | 5.95e-05
BP (50 it) | 2.14e-03 | 3.83e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w10 | 1.47e-04 | 2.27e-05 | 2.99e-06
SGT r2_w30 | 1.32e-04 | 2.18e-05 | 2.42e-06
SGT r2_w50 | 1.55e-04 | 1.98e-05 | 2.45e-06
BP (50 it) | 1.70e-04 | 2.33e-05 | 2.50e-06

