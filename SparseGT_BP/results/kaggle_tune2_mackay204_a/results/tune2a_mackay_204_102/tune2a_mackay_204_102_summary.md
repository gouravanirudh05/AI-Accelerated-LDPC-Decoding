## tune2a_mackay_204_102

GPU: Tesla T4

* SGT r2_k30: 68097 parameters, trained 20.13 min on mackay_204_102
* SGT r2_k50: 68097 parameters, trained 20.04 min on mackay_204_102
* SGT r2_k30c: 68161 parameters, trained 20.04 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_k30 | 1.90e-03 | 2.37e-04 | 2.66e-05
SGT r2_k50 | 2.03e-03 | 1.95e-04 | 2.40e-05
SGT r2_k30c | 2.03e-03 | 2.60e-04 | 4.19e-05
BP (50 it) | 2.12e-03 | 3.83e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_k30 | 1.12e-04 | 1.08e-05 | 1.22e-06
SGT r2_k50 | 1.11e-04 | 9.10e-06 | 9.86e-07
SGT r2_k30c | 1.32e-04 | 1.58e-05 | 2.67e-06
BP (50 it) | 1.70e-04 | 2.32e-05 | 2.55e-06

