## tune3a_mackay_204_102

GPU: Tesla T4

* SGT r3_k50: 68097 parameters, trained 30.12 min on mackay_204_102
* SGT r3_w50m50: 68097 parameters, trained 30.03 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_k50 | 1.60e-03 | 1.80e-04 | 1.93e-05
SGT r3_w50m50 | 2.47e-03 | 5.47e-04 | 6.20e-05
BP (50 it) | 2.16e-03 | 3.83e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_k50 | 9.83e-05 | 8.89e-06 | 8.07e-07
SGT r3_w50m50 | 1.72e-04 | 3.69e-05 | 5.13e-06
BP (50 it) | 1.71e-04 | 2.35e-05 | 2.55e-06

