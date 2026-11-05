## tune3b_mackay_204_102

GPU: Tesla T4

* SGT r3_w20m50: 68097 parameters, trained 30.12 min on mackay_204_102
* SGT r3_w50m50lr: 68097 parameters, trained 30.03 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_w20m50 | 3.04e-03 | 4.13e-04 | 5.39e-05
SGT r3_w50m50lr | 3.14e-03 | 5.24e-04 | 6.72e-05
BP (50 it) | 2.14e-03 | 3.83e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r3_w20m50 | 2.45e-04 | 2.75e-05 | 3.90e-06
SGT r3_w50m50lr | 2.48e-04 | 3.56e-05 | 4.64e-06
BP (50 it) | 1.71e-04 | 2.34e-05 | 2.56e-06

