## tune2c_mackay_204_102

GPU: Tesla T4

* SGT r2_w30c: 68161 parameters, trained 20.13 min on mackay_204_102
* SGT r2_w20: 68097 parameters, trained 20.03 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30c | 2.22e-03 | 4.00e-04 | 5.83e-05
SGT r2_w20 | 2.22e-03 | 3.70e-04 | 4.06e-05
BP (50 it) | 2.14e-03 | 3.79e-04 | 6.03e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT r2_w30c | 1.63e-04 | 2.13e-05 | 2.49e-06
SGT r2_w20 | 1.66e-04 | 2.67e-05 | 2.54e-06
BP (50 it) | 1.67e-04 | 2.27e-05 | 2.54e-06

