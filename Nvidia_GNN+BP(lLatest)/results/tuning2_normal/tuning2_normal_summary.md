## tuning2_normal

GPU: Tesla T4

* BP-GNN 5x10, BP(50)-failure mining: 22561 parameters, trained 40.13 min on mackay_204_102
* BP-GNN 5x20, BP(50)-failure mining: 22561 parameters, trained 40.06 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|
BP-GNN 5x10, BP(50)-failure mining | 1.61e-02 | 2.54e-03 | 2.94e-04 | 3.74e-05 | 3.00e-06
BP-GNN 5x20, BP(50)-failure mining | 1.29e-02 | 2.31e-03 | 3.33e-04 | 6.03e-05 | 7.33e-06
BP (50 it) | 1.34e-02 | 2.07e-03 | 3.94e-04 | 5.94e-05 | 6.99e-06
BP (200 it) | 8.95e-03 | 1.19e-03 | 1.98e-04 | 3.39e-05 | 6.33e-06

BER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|
BP-GNN 5x10, BP(50)-failure mining | 9.51e-04 | 1.33e-04 | 1.43e-05 | 1.64e-06 | 8.49e-08
BP-GNN 5x20, BP(50)-failure mining | 9.54e-04 | 1.37e-04 | 2.00e-05 | 2.95e-06 | 4.83e-07
BP (50 it) | 9.95e-04 | 1.63e-04 | 2.32e-05 | 2.55e-06 | 2.68e-07
BP (200 it) | 7.11e-04 | 8.49e-05 | 1.13e-05 | 1.49e-06 | 1.80e-07

