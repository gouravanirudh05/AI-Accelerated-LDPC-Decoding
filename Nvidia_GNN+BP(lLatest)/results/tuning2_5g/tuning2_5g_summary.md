## tuning2_5g

GPU: Tesla T4

* BP-GNN 5x10, BP(50)-failure mining: 25377 parameters, trained 40.14 min on 5G_BG2_k64_n128
* BP-GNN 5x20, BP(50)-failure mining: 25377 parameters, trained 40.09 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB | 5.5 dB
---|---|---|---|---|---|---|
BP-GNN 5x10, BP(50)-failure mining | 1.97e-02 | 4.15e-03 | 5.25e-04 | 7.56e-05 | 8.99e-06 | 6.66e-07
BP-GNN 5x20, BP(50)-failure mining | 1.64e-02 | 4.07e-03 | 8.27e-04 | 1.36e-04 | 1.83e-05 | 3.00e-06
BP (50 it) | 1.37e-02 | 3.33e-03 | 4.98e-04 | 6.67e-05 | 1.03e-05 | 9.99e-07
BP (200 it) | 8.87e-03 | 1.45e-03 | 1.91e-04 | 1.73e-05 | 1.67e-06 | 1.33e-06

BER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB | 5.5 dB
---|---|---|---|---|---|---|
BP-GNN 5x10, BP(50)-failure mining | 2.35e-03 | 5.11e-04 | 5.80e-05 | 8.34e-06 | 1.09e-06 | 1.04e-08
BP-GNN 5x20, BP(50)-failure mining | 3.39e-03 | 8.51e-04 | 1.61e-04 | 2.59e-05 | 3.60e-06 | 5.41e-07
BP (50 it) | 2.48e-03 | 6.24e-04 | 8.35e-05 | 1.08e-05 | 1.87e-06 | 1.72e-07
BP (200 it) | 1.58e-03 | 2.62e-04 | 3.48e-05 | 3.36e-06 | 3.12e-07 | 9.89e-08

