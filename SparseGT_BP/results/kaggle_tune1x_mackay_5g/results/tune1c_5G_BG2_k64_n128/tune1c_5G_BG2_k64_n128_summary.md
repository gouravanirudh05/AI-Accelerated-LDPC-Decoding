## tune1c_5G_BG2_k64_n128

GPU: Tesla T4

* SGT nobias: 71105 parameters, trained 20.14 min on 5G_BG2_k64_n128
* SGT h8: 71169 parameters, trained 20.05 min on 5G_BG2_k64_n128
* SGT k15: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT plateau: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT hard50: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT syn0: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT nobias | 6.77e-03 | 1.38e-03 | 2.71e-04
SGT h8 | 6.29e-03 | 1.22e-03 | 2.71e-04
SGT k15 | 6.16e-03 | 1.15e-03 | 2.24e-04
SGT plateau | 7.14e-03 | 1.53e-03 | 3.05e-04
SGT hard50 | 5.37e-03 | 1.19e-03 | 2.27e-04
SGT syn0 | 6.90e-03 | 1.25e-03 | 2.52e-04
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.60e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT nobias | 1.10e-03 | 1.87e-04 | 4.08e-05
SGT h8 | 9.74e-04 | 1.99e-04 | 4.16e-05
SGT k15 | 9.95e-04 | 1.90e-04 | 3.97e-05
SGT plateau | 1.06e-03 | 2.18e-04 | 4.37e-05
SGT hard50 | 8.67e-04 | 1.77e-04 | 3.63e-05
SGT syn0 | 1.15e-03 | 1.72e-04 | 3.89e-05
BP (50 it) | 6.23e-04 | 8.40e-05 | 1.06e-05

