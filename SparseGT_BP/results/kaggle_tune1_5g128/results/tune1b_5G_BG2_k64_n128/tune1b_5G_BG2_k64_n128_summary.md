## tune1b_5G_BG2_k64_n128

GPU: Tesla T4

* SGT local: 71137 parameters, trained 20.12 min on 5G_BG2_k64_n128
* SGT unsat: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT replace: 71137 parameters, trained 20.05 min on 5G_BG2_k64_n128
* SGT llrbase: 71137 parameters, trained 20.05 min on 5G_BG2_k64_n128
* SGT lr5e4: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT local | 7.20e-03 | 1.34e-03 | 2.87e-04
SGT unsat | 6.23e-03 | 1.22e-03 | 2.25e-04
SGT replace | 7.39e-03 | 1.54e-03 | 2.68e-04
SGT llrbase | 5.37e-03 | 1.00e-03 | 1.84e-04
SGT lr5e4 | 6.77e-03 | 1.28e-03 | 2.49e-04
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.62e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT local | 1.04e-03 | 1.89e-04 | 4.37e-05
SGT unsat | 1.01e-03 | 1.77e-04 | 3.45e-05
SGT replace | 1.13e-03 | 2.37e-04 | 4.17e-05
SGT llrbase | 8.22e-04 | 1.54e-04 | 2.66e-05
SGT lr5e4 | 9.98e-04 | 1.83e-04 | 3.72e-05
BP (50 it) | 6.17e-04 | 8.54e-05 | 9.93e-06

