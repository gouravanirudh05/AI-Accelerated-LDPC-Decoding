## tune1c_mackay_204_102

GPU: Tesla T4

* SGT nobias: 68033 parameters, trained 20.13 min on mackay_204_102
* SGT h8: 68097 parameters, trained 20.04 min on mackay_204_102
* SGT k15: 68065 parameters, trained 20.04 min on mackay_204_102
* SGT plateau: 68065 parameters, trained 20.04 min on mackay_204_102
* SGT hard50: 68065 parameters, trained 20.04 min on mackay_204_102
* SGT syn0: 68065 parameters, trained 20.04 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT nobias | 5.71e-03 | 8.42e-04 | 1.01e-04
SGT h8 | 3.20e-03 | 5.03e-04 | 8.66e-05
SGT k15 | 3.27e-03 | 4.88e-04 | 6.69e-05
SGT plateau | 4.88e-03 | 6.42e-04 | 8.81e-05
SGT hard50 | 4.48e-03 | 6.92e-04 | 9.65e-05
SGT syn0 | 5.42e-03 | 9.22e-04 | 1.20e-04
BP (50 it) | 2.14e-03 | 3.79e-04 | 5.94e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT nobias | 3.57e-04 | 5.29e-05 | 6.90e-06
SGT h8 | 2.67e-04 | 4.14e-05 | 5.75e-06
SGT k15 | 2.21e-04 | 2.85e-05 | 4.05e-06
SGT plateau | 3.47e-04 | 4.57e-05 | 6.62e-06
SGT hard50 | 3.18e-04 | 4.54e-05 | 5.46e-06
SGT syn0 | 3.62e-04 | 6.70e-05 | 7.88e-06
BP (50 it) | 1.70e-04 | 2.34e-05 | 2.46e-06

