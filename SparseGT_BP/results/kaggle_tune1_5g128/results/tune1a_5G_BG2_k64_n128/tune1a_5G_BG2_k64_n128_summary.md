## tune1a_5G_BG2_k64_n128

GPU: Tesla T4

* SGT base: 71137 parameters, trained 20.12 min on 5G_BG2_k64_n128
* SGT k05: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT k20: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT k30: 71137 parameters, trained 20.04 min on 5G_BG2_k64_n128
* SGT L3: 104609 parameters, trained 20.05 min on 5G_BG2_k64_n128
* SGT d32: 19201 parameters, trained 20.04 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT base | 6.29e-03 | 1.20e-03 | 2.49e-04
SGT k05 | 8.71e-03 | 1.97e-03 | 3.64e-04
SGT k20 | 6.23e-03 | 1.20e-03 | 2.44e-04
SGT k30 | 4.84e-03 | 9.40e-04 | 1.63e-04
SGT L3 | 7.08e-03 | 1.55e-03 | 3.21e-04
SGT d32 | 6.77e-03 | 1.38e-03 | 2.47e-04
BP (50 it) | 3.36e-03 | 4.98e-04 | 6.55e-05

BER | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|
SGT base | 1.06e-03 | 1.82e-04 | 3.96e-05
SGT k05 | 1.15e-03 | 2.41e-04 | 4.48e-05
SGT k20 | 1.08e-03 | 1.93e-04 | 4.10e-05
SGT k30 | 7.71e-04 | 1.63e-04 | 2.74e-05
SGT L3 | 1.17e-03 | 2.15e-04 | 4.19e-05
SGT d32 | 1.10e-03 | 1.70e-04 | 3.35e-05
BP (50 it) | 6.26e-04 | 8.42e-05 | 1.07e-05

