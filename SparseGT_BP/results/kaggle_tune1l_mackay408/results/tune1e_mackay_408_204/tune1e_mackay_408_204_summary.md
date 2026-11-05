## tune1e_mackay_408_204

GPU: Tesla T4

* SGT base: 68065 parameters, trained 20.16 min on mackay_408_204
* SGT k05: 68065 parameters, trained 20.04 min on mackay_408_204
* SGT k20: 68065 parameters, trained 20.04 min on mackay_408_204

### mackay_408_204  (n=408, k=204, R=0.5, edges=1224)

BLER | 3.0 dB | 3.2 dB | 3.5 dB
---|---|---|---|
SGT base | 4.15e-03 | 1.46e-03 | 3.13e-04
SGT k05 | 4.15e-03 | 1.44e-03 | 4.00e-04
SGT k20 | 2.44e-03 | 7.40e-04 | 2.35e-04
BP (50 it) | 1.97e-03 | 4.70e-04 | 1.57e-04

BER | 3.0 dB | 3.2 dB | 3.5 dB
---|---|---|---|
SGT base | 1.78e-04 | 6.70e-05 | 1.53e-05
SGT k05 | 1.96e-04 | 6.64e-05 | 2.10e-05
SGT k20 | 1.43e-04 | 4.37e-05 | 1.11e-05
BP (50 it) | 1.05e-04 | 2.21e-05 | 6.69e-06

