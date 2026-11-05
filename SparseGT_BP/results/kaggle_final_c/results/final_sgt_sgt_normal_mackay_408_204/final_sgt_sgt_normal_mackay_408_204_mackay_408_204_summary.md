## final_sgt_sgt_normal_mackay_408_204_mackay_408_204

GPU: Tesla T4

* SGT (5x10 BP it): 68097 parameters, trained 80.13 min on mackay_408_204
* SGT (5x40 BP it): 68097 parameters, trained 80.13 min on mackay_408_204

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
SGT (5x10 BP it) | 3816.4 (256) | 512 | 2761.1 (4096)
SGT (5x40 BP it) | 4932.1 (256) | 512 | 2761.1 (4096)

### mackay_408_204  (n=408, k=204, R=0.5, edges=1224)

BLER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 7.24e-01 | 3.84e-01 | 1.17e-01 | 1.74e-02 | 1.54e-03 | 8.57e-05 | 8.00e-06 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 4.60e-01 | 1.73e-01 | 3.88e-02 | 4.15e-03 | 2.74e-04 | 1.20e-05 | 5.00e-07 | 0.00e+00
SGT (5x40 BP it) | 6.19e-01 | 2.71e-01 | 8.15e-02 | 1.01e-02 | 7.47e-04 | 3.35e-05 | 3.00e-06 | 0.00e+00
SGT (5x40 BP it) + OSD-2 | 4.75e-01 | 1.81e-01 | 4.10e-02 | 3.94e-03 | 2.49e-04 | 1.25e-05 | 1.00e-06 | 0.00e+00
BP (50 it) | 6.67e-01 | 3.20e-01 | 1.01e-01 | 1.64e-02 | 1.81e-03 | 1.56e-04 | 1.25e-05 | 5.00e-07

BER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 6.64e-02 | 2.93e-02 | 7.81e-03 | 9.03e-04 | 6.84e-05 | 2.88e-06 | 2.06e-07 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 7.10e-02 | 2.48e-02 | 5.24e-03 | 5.29e-04 | 3.25e-05 | 1.39e-06 | 4.90e-08 | 0.00e+00
SGT (5x40 BP it) | 6.35e-02 | 2.50e-02 | 6.67e-03 | 6.98e-04 | 4.59e-05 | 1.84e-06 | 1.62e-07 | 0.00e+00
SGT (5x40 BP it) + OSD-2 | 7.38e-02 | 2.68e-02 | 5.63e-03 | 5.21e-04 | 2.97e-05 | 1.38e-06 | 1.05e-07 | 0.00e+00
BP (50 it) | 6.15e-02 | 2.60e-02 | 7.42e-03 | 1.10e-03 | 9.51e-05 | 6.62e-06 | 5.29e-07 | 2.45e-08

Inference profile at Eb/N0 = 2.75 dB (MACs / codeword: SGT (5x10 BP it) 41,137,824 per GNN call, SGT (5x40 BP it) 41,137,824 per GNN call; idle GPU power 32.6 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
SGT (5x10 BP it) [fixed] | 34.188 | 1833.82 (B=2048) | 1117 | 65.0 | 58206.71 | 28979.98 | 27.67
SGT (5x10 BP it) [fixed, CUDA graph] | 6.742 | 2042.82 (B=2048) | 1003 | 64.6 | 62003.42 | 30695.85 | -
SGT (5x10 BP it) [early-stop] | 16.632 | 74.23 (B=2048) | 27589 | 60.0 | 3711.30 | 1691.02 | -
SGT (5x10 BP it) + OSD-2 [early-stop] | 2.717 | 271.08 (B=2048) | 7555 | 38.7 | 9868.36 | 1548.45 | -
SGT (5x40 BP it) [fixed] | 107.306 | 2284.88 (B=2048) | 896 | 63.9 | 72872.27 | 35659.95 | 69.75
SGT (5x40 BP it) [fixed, CUDA graph] | 19.851 | 2294.80 (B=2048) | 892 | 65.2 | 72848.80 | 36356.04 | -
SGT (5x40 BP it) [early-stop] | 3.196 | 281.87 (B=2048) | 7266 | 34.8 | 9639.16 | 587.24 | -
SGT (5x40 BP it) + OSD-2 [early-stop] | 8.864 | 355.25 (B=2048) | 5765 | 34.6 | 11366.41 | 652.05 | -
BP (50 it) [fixed] | 21.982 | 120.88 (B=2048) | 16943 | 66.5 | 4004.25 | 2037.48 | 12.70
BP (50 it) [fixed, CUDA graph] | 3.321 | 122.88 (B=2048) | 16667 | 65.7 | 4014.39 | 2020.00 | -
BP (50 it) [early-stop] | 2.816 | 48.70 (B=2048) | 42053 | 47.8 | 1988.98 | 630.43 | -

