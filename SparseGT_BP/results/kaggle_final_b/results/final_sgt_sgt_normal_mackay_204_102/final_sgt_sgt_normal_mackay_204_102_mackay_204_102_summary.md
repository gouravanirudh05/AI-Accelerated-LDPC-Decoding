## final_sgt_sgt_normal_mackay_204_102_mackay_204_102

GPU: Tesla T4

* SGT (5x10 BP it): 68097 parameters, trained 80.13 min on mackay_204_102
* SGT (5x40 BP it): 68097 parameters, trained 80.13 min on mackay_204_102

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
SGT (5x10 BP it) | 1384.4 (256) | 2048 | 2972.5 (4096)
SGT (5x40 BP it) | 2057.6 (256) | 1024 | 2972.5 (4096)

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 6.64e-01 | 4.04e-01 | 1.83e-01 | 5.03e-02 | 1.03e-02 | 1.28e-03 | 1.35e-04 | 1.23e-05 | 2.33e-06
SGT (5x10 BP it) + OSD-2 | 3.62e-01 | 1.86e-01 | 6.88e-02 | 1.46e-02 | 2.33e-03 | 2.65e-04 | 3.10e-05 | 3.00e-06 | 6.66e-07
SGT (5x40 BP it) | 5.86e-01 | 3.36e-01 | 1.40e-01 | 3.22e-02 | 6.29e-03 | 7.47e-04 | 7.90e-05 | 6.99e-06 | 2.00e-06
SGT (5x40 BP it) + OSD-2 | 3.68e-01 | 1.84e-01 | 7.10e-02 | 1.35e-02 | 2.24e-03 | 2.26e-04 | 2.66e-05 | 2.33e-06 | 3.33e-07
BP (50 it) | 6.38e-01 | 3.80e-01 | 1.75e-01 | 4.88e-02 | 1.22e-02 | 1.97e-03 | 3.76e-04 | 6.04e-05 | 6.99e-06

BER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 6.87e-02 | 3.72e-02 | 1.55e-02 | 3.73e-03 | 7.57e-04 | 7.16e-05 | 6.98e-06 | 5.19e-07 | 7.84e-08
SGT (5x10 BP it) + OSD-2 | 6.30e-02 | 3.04e-02 | 1.08e-02 | 2.17e-03 | 3.29e-04 | 3.41e-05 | 3.33e-06 | 3.04e-07 | 6.20e-08
SGT (5x40 BP it) | 6.56e-02 | 3.40e-02 | 1.36e-02 | 2.89e-03 | 5.50e-04 | 5.69e-05 | 5.28e-06 | 4.08e-07 | 1.11e-07
SGT (5x40 BP it) + OSD-2 | 6.37e-02 | 3.03e-02 | 1.12e-02 | 2.10e-03 | 3.16e-04 | 2.98e-05 | 2.84e-06 | 2.81e-07 | 2.94e-08
BP (50 it) | 6.67e-02 | 3.57e-02 | 1.54e-02 | 3.97e-03 | 9.69e-04 | 1.37e-04 | 2.30e-05 | 2.67e-06 | 2.51e-07

Inference profile at Eb/N0 = 3.0 dB (MACs / codeword: SGT (5x10 BP it) 11,538,240 per GNN call, SGT (5x40 BP it) 11,538,240 per GNN call; idle GPU power 32.8 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
SGT (5x10 BP it) [fixed] | 38.961 | 1224.14 (B=4096) | 3346 | 64.2 | 18998.62 | 9283.50 | 30.09
SGT (5x10 BP it) [fixed, CUDA graph] | 5.771 | 1321.46 (B=4096) | 3100 | 63.5 | 18913.77 | 9132.82 | -
SGT (5x10 BP it) [early-stop] | 2.601 | 77.22 (B=4096) | 53046 | 44.7 | 2732.80 | 726.07 | -
SGT (5x10 BP it) + OSD-2 [early-stop] | 26.715 | 187.68 (B=4096) | 21824 | 36.6 | 6008.40 | 610.68 | -
SGT (5x40 BP it) [fixed] | 110.542 | 1644.02 (B=4096) | 2491 | 64.0 | 24202.30 | 11790.87 | 89.31
SGT (5x40 BP it) [fixed, CUDA graph] | 19.220 | 1608.72 (B=4096) | 2546 | 64.5 | 23352.99 | 11462.01 | -
SGT (5x40 BP it) [early-stop] | 4.054 | 198.90 (B=4096) | 20593 | 34.5 | 6296.22 | 306.20 | -
SGT (5x40 BP it) + OSD-2 [early-stop] | 3.868 | 315.68 (B=4096) | 12975 | 33.9 | 10012.24 | 316.10 | -
BP (50 it) [fixed] | 26.106 | 119.31 (B=4096) | 34331 | 67.1 | 1804.89 | 921.71 | 14.88
BP (50 it) [fixed, CUDA graph] | 3.297 | 119.65 (B=4096) | 34234 | 65.4 | 1470.43 | 732.40 | -
BP (50 it) [early-stop] | 4.296 | 56.22 (B=4096) | 72852 | 36.7 | 1731.47 | 179.88 | -

