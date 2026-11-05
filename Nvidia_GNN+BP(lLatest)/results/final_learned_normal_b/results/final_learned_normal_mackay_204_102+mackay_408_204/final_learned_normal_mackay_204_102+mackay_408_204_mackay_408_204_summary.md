## final_learned_normal_mackay_204_102+mackay_408_204_mackay_408_204

GPU: Tesla T4

* BP-GNN (5x10 BP it): 22481 parameters, trained 80.04 min on mackay_408_204
* BP-GNN (5x40 BP it): 22481 parameters, trained 80.04 min on mackay_408_204
* NBP per-edge (50 it): 61200 parameters, trained 20.01 min on mackay_408_204
* GNN only (10 it): 11513 parameters, trained 45.08 min on mackay_408_204

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
BP-GNN (5x10 BP it) | 3694.9 (256) | 512 | 3488.7 (4096)
BP-GNN (5x40 BP it) | 5194.9 (256) | 512 | 3488.7 (4096)
NBP per-edge (50 it) | 723.5 (256) | 4096 | 206.4 (4096)
GNN only (10 it) | 4087.3 (256) | 512 | 3358.3 (4096)

### mackay_408_204  (n=408, k=204, R=0.5, edges=1224)

BLER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 7.20e-01 | 3.71e-01 | 1.05e-01 | 1.75e-02 | 1.09e-03 | 6.42e-05 | 3.99e-06 | 0.00e+00
BP-GNN (5x10 BP it) + OSD-2 | 4.29e-01 | 1.64e-01 | 3.27e-02 | 3.36e-03 | 2.22e-04 | 7.49e-06 | 9.99e-07 | 0.00e+00
BP-GNN (5x40 BP it) | 6.03e-01 | 2.71e-01 | 6.47e-02 | 9.68e-03 | 6.16e-04 | 3.35e-05 | 2.00e-06 | 0.00e+00
BP-GNN (5x40 BP it) + OSD-2 | 4.43e-01 | 1.71e-01 | 3.34e-02 | 3.66e-03 | 1.80e-04 | 8.99e-06 | 4.99e-07 | 0.00e+00
NBP per-edge (50 it) | 6.95e-01 | 3.39e-01 | 8.98e-02 | 1.64e-02 | 1.03e-03 | 1.00e-04 | 7.49e-06 | 4.99e-07
NBP per-edge (50 it) + OSD-2 | 4.70e-01 | 1.94e-01 | 4.25e-02 | 5.37e-03 | 4.02e-04 | 1.60e-05 | 9.99e-07 | 0.00e+00
GNN only (10 it) | 9.55e-01 | 7.90e-01 | 4.72e-01 | 1.84e-01 | 4.08e-02 | 6.77e-03 | 1.02e-03 | 1.94e-04
BP (50 it) | 6.59e-01 | 3.19e-01 | 8.86e-02 | 1.83e-02 | 1.38e-03 | 1.77e-04 | 1.50e-05 | 2.50e-06

BER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB
---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 6.26e-02 | 2.69e-02 | 6.34e-03 | 9.14e-04 | 5.06e-05 | 2.07e-06 | 1.22e-07 | 0.00e+00
BP-GNN (5x10 BP it) + OSD-2 | 6.62e-02 | 2.37e-02 | 4.41e-03 | 4.24e-04 | 2.55e-05 | 8.13e-07 | 7.83e-08 | 0.00e+00
BP-GNN (5x40 BP it) | 5.93e-02 | 2.39e-02 | 5.09e-03 | 6.64e-04 | 3.89e-05 | 1.67e-06 | 6.85e-08 | 0.00e+00
BP-GNN (5x40 BP it) + OSD-2 | 6.84e-02 | 2.49e-02 | 4.59e-03 | 4.91e-04 | 2.31e-05 | 1.13e-06 | 3.67e-08 | 0.00e+00
NBP per-edge (50 it) | 6.59e-02 | 2.85e-02 | 6.66e-03 | 1.03e-03 | 5.53e-05 | 3.85e-06 | 2.03e-07 | 7.34e-09
NBP per-edge (50 it) + OSD-2 | 7.09e-02 | 2.74e-02 | 5.55e-03 | 6.53e-04 | 4.85e-05 | 1.87e-06 | 1.03e-07 | 0.00e+00
GNN only (10 it) | 7.66e-02 | 4.25e-02 | 1.71e-02 | 4.43e-03 | 8.03e-04 | 8.50e-05 | 9.19e-06 | 1.15e-06
BP (50 it) | 6.00e-02 | 2.59e-02 | 6.38e-03 | 1.17e-03 | 7.33e-05 | 7.67e-06 | 3.87e-07 | 4.16e-08

Inference profile at Eb/N0 = 2.75 dB (MACs / codeword: BP-GNN (5x10 BP it) 11,012,736 per GNN call, BP-GNN (5x40 BP it) 11,012,736 per GNN call, NBP per-edge (50 it) 1,224 per GNN call, GNN only (10 it) 5,552,064 per GNN call; idle GPU power 31.4 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
BP-GNN (5x10 BP it) [fixed] | 39.837 | 2093.69 (B=4096) | 1956 | 64.2 | 32744.32 | 16731.02 | 34.10
BP-GNN (5x10 BP it) [fixed, CUDA graph] | 5.759 | 2057.33 (B=4096) | 1991 | 65.9 | 32504.88 | 17014.41 | -
BP-GNN (5x10 BP it) [early-stop] | 4.003 | 101.08 (B=4096) | 40521 | 48.4 | 3083.96 | 1084.51 | -
BP-GNN (5x10 BP it) + OSD-2 [early-stop] | 5.689 | 292.25 (B=4096) | 14015 | 37.3 | 9561.96 | 1505.23 | -
BP-GNN (5x40 BP it) [fixed] | 114.104 | 2889.37 (B=4096) | 1418 | 64.9 | 45154.05 | 23288.16 | 88.24
BP-GNN (5x40 BP it) [fixed, CUDA graph] | 19.313 | 2823.63 (B=4096) | 1451 | 65.4 | 45244.14 | 23517.51 | -
BP-GNN (5x40 BP it) [early-stop] | 3.372 | 191.83 (B=4096) | 21352 | 34.5 | 5775.68 | 521.45 | -
BP-GNN (5x40 BP it) + OSD-2 [early-stop] | 2.950 | 393.38 (B=4096) | 10412 | 37.1 | 1946.14 | 298.79 | -
NBP per-edge (50 it) [fixed] | 25.348 | 246.35 (B=4096) | 16627 | 66.2 | 4095.59 | 2153.86 | 16.59
NBP per-edge (50 it) [fixed, CUDA graph] | 3.409 | 248.94 (B=4096) | 16454 | 65.2 | 4054.38 | 2100.62 | -
NBP per-edge (50 it) [early-stop] | 3.260 | 69.44 (B=4096) | 58989 | 46.0 | 2109.88 | 668.35 | -
NBP per-edge (50 it) + OSD-2 [early-stop] | 2.591 | 264.87 (B=4096) | 15464 | 33.9 | 8328.22 | 619.57 | -
GNN only (10 it) [fixed] | 16.452 | 2104.41 (B=4096) | 1946 | 64.2 | 32539.88 | 16619.10 | 18.29
GNN only (10 it) [fixed, CUDA graph] | 2.166 | 2049.30 (B=4096) | 1999 | 65.2 | 31902.18 | 16539.28 | -
GNN only (10 it) [early-stop] | 12.225 | 1278.96 (B=4096) | 3203 | 65.6 | 20069.64 | 10459.41 | -
BP (50 it) [fixed] | 23.082 | 238.34 (B=4096) | 17185 | 66.2 | 3950.78 | 2076.12 | 15.58
BP (50 it) [fixed, CUDA graph] | 3.298 | 237.51 (B=4096) | 17246 | 66.1 | 3911.96 | 2053.45 | -
BP (50 it) [early-stop] | 3.060 | 65.56 (B=4096) | 62481 | 45.7 | 1828.73 | 571.14 | -

