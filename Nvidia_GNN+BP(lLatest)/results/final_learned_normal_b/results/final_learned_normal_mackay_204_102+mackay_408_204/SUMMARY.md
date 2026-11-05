# final_learned_normal_mackay_204_102+mackay_408_204

## final_learned_normal_mackay_204_102+mackay_408_204_mackay_204_102

GPU: Tesla T4

* BP-GNN (5x10 BP it): 22481 parameters, trained 80.16 min on mackay_204_102
* BP-GNN (5x40 BP it): 22481 parameters, trained 80.16 min on mackay_204_102
* NBP per-edge (50 it): 30600 parameters, trained 20.01 min on mackay_204_102
* GNN only (10 it): 11513 parameters, trained 45.03 min on mackay_204_102

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
BP-GNN (5x10 BP it) | 1826.1 (256) | 2048 | 1749.3 (4096)
BP-GNN (5x40 BP it) | 2543.2 (256) | 1024 | 1749.3 (4096)
NBP per-edge (50 it) | 347.3 (256) | 8192 | 102.4 (4096)
GNN only (10 it) | 2034.7 (256) | 1024 | 1685.2 (4096)

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 6.50e-01 | 3.88e-01 | 1.79e-01 | 4.79e-02 | 9.44e-03 | 1.16e-03 | 1.35e-04 | 1.33e-05 | 2.00e-06
BP-GNN (5x10 BP it) + OSD-2 | 3.63e-01 | 1.82e-01 | 7.13e-02 | 1.40e-02 | 2.26e-03 | 2.28e-04 | 2.70e-05 | 3.00e-06 | 9.99e-07
BP-GNN (5x40 BP it) | 5.80e-01 | 3.36e-01 | 1.40e-01 | 3.34e-02 | 6.90e-03 | 7.71e-04 | 9.30e-05 | 7.66e-06 | 1.33e-06
BP-GNN (5x40 BP it) + OSD-2 | 3.74e-01 | 1.91e-01 | 7.32e-02 | 1.43e-02 | 2.26e-03 | 2.30e-04 | 3.13e-05 | 3.00e-06 | 3.33e-07
NBP per-edge (50 it) | 6.57e-01 | 3.98e-01 | 1.80e-01 | 4.83e-02 | 1.16e-02 | 1.42e-03 | 1.97e-04 | 1.90e-05 | 2.33e-06
NBP per-edge (50 it) + OSD-2 | 3.45e-01 | 1.80e-01 | 6.96e-02 | 1.73e-02 | 3.04e-03 | 3.79e-04 | 5.03e-05 | 4.33e-06 | 6.66e-07
GNN only (10 it) | 8.45e-01 | 6.15e-01 | 3.70e-01 | 1.48e-01 | 4.93e-02 | 1.00e-02 | 1.90e-03 | 3.13e-04 | 5.64e-05
BP (50 it) | 6.38e-01 | 3.80e-01 | 1.75e-01 | 4.88e-02 | 1.23e-02 | 1.95e-03 | 3.76e-04 | 6.04e-05 | 6.99e-06

BER | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 6.87e-02 | 3.67e-02 | 1.53e-02 | 3.66e-03 | 7.19e-04 | 7.35e-05 | 7.58e-06 | 5.94e-07 | 1.44e-07
BP-GNN (5x10 BP it) + OSD-2 | 6.32e-02 | 2.98e-02 | 1.11e-02 | 2.14e-03 | 3.17e-04 | 2.98e-05 | 2.81e-06 | 3.17e-07 | 9.14e-08
BP-GNN (5x40 BP it) | 6.64e-02 | 3.50e-02 | 1.35e-02 | 2.99e-03 | 5.43e-04 | 5.92e-05 | 6.30e-06 | 4.77e-07 | 2.94e-08
BP-GNN (5x40 BP it) + OSD-2 | 6.53e-02 | 3.15e-02 | 1.16e-02 | 2.20e-03 | 3.16e-04 | 3.17e-05 | 3.56e-06 | 3.76e-07 | 7.51e-08
NBP per-edge (50 it) | 7.00e-02 | 3.80e-02 | 1.59e-02 | 3.84e-03 | 8.98e-04 | 9.39e-05 | 1.16e-05 | 8.52e-07 | 7.84e-08
NBP per-edge (50 it) + OSD-2 | 5.90e-02 | 2.86e-02 | 1.05e-02 | 2.49e-03 | 4.08e-04 | 4.95e-05 | 5.85e-06 | 3.89e-07 | 7.84e-08
GNN only (10 it) | 7.63e-02 | 4.56e-02 | 2.23e-02 | 7.08e-03 | 2.02e-03 | 3.33e-04 | 5.87e-05 | 8.01e-06 | 1.17e-06
BP (50 it) | 6.67e-02 | 3.57e-02 | 1.54e-02 | 3.97e-03 | 9.66e-04 | 1.38e-04 | 2.21e-05 | 2.55e-06 | 2.61e-07

Inference profile at Eb/N0 = 3.0 dB (MACs / codeword: BP-GNN (5x10 BP it) 5,506,368 per GNN call, BP-GNN (5x40 BP it) 5,506,368 per GNN call, NBP per-edge (50 it) 612 per GNN call, GNN only (10 it) 2,776,032 per GNN call; idle GPU power 31.8 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
BP-GNN (5x10 BP it) [fixed] | 41.419 | 1037.19 (B=4096) | 3949 | 64.1 | 15710.21 | 7923.52 | 28.56
BP-GNN (5x10 BP it) [fixed, CUDA graph] | 5.392 | 1041.68 (B=4096) | 3932 | 65.4 | 15826.05 | 8137.00 | -
BP-GNN (5x10 BP it) [early-stop] | 2.651 | 81.31 (B=4096) | 50373 | 42.0 | 2561.01 | 623.02 | -
BP-GNN (5x10 BP it) + OSD-2 [early-stop] | 49.502 | 188.59 (B=4096) | 21719 | 33.8 | 5445.75 | 324.91 | -
BP-GNN (5x40 BP it) [fixed] | 112.304 | 1412.03 (B=4096) | 2901 | 64.3 | 20441.96 | 10335.46 | 77.84
BP-GNN (5x40 BP it) [fixed, CUDA graph] | 19.212 | 1421.53 (B=4096) | 2881 | 63.6 | 20100.71 | 10049.62 | -
BP-GNN (5x40 BP it) [early-stop] | 3.639 | 184.11 (B=4096) | 22247 | 32.3 | 5131.58 | 86.30 | -
BP-GNN (5x40 BP it) + OSD-2 [early-stop] | 3.682 | 283.77 (B=4096) | 14434 | 32.3 | 8467.41 | 123.23 | -
NBP per-edge (50 it) [fixed] | 24.342 | 122.69 (B=4096) | 33384 | 67.1 | 1660.37 | 873.39 | 15.24
NBP per-edge (50 it) [fixed, CUDA graph] | 3.387 | 122.52 (B=4096) | 33431 | 66.7 | 1532.81 | 802.38 | -
NBP per-edge (50 it) [early-stop] | 3.880 | 53.41 (B=4096) | 76695 | 36.0 | 1535.50 | 179.69 | -
NBP per-edge (50 it) + OSD-2 [early-stop] | 1.790 | 156.15 (B=4096) | 26231 | 33.4 | 4701.65 | 232.90 | -
GNN only (10 it) [fixed] | 14.506 | 1037.01 (B=4096) | 3950 | 64.1 | 16023.68 | 8078.74 | 15.41
GNN only (10 it) [fixed, CUDA graph] | 1.577 | 1024.86 (B=4096) | 3997 | 64.4 | 15763.53 | 7982.80 | -
GNN only (10 it) [early-stop] | 4.276 | 466.29 (B=4096) | 8784 | 64.7 | 7553.01 | 3841.95 | -
BP (50 it) [fixed] | 22.541 | 118.32 (B=4096) | 34617 | 65.6 | 1483.24 | 763.91 | 14.86
BP (50 it) [fixed, CUDA graph] | 3.364 | 118.13 (B=4096) | 34672 | 66.0 | 1453.39 | 752.82 | -
BP (50 it) [early-stop] | 2.446 | 51.22 (B=4096) | 79965 | 35.3 | 1479.92 | 148.66 | -


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
