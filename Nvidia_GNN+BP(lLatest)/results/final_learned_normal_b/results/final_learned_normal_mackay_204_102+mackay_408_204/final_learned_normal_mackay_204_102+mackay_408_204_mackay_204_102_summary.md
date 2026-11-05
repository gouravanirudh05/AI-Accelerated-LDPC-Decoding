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

