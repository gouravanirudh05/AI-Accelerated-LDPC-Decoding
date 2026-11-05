## final_learned_5g_5G_BG2_k256_n512+5G_BG1_k528_n704_5G_BG2_k256_n512

GPU: Tesla T4

* BP-GNN (5x10 BP it): 25377 parameters, trained 80.13 min on 5G_BG2_k256_n512
* BP-GNN (5x40 BP it): 25377 parameters, trained 80.13 min on 5G_BG2_k256_n512
* NBP per-edge (50 it): 94400 parameters, trained 20.01 min on 5G_BG2_k256_n512
* GNN only (10 it): 14409 parameters, trained 45.19 min on 5G_BG2_k256_n512

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
BP-GNN (5x10 BP it) | 5536.1 (256) | 512 | 5356.1 (4096)
BP-GNN (5x40 BP it) | 7748.6 (256) | 256 | 5356.1 (4096)
NBP per-edge (50 it) | 1067.7 (256) | 2048 | 304.6 (4096)
GNN only (10 it) | 6103.2 (256) | 512 | 5163.1 (4096)

### 5G_BG2_k256_n512  (n=512, k=256, R=0.5, edges=1888)

BLER | 0.5 dB | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 7.79e-01 | 3.89e-01 | 9.94e-02 | 1.04e-02 | 8.42e-04 | 4.38e-05 | 2.99e-06 | 9.96e-07
BP-GNN (5x10 BP it) + OSD-2 | 6.65e-01 | 2.99e-01 | 6.76e-02 | 6.90e-03 | 5.41e-04 | 1.69e-05 | 2.99e-06 | 9.96e-07
BP-GNN (5x40 BP it) | 6.89e-01 | 2.92e-01 | 6.01e-02 | 4.80e-03 | 2.39e-04 | 1.10e-05 | 9.96e-07 | 0.00e+00
BP-GNN (5x40 BP it) + OSD-2 | 6.37e-01 | 2.61e-01 | 5.35e-02 | 4.27e-03 | 2.09e-04 | 1.30e-05 | 9.96e-07 | 0.00e+00
NBP per-edge (50 it) | 7.40e-01 | 3.58e-01 | 9.03e-02 | 8.71e-03 | 4.61e-04 | 1.79e-05 | 1.99e-06 | 9.96e-07
NBP per-edge (50 it) + OSD-2 | 6.99e-01 | 3.23e-01 | 7.47e-02 | 6.90e-03 | 3.37e-04 | 1.40e-05 | 9.96e-07 | 9.96e-07
GNN only (10 it) | 1.00e+00 | 1.00e+00 | 9.88e-01 | 8.96e-01 | 6.83e-01 | 3.84e-01 | 1.74e-01 | 6.71e-02
BP (50 it) | 7.15e-01 | 3.39e-01 | 8.42e-02 | 8.54e-03 | 6.26e-04 | 1.99e-05 | 1.99e-06 | 9.96e-07

BER | 0.5 dB | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|
BP-GNN (5x10 BP it) | 1.05e-01 | 4.56e-02 | 1.03e-02 | 1.13e-03 | 7.90e-05 | 2.24e-06 | 5.45e-08 | 2.34e-08
BP-GNN (5x10 BP it) + OSD-2 | 1.34e-01 | 5.56e-02 | 1.20e-02 | 1.23e-03 | 8.65e-05 | 2.27e-06 | 8.56e-08 | 3.11e-08
BP-GNN (5x40 BP it) | 1.00e-01 | 3.92e-02 | 7.77e-03 | 6.37e-04 | 3.25e-05 | 9.73e-07 | 1.95e-08 | 0.00e+00
BP-GNN (5x40 BP it) + OSD-2 | 1.30e-01 | 4.97e-02 | 9.98e-03 | 7.69e-04 | 3.80e-05 | 1.63e-06 | 1.95e-08 | 0.00e+00
NBP per-edge (50 it) | 1.08e-01 | 4.68e-02 | 1.08e-02 | 1.07e-03 | 5.76e-05 | 1.39e-06 | 2.34e-08 | 1.95e-08
NBP per-edge (50 it) + OSD-2 | 1.39e-01 | 5.82e-02 | 1.29e-02 | 1.20e-03 | 6.03e-05 | 1.69e-06 | 1.95e-08 | 3.11e-08
GNN only (10 it) | 1.49e-01 | 1.09e-01 | 6.73e-02 | 2.97e-02 | 1.02e-02 | 2.49e-03 | 4.77e-04 | 1.34e-04
BP (50 it) | 1.04e-01 | 4.49e-02 | 1.05e-02 | 1.13e-03 | 7.94e-05 | 1.30e-06 | 3.11e-08 | 1.95e-08

Inference profile at Eb/N0 = 2.0 dB (MACs / codeword: BP-GNN (5x10 BP it) 16,540,160 per GNN call, BP-GNN (5x40 BP it) 16,540,160 per GNN call, NBP per-edge (50 it) 1,888 per GNN call, GNN only (10 it) 8,331,264 per GNN call; idle GPU power 29.5 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
BP-GNN (5x10 BP it) [fixed] | 38.945 | 3167.74 (B=4096) | 1293 | 66.5 | 49346.19 | 27456.48 | 34.33
BP-GNN (5x10 BP it) [fixed, CUDA graph] | 6.977 | 3102.94 (B=4096) | 1320 | 66.9 | 49426.39 | 27634.80 | -
BP-GNN (5x10 BP it) [early-stop] | 4.893 | 323.10 (B=4096) | 12677 | 64.8 | 6764.92 | 3684.84 | -
BP-GNN (5x10 BP it) + OSD-2 [early-stop] | 6.093 | 599.70 (B=4096) | 6830 | 55.2 | 20849.68 | 9723.33 | -
BP-GNN (5x40 BP it) [fixed] | 110.892 | 4259.59 (B=4096) | 962 | 67.9 | 69659.57 | 39424.72 | 82.98
BP-GNN (5x40 BP it) [fixed, CUDA graph] | 19.923 | 4265.88 (B=4096) | 960 | 67.5 | 69210.94 | 38978.06 | -
BP-GNN (5x40 BP it) [early-stop] | 5.106 | 259.05 (B=4096) | 15812 | 42.4 | 7634.59 | 2321.99 | -
BP-GNN (5x40 BP it) + OSD-2 [early-stop] | 8.553 | 529.39 (B=4096) | 7737 | 36.9 | 16274.82 | 3283.53 | -
NBP per-edge (50 it) [fixed] | 23.674 | 371.29 (B=4096) | 11032 | 68.1 | 6323.40 | 3586.05 | 15.52
NBP per-edge (50 it) [fixed, CUDA graph] | 3.463 | 371.29 (B=4096) | 11032 | 67.5 | 6223.02 | 3506.44 | -
NBP per-edge (50 it) [early-stop] | 7.600 | 113.52 (B=4096) | 36082 | 61.0 | 3320.82 | 1715.48 | -
NBP per-edge (50 it) + OSD-2 [early-stop] | 6.545 | 384.93 (B=4096) | 10641 | 42.4 | 13676.03 | 4165.79 | -
GNN only (10 it) [fixed] | 14.449 | 3178.28 (B=4096) | 1289 | 67.0 | 49192.87 | 27558.68 | 20.53
GNN only (10 it) [fixed, CUDA graph] | 2.316 | 3107.18 (B=4096) | 1318 | 67.4 | 49243.90 | 27704.83 | -
GNN only (10 it) [early-stop] | 15.765 | 3120.06 (B=4096) | 1313 | 68.2 | 49693.78 | 28223.31 | -
BP (50 it) [fixed] | 21.956 | 355.57 (B=4096) | 11519 | 69.3 | 6146.28 | 3530.96 | 14.24
BP (50 it) [fixed, CUDA graph] | 4.311 | 355.23 (B=4096) | 11531 | 68.0 | 6014.74 | 3406.15 | -
BP (50 it) [early-stop] | 4.026 | 107.77 (B=4096) | 38006 | 60.3 | 2964.55 | 1515.83 | -

