# final_sgt_sgt_5g_5G_BG2_k256_n512

## final_sgt_sgt_5g_5G_BG2_k256_n512_5G_BG2_k256_n512

GPU: Tesla T4

* SGT (5x10 BP it): 71169 parameters, trained 80.15 min on 5G_BG2_k256_n512
* SGT (5x40 BP it): 71169 parameters, trained 80.15 min on 5G_BG2_k256_n512

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
SGT (5x10 BP it) | 6690.4 (256) | 512 | 2739.8 (4096)
SGT (5x40 BP it) | 8357.2 (256) | 256 | 2739.8 (4096)

### 5G_BG2_k256_n512  (n=512, k=256, R=0.5, edges=1888)

BLER | 0.5 dB | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 7.84e-01 | 4.22e-01 | 1.18e-01 | 1.38e-02 | 1.01e-03 | 5.40e-05 | 4.00e-06 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 6.85e-01 | 3.48e-01 | 9.03e-02 | 1.09e-02 | 7.08e-04 | 3.70e-05 | 2.00e-06 | 0.00e+00
SGT (5x40 BP it) | 6.55e-01 | 3.02e-01 | 5.81e-02 | 5.18e-03 | 3.04e-04 | 1.40e-05 | 0.00e+00 | -
SGT (5x40 BP it) + OSD-2 | 6.21e-01 | 2.84e-01 | 5.57e-02 | 4.93e-03 | 2.62e-04 | 1.30e-05 | 0.00e+00 | -
BP (50 it) | 7.03e-01 | 3.38e-01 | 7.81e-02 | 8.97e-03 | 5.11e-04 | 3.40e-05 | 0.00e+00 | -

BER | 0.5 dB | 1.0 dB | 1.5 dB | 2.0 dB | 2.5 dB | 3.0 dB | 3.5 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 1.08e-01 | 5.13e-02 | 1.23e-02 | 1.40e-03 | 9.58e-05 | 3.96e-06 | 7.03e-08 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 1.35e-01 | 6.42e-02 | 1.49e-02 | 1.69e-03 | 1.10e-04 | 4.90e-06 | 5.86e-08 | 0.00e+00
SGT (5x40 BP it) | 9.91e-02 | 4.31e-02 | 7.73e-03 | 6.39e-04 | 3.42e-05 | 1.11e-06 | 0.00e+00 | -
SGT (5x40 BP it) + OSD-2 | 1.28e-01 | 5.53e-02 | 1.00e-02 | 8.56e-04 | 4.30e-05 | 1.32e-06 | 0.00e+00 | -
BP (50 it) | 1.01e-01 | 4.66e-02 | 9.34e-03 | 1.07e-03 | 5.75e-05 | 2.25e-06 | 0.00e+00 | -

Inference profile at Eb/N0 = 2.0 dB (MACs / codeword: SGT (5x10 BP it) 93,384,064 per GNN call, SGT (5x40 BP it) 93,384,064 per GNN call; idle GPU power 59.7 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
SGT (5x10 BP it) [fixed] | 36.063 | 1792.80 (B=1024) | 571 | 66.0 | 116440.19 | 11081.18 | 35.02
SGT (5x10 BP it) [fixed, CUDA graph] | 7.408 | 1726.57 (B=1024) | 593 | 65.1 | 109638.43 | 9192.52 | -
SGT (5x10 BP it) [early-stop] | 5.908 | 173.31 (B=1024) | 5908 | 64.2 | 11467.22 | 805.23 | -
SGT (5x10 BP it) + OSD-2 [early-stop] | 4.826 | 461.24 (B=1024) | 2220 | 63.9 | 28969.02 | 1904.19 | -
SGT (5x40 BP it) [fixed] | 110.425 | 2063.05 (B=1024) | 496 | 60.2 | 119359.70 | 940.33 | 93.90
SGT (5x40 BP it) [fixed, CUDA graph] | 20.896 | 2029.41 (B=1024) | 505 | 64.5 | 127856.45 | 9574.43 | -
SGT (5x40 BP it) [early-stop] | 8.138 | 180.12 (B=1024) | 5685 | 63.5 | 12308.89 | 742.20 | -
SGT (5x40 BP it) + OSD-2 [early-stop] | 5.517 | 455.89 (B=1024) | 2246 | 63.2 | 30860.62 | 1718.20 | -
BP (50 it) [fixed] | 22.766 | 91.36 (B=1024) | 11209 | 63.7 | 5677.94 | 359.52 | 13.63
BP (50 it) [fixed, CUDA graph] | 3.423 | 90.89 (B=1024) | 11267 | 63.7 | 5648.82 | 358.22 | -
BP (50 it) [early-stop] | 5.424 | 51.60 (B=1024) | 19843 | 65.4 | 3568.73 | 310.29 | -
