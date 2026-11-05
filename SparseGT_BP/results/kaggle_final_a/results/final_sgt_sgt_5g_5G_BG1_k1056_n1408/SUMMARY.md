# final_sgt_sgt_5g_5G_BG1_k1056_n1408

## final_sgt_sgt_5g_5G_BG1_k1056_n1408_5G_BG1_k1056_n1408

GPU: Tesla T4

* SGT (5x10 BP it): 72193 parameters, trained 80.39 min on 5G_BG1_k1056_n1408
* SGT (5x40 BP it): 72193 parameters, trained 80.39 min on 5G_BG1_k1056_n1408

### GPU memory (measured on the training code)

decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)
---|---|---|---
SGT (5x10 BP it) | 4614.4 (32) | 64 | 2508.7 (1024)
SGT (5x40 BP it) | 5257.7 (32) | 64 | 2508.7 (1024)

### 5G_BG1_k1056_n1408  (n=1408, k=1056, R=0.75, edges=5856)

BLER | 2.0 dB | 2.2 dB | 2.5 dB | 2.8 dB | 3.0 dB | 3.2 dB | 3.5 dB | 3.8 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 7.09e-01 | 4.10e-01 | 1.21e-01 | 2.24e-02 | 2.26e-03 | 3.03e-04 | 2.60e-05 | 1.20e-05 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 6.31e-01 | 3.32e-01 | 9.31e-02 | 1.66e-02 | 1.78e-03 | 2.10e-04 | 2.00e-05 | 8.00e-06 | 0.00e+00
SGT (5x40 BP it) | 5.57e-01 | 2.30e-01 | 5.66e-02 | 6.10e-03 | 4.24e-04 | 4.20e-05 | 8.00e-06 | 0.00e+00 | -
SGT (5x40 BP it) + OSD-2 | 5.49e-01 | 2.23e-01 | 5.32e-02 | 6.16e-03 | 3.95e-04 | 3.80e-05 | 6.00e-06 | 0.00e+00 | -
BP (50 it) | 6.05e-01 | 2.95e-01 | 6.77e-02 | 9.49e-03 | 8.42e-04 | 7.60e-05 | 1.20e-05 | 2.00e-06 | 0.00e+00

BER | 2.0 dB | 2.2 dB | 2.5 dB | 2.8 dB | 3.0 dB | 3.2 dB | 3.5 dB | 3.8 dB | 4.0 dB
---|---|---|---|---|---|---|---|---|---|
SGT (5x10 BP it) | 3.47e-02 | 1.66e-02 | 4.38e-03 | 6.81e-04 | 6.57e-05 | 7.14e-06 | 4.64e-07 | 4.05e-07 | 0.00e+00
SGT (5x10 BP it) + OSD-2 | 4.99e-02 | 2.31e-02 | 5.66e-03 | 9.81e-04 | 9.04e-05 | 8.96e-06 | 7.63e-07 | 3.86e-07 | 0.00e+00
SGT (5x40 BP it) | 3.04e-02 | 1.14e-02 | 2.50e-03 | 2.63e-04 | 1.67e-05 | 1.27e-06 | 3.01e-07 | 0.00e+00 | -
SGT (5x40 BP it) + OSD-2 | 4.45e-02 | 1.66e-02 | 3.55e-03 | 3.98e-04 | 2.33e-05 | 1.76e-06 | 3.33e-07 | 0.00e+00 | -
BP (50 it) | 3.18e-02 | 1.32e-02 | 2.86e-03 | 3.68e-04 | 2.77e-05 | 2.03e-06 | 3.09e-07 | 1.51e-08 | 0.00e+00

Inference profile at Eb/N0 = 3.25 dB (MACs / codeword: SGT (5x10 BP it) 768,072,576 per GNN call, SGT (5x40 BP it) 768,072,576 per GNN call; idle GPU power 65.7 W)

decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]
---|---|---|---|---|---|---|---
SGT (5x10 BP it) [fixed] | 38.564 | 5793.16 (B=512) | 88 | 65.9 | 741697.27 | 1927.06 | 217.75
SGT (5x10 BP it) [fixed, CUDA graph] | 16.262 | 5782.76 (B=512) | 89 | 66.2 | 753388.67 | 5991.61 | -
SGT (5x10 BP it) [early-stop] | 5.669 | 175.40 (B=512) | 2919 | 66.8 | 24846.57 | 408.58 | -
SGT (5x10 BP it) + OSD-2 [early-stop] | 7.131 | 187.87 (B=512) | 2725 | 65.9 | 25280.20 | 82.00 | -
SGT (5x40 BP it) [fixed] | 114.030 | 6273.53 (B=512) | 82 | 65.9 | 803455.08 | 3055.30 | 274.04
SGT (5x40 BP it) [fixed, CUDA graph] | 29.648 | 6255.04 (B=512) | 82 | 66.1 | 815179.69 | 5519.23 | -
SGT (5x40 BP it) [early-stop] | 6.192 | 39.29 (B=512) | 13031 | 65.7 | 5157.89 | 2.73 | -
SGT (5x40 BP it) + OSD-2 [early-stop] | 6.300 | 40.41 (B=512) | 12671 | 65.9 | 5202.92 | 19.50 | -
BP (50 it) [fixed] | 23.168 | 148.86 (B=512) | 3439 | 67.4 | 19565.40 | 488.94 | 35.63
BP (50 it) [fixed, CUDA graph] | 3.534 | 148.26 (B=512) | 3453 | 67.1 | 19417.05 | 403.42 | -
BP (50 it) [early-stop] | 5.141 | 35.30 (B=512) | 14504 | 65.9 | 4685.80 | 11.41 | -
