## tuning_5g

GPU: Tesla T4

* BP-GNN base (d16,h48,L2): 25377 parameters, trained 40.14 min on 5G_BG2_k64_n128
* BP-GNN large (d32,h96,L3): 127233 parameters, trained 40.06 min on 5G_BG2_k64_n128
* BP-GNN base, low-SNR mining: 25377 parameters, trained 40.05 min on 5G_BG2_k64_n128
* GNN lr 5e-4: 14409 parameters, trained 40.05 min on 5G_BG2_k64_n128
* GNN no soft-syndrome: 14409 parameters, trained 40.05 min on 5G_BG2_k64_n128
* GNN large, 20 it: 47209 parameters, trained 40.2 min on 5G_BG2_k64_n128
* NBP per-edge (10 it): 4630 parameters, trained 10.47 min on 5G_BG2_k64_n128
* NBP per-edge (50 it): 23150 parameters, trained 20.01 min on 5G_BG2_k64_n128

### 5G_BG2_k64_n128  (n=128, k=64, R=0.5, edges=463)

BLER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB | 5.5 dB
---|---|---|---|---|---|---|
BP-GNN base (d16,h48,L2) | 1.54e-02 | 2.98e-03 | 4.28e-04 | 6.92e-05 | 1.10e-05 | 1.67e-06
BP-GNN large (d32,h96,L3) | 1.50e-02 | 3.08e-03 | 4.88e-04 | 5.95e-05 | 1.07e-05 | 6.66e-07
BP-GNN base, low-SNR mining | 1.34e-02 | 2.88e-03 | 4.70e-04 | 5.57e-05 | 1.13e-05 | 6.66e-07
GNN lr 5e-4 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00
GNN no soft-syndrome | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00
GNN large, 20 it | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 1.00e+00
NBP per-edge (10 it) | 4.47e-02 | 1.39e-02 | 3.51e-03 | 5.87e-04 | 9.25e-05 | 1.50e-05
NBP per-edge (50 it) | 1.33e-02 | 2.79e-03 | 3.64e-04 | 4.64e-05 | 7.99e-06 | 9.99e-07
BP (50 it) | 1.37e-02 | 3.33e-03 | 4.88e-04 | 6.65e-05 | 9.66e-06 | 9.99e-07

BER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB | 5.5 dB
---|---|---|---|---|---|---|
BP-GNN base (d16,h48,L2) | 2.23e-03 | 5.33e-04 | 6.45e-05 | 9.80e-06 | 1.92e-06 | 1.35e-07
BP-GNN large (d32,h96,L3) | 2.43e-03 | 5.28e-04 | 8.52e-05 | 1.10e-05 | 1.88e-06 | 7.29e-08
BP-GNN base, low-SNR mining | 2.17e-03 | 4.79e-04 | 6.80e-05 | 8.25e-06 | 1.97e-06 | 6.25e-08
GNN lr 5e-4 | 2.23e-01 | 2.15e-01 | 2.08e-01 | 2.02e-01 | 1.97e-01 | 1.92e-01
GNN no soft-syndrome | 2.23e-01 | 2.15e-01 | 2.08e-01 | 2.02e-01 | 1.97e-01 | 1.92e-01
GNN large, 20 it | 2.23e-01 | 2.15e-01 | 2.08e-01 | 2.02e-01 | 1.97e-01 | 1.92e-01
NBP per-edge (10 it) | 4.43e-03 | 1.44e-03 | 3.14e-04 | 4.86e-05 | 6.86e-06 | 1.47e-06
NBP per-edge (50 it) | 2.23e-03 | 4.72e-04 | 6.51e-05 | 8.86e-06 | 1.74e-06 | 1.87e-07
BP (50 it) | 2.50e-03 | 6.24e-04 | 8.11e-05 | 1.05e-05 | 1.85e-06 | 1.56e-07

