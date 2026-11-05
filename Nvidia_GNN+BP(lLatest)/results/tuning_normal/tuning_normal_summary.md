## tuning_normal

GPU: Tesla T4

* BP-GNN base (d16,h48,L2): 22561 parameters, trained 40.16 min on mackay_204_102
* BP-GNN large (d32,h96,L3): 123169 parameters, trained 40.05 min on mackay_204_102
* BP-GNN base, low-SNR mining: 22561 parameters, trained 40.04 min on mackay_204_102
* GNN base (d16,h48,10 it): 11593 parameters, trained 40.03 min on mackay_204_102
* GNN large (d32,h96,10 it): 43145 parameters, trained 40.06 min on mackay_204_102
* NBP per-edge (10 it): 6120 parameters, trained 12.56 min on mackay_204_102
* NBP per-edge (50 it): 30600 parameters, trained 20.01 min on mackay_204_102

### mackay_204_102  (n=204, k=102, R=0.5, edges=612)

BLER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|
BP-GNN base (d16,h48,L2) | 1.46e-02 | 1.92e-03 | 2.16e-04 | 2.56e-05 | 5.66e-06
BP-GNN large (d32,h96,L3) | 1.33e-02 | 1.81e-03 | 2.08e-04 | 1.93e-05 | 6.00e-06
BP-GNN base, low-SNR mining | 1.22e-02 | 1.54e-03 | 1.84e-04 | 1.63e-05 | 4.33e-06
GNN base (d16,h48,10 it) | 5.27e-02 | 1.27e-02 | 2.20e-03 | 3.49e-04 | 6.55e-05
GNN large (d32,h96,10 it) | 6.25e-02 | 1.73e-02 | 3.42e-03 | 6.49e-04 | 1.70e-04
NBP per-edge (10 it) | 3.49e-02 | 7.39e-03 | 1.11e-03 | 1.87e-04 | 2.76e-05
NBP per-edge (50 it) | 1.16e-02 | 1.62e-03 | 2.09e-04 | 2.23e-05 | 4.00e-06
BP (50 it) | 1.34e-02 | 2.07e-03 | 3.88e-04 | 6.04e-05 | 7.33e-06

BER | 3.0 dB | 3.5 dB | 4.0 dB | 4.5 dB | 5.0 dB
---|---|---|---|---|---|
BP-GNN base (d16,h48,L2) | 1.03e-03 | 1.38e-04 | 1.92e-05 | 1.82e-06 | 3.20e-07
BP-GNN large (d32,h96,L3) | 8.51e-04 | 1.14e-04 | 1.14e-05 | 9.27e-07 | 2.16e-07
BP-GNN base, low-SNR mining | 7.78e-04 | 8.74e-05 | 9.16e-06 | 7.31e-07 | 1.60e-07
GNN base (d16,h48,10 it) | 2.07e-03 | 3.95e-04 | 6.40e-05 | 8.31e-06 | 1.29e-06
GNN large (d32,h96,10 it) | 2.30e-03 | 6.06e-04 | 9.42e-05 | 1.39e-05 | 2.76e-06
NBP per-edge (10 it) | 1.60e-03 | 3.35e-04 | 4.63e-05 | 6.82e-06 | 7.15e-07
NBP per-edge (50 it) | 8.39e-04 | 1.06e-04 | 1.10e-05 | 1.02e-06 | 1.37e-07
BP (50 it) | 9.97e-04 | 1.65e-04 | 2.34e-05 | 2.57e-06 | 2.84e-07

