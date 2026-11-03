# Larger LDPC codes

Parity-check matrices for training a BP + sparse graph transformer decoder
end to end. Every file comes from a published source, is byte-identical to it
(one exception, noted below), and is checked by `validate_codes.py`.

| File | n | k | rate | girth | VN / CN degrees | Source |
|---|---:|---:|---:|---:|---|---|
| `MACKAY_N204_K102.alist` | 204 | 102 | 0.50 | 6 | 3 / 6 | MacKay `204.33.484` |
| `WIMAX_N576_K288.alist` | 576 | 288 | 0.50 | 6 | {2,3,6} / {6,7} | IEEE 802.16e, via AFF3CT |
| `WIFI_N648_K540.alist` | 648 | 540 | 0.83 | 6 | {2,3,4} / 22 | IEEE 802.11n, via AFF3CT |
| `MACKAY_N816_K410.alist` | 816 | 410 | 0.50 | 6 | 3 / 6 | MacKay `816.3.174` |
| `MACKAY_N1008_K504.alist` | 1008 | 504 | 0.50 | 6 | 3 / 6 | MacKay `504.504.3.504` |
| `PEG_N1008_K504.alist` | 1008 | 504 | 0.50 | 8 | 3 / {5..8} | PEG (Hu et al.), via AFF3CT |
| `ETH10G_N2048_K1723.alist` | 2048 | 1723 | 0.84 | 6 | 6 / 32 | IEEE 802.3an, via AFF3CT |

Sources:

* MacKay, *Encyclopedia of Sparse Graph Codes* —
  <https://www.inference.org.uk/mackay/codes/data.html>
* AFF3CT FEC toolbox matrices —
  <https://github.com/aff3ct/configuration_files/tree/develop/dec/LDPC>

`manifest.json` records each file's URL, the SHA-256 of the downloaded
original and of the stored file, and the validated properties.

## Things to know before using them

* **k is computed, not copied.** alist files do not store k. `816.3.174` is
  listed by MacKay as K=408, but two rows of H are dependent, so the true
  dimension is **410** — use 410/816 as the rate in the Eb/N0 → σ conversion.
* **`ETH10G_N2048_K1723.alist` is the only modified file**: its leading `#`
  comment line was removed so every file is plain alist with no comments.
  The matrix is unchanged.
* **Rejected candidates**: every MacKay n=408 code still downloadable
  (`408.33.844`, `408.3.854`, `408.33.864`) contains 4-cycles despite the
  `33` naming, and AFF3CT's WiMAX rate-5/6 matrix has girth 4. None are here.
* Every file is one adjacency list per line (no comments), so a simple
  line-based alist reader works as well as `validate_codes.read_alist`.

## Reference BER/FER curves

`references/` holds AFF3CT's published simulation results for four of these
codes (AWGN, BPSK). Before training, our BP should reproduce them — that
confirms the code is loaded correctly and BP is right at this scale.

| Reference file | Decoder |
|---|---|
| `LDPC_N576_K288_flooding_SPA_i100_WiMAX.txt` | flooding sum-product, 100 iters |
| `LDPC_N648_K540_hlayered_SPA_i10_Wi-Fi.txt` | layered sum-product, 10 iters |
| `LDPC_N1008_K504_flooding_GALA_i200_MacKay.txt` | Gallager-A (hard), 200 iters |
| `LDPC_N2048_K1723_flooding_SPA_i100_eth_10Gbps.txt` | flooding sum-product, 100 iters |

The WiMAX and 10G-Ethernet curves are directly comparable to our flooding
sum-product BP; the Wi-Fi one uses a layered schedule, and the MacKay one a
hard-decision decoder.

## Validate

```bash
python codes/validate_codes.py --manifest   # pure Python, < 1 s
```
