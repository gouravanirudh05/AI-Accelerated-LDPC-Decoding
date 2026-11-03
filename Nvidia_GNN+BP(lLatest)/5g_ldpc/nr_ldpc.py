"""5G NR LDPC codes (3GPP TS 38.212, Sec. 5.2.2 / 5.3.2 / 5.4.2).

Base-graph tables 5G_bg1.csv / 5G_bg2.csv are taken from NVIDIA Sionna
(Apache-2.0), which transcribes Tables 5.3.2-2/3 of 38.212.

Construction for a target (k, n) with redundancy version 0 and BPSK:
* base graph, lifting size Z and set index i_LS as in 38.212 5.2.2 / 5.3.2,
* K_ldpc - k filler (shortening) bits are known zeros -> removed from H,
* the first 2Z systematic bits are never transmitted (punctured, LLR = 0),
* the circular buffer is read from position 2Z skipping fillers until n bits,
* extension parity groups that are not reached are pruned together with their
  (degree-one) check rows, which does not change the code.
Every VN / CN / edge keeps the index of its base-graph column / row / entry,
used by the GNN's protograph-aware embeddings.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.code import Code  # noqa: E402

CODES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codes")

LIFTING_SETS = [
    [2, 4, 8, 16, 32, 64, 128, 256],
    [3, 6, 12, 24, 48, 96, 192, 384],
    [5, 10, 20, 40, 80, 160, 320],
    [7, 14, 28, 56, 112, 224],
    [9, 18, 36, 72, 144, 288],
    [11, 22, 44, 88, 176, 352],
    [13, 26, 52, 104, 208],
    [15, 30, 60, 120, 240],
]
BG_SHAPE = {"bg1": (46, 68), "bg2": (42, 52)}
BG_INFO_COLS = {"bg1": 22, "bg2": 10}
NUM_BG_EDGES = {"bg1": 316, "bg2": 197}


def select_basegraph(k, rate):
    if k <= 292 or (k <= 3824 and rate <= 0.67) or rate <= 0.25:
        return "bg2"
    return "bg1"


def select_lifting(k, bg):
    if bg == "bg1":
        kb = 22
    else:
        kb = 10 if k > 640 else 9 if k > 560 else 8 if k > 192 else 6
    best = None
    for i_ls, s in enumerate(LIFTING_SETS):
        for z in s:
            if kb * z >= k and (best is None or z < best[0]):
                best = (z, i_ls)
    return best


def load_basegraph(bg, i_ls):
    """Shift-value matrix (-1 = zero block) for set index i_ls."""
    rows, cols = BG_SHAPE[bg]
    bm = -np.ones((rows, cols), dtype=np.int64)
    r = 0
    with open(os.path.join(CODES_DIR, f"5G_{bg}.csv")) as f:
        lines = f.read().splitlines()[2:]
    for ln in lines:
        p = ln.split(";")
        if not p[1].strip():
            continue
        if p[0].strip():
            r = int(p[0])
        bm[r, int(p[1])] = int(p[2 + i_ls])
    return bm


def nr_code(k, n, bg=None, device="cpu"):
    rate = k / n
    bg = bg or select_basegraph(k, rate)
    z, i_ls = select_lifting(k, bg)
    bm = load_basegraph(bg, i_ls)
    assert (bm >= 0).sum() == NUM_BG_EDGES[bg]
    kb = BG_INFO_COLS[bg]
    k_ldpc = kb * z
    n_fill = k_ldpc - k
    rows_b, cols_b = bm.shape

    # rate matching (rv0): read from 2Z, skip filler, take n bits
    cols = np.arange(cols_b * z)
    filler = (cols >= k) & (cols < k_ldpc)
    buf = cols[(cols >= 2 * z) & ~filler]
    if n > len(buf):
        raise ValueError("repetition (n > N_cb) not supported")
    tx_cols = buf[:n]
    last_group = tx_cols[-1] // z
    keep_groups = max(last_group + 1, kb + 4)  # at least the 4 core parity groups
    keep_rows_b = keep_groups - kb

    # lift the kept part of the base graph
    m_eff, n_full = keep_rows_b * z, keep_groups * z
    H = np.zeros((m_eff, n_full), dtype=np.uint8)
    edge_id = -np.ones((m_eff, n_full), dtype=np.int64)
    eid = 0
    nz = {}
    for r in range(rows_b):
        for c in range(cols_b):
            if bm[r, c] >= 0:
                nz[(r, c)] = eid
                eid += 1
    im = np.arange(z)
    for r in range(keep_rows_b):
        for c in range(keep_groups):
            if bm[r, c] < 0:
                continue
            s = bm[r, c] % z
            H[r * z + im, c * z + (im + s) % z] = 1
            edge_id[r * z + im, c * z + (im + s) % z] = nz[(r, c)]

    keep_cols = ~filler[:n_full]
    H = H[:, keep_cols]
    edge_id = edge_id[:, keep_cols]
    col_index = cols[:n_full][keep_cols]
    tx_mask = np.isin(col_index, tx_cols)
    vn_type = col_index // z
    cn_type = np.arange(m_eff) // z
    cn, vn = np.nonzero(H)
    edge_type = edge_id[cn, vn]
    info_pos = np.nonzero(col_index < k)[0]
    # parity columns first in the elimination -> information bits stay systematic
    col_order = np.concatenate([np.nonzero(col_index >= k_ldpc)[0], info_pos])
    name = f"5G_{bg.upper()}_k{k}_n{n}"
    meta = dict(bg=bg, Z=z, i_ls=i_ls, k_ldpc=k_ldpc, fillers=n_fill,
                punctured=int((~tx_mask).sum()), base_rows_kept=int(keep_rows_b),
                base_cols_kept=int(keep_groups))
    code = Code(name, H, k=k, tx_mask=tx_mask, info_pos=info_pos, vn_type=vn_type,
                cn_type=cn_type, edge_type=edge_type, col_order=col_order,
                device=device, meta=meta)
    assert code.n_tx == n
    return code
