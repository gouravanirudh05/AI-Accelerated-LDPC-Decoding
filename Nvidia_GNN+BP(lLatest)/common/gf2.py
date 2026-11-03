"""GF(2) utilities: alist parsing, rank / reduced row echelon form, encoding."""

import numpy as np
import torch


def parse_alist(path):
    """Read an alist file (MacKay / RPTU channel-codes format) into H (m x n, uint8).

    Parsing is line based so it accepts both zero-padded and unpadded lists.
    """
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    n, m = int(lines[0][0]), int(lines[0][1])
    col_lines = lines[4:4 + n]
    row_lines = lines[4 + n:4 + n + m]
    H = np.zeros((m, n), dtype=np.uint8)
    for j, ln in enumerate(col_lines):
        for r in ln:
            r = int(r)
            if r > 0:
                H[r - 1, j] = 1
    # cross-check with the row lists when present
    if len(row_lines) == m:
        H2 = np.zeros_like(H)
        for i, ln in enumerate(row_lines):
            for c in ln:
                c = int(c)
                if c > 0:
                    H2[i, c - 1] = 1
        if not np.array_equal(H, H2):
            raise ValueError(f"{path}: column and row lists disagree")
    return H


def gf2_rref(H, col_order=None):
    """Reduced row echelon form of H over GF(2).

    Columns are visited in ``col_order`` (default: left to right), so the columns
    visited first become pivots. Returns (R, pivots) where R has ``rank`` rows and
    R[i, pivots[i]] = 1 is the only one in that column.
    """
    A = (np.asarray(H) & 1).astype(bool).copy()
    m, n = A.shape
    order = np.arange(n) if col_order is None else np.asarray(col_order)
    pivots = []
    r = 0
    for j in order:
        if r == m:
            break
        rows = np.nonzero(A[r:, j])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        others = np.nonzero(A[:, j])[0]
        others = others[others != r]
        A[others] ^= A[r]
        pivots.append(j)
        r += 1
    return A[:r].astype(np.uint8), np.array(pivots, dtype=np.int64)


def gf2_rank(H):
    return len(gf2_rref(H)[1])


class LinearEncoder:
    """Systematic-style encoder built from the RREF of H.

    Free (non-pivot) columns carry the information bits; pivot bits are computed
    as x_P = R[:, F] u (mod 2). ``col_order`` lets the caller force which columns
    become parity (pivot) positions, e.g. the 5G parity part.
    """

    def __init__(self, H, col_order=None, device="cpu"):
        R, piv = gf2_rref(H, col_order)
        n = H.shape[1]
        free = np.setdiff1d(np.arange(n), piv)
        self.n, self.k, self.rank = n, len(free), len(piv)
        self.info_pos = torch.as_tensor(free, device=device)
        self.pivots = torch.as_tensor(piv, device=device)
        self.A = torch.as_tensor(R[:, free].astype(np.float32), device=device)  # rank x k
        self.H = torch.as_tensor(np.asarray(H, dtype=np.float32), device=device)

    def to(self, device):
        for name in ("info_pos", "pivots", "A", "H"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def encode(self, u):
        """u: [B, k] float {0,1} -> codewords [B, n] float {0,1}."""
        x = torch.zeros(u.shape[0], self.n, device=u.device, dtype=u.dtype)
        x[:, self.info_pos] = u
        x[:, self.pivots] = torch.remainder(u @ self.A.T, 2.0)
        return x

    def random_codewords(self, batch, device):
        u = torch.randint(0, 2, (batch, self.k), device=device).float()
        return self.encode(u)

    def check(self, x):
        """Return True where x satisfies all parity checks."""
        return (torch.remainder(x @ self.H.T, 2.0) == 0).all(dim=1)
