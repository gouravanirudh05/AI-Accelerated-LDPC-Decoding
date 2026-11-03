"""LDPC codes: alist loading and the Tanner-graph edge layout BP runs on.

A code is identified by its file stem in ``codes/`` (e.g. ``WIMAX_N576_K288``).
The dimension ``k`` is computed from the GF(2) rank of H, never parsed from
the name: some published matrices have dependent rows (MacKay 816.3.174 is
listed as K=408 but has k=410), and the rate feeds the Eb/N0 -> sigma map.

Edges are stored once, ordered by check node, as two index vectors
``edge_vn`` / ``edge_cn``. Every BP quantity is a ``[batch, E]`` tensor over
that edge list, and per-node sums are ``index_add`` over one of the vectors,
so one BP iteration is a handful of dense GPU kernels with no Python loop
over checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CODES_DIR = Path(os.environ.get("SPARSE_GT_CODES_DIR", REPO_ROOT / "codes"))


def read_alist(path: str | Path) -> np.ndarray:
    """Parse an alist file into a dense binary ``H`` of shape ``[m, n]``.

    Accepts both dialects (MacKay zero-pads short adjacency lines, AFF3CT
    does not) and skips ``#`` comment lines. Only the column lists are used
    for the matrix; the row lists are cross-checked against them.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
    tokens = np.array([int(t) for ln in lines for t in ln.split()], dtype=np.int64)
    n, m, max_col_deg, max_row_deg = tokens[:4]
    col_deg = tokens[4:4 + n]
    row_deg = tokens[4 + n:4 + n + m]
    body = tokens[4 + n + m:]

    padded = n * max_col_deg + m * max_row_deg
    if body.size == padded:
        col_width, row_width = np.full(n, max_col_deg), np.full(m, max_row_deg)
    elif body.size == col_deg.sum() + row_deg.sum():
        col_width, row_width = col_deg, row_deg
    else:
        raise ValueError(f"{path.name}: body has {body.size} entries, matches neither alist dialect")

    H = np.zeros((m, n), dtype=np.uint8)
    pos = 0
    for j in range(n):
        entries = body[pos:pos + col_width[j]]
        pos += col_width[j]
        H[entries[entries > 0] - 1, j] = 1
    H_rows = np.zeros_like(H)
    for i in range(m):
        entries = body[pos:pos + row_width[i]]
        pos += row_width[i]
        H_rows[i, entries[entries > 0] - 1] = 1

    if not np.array_equal(H, H_rows):
        raise ValueError(f"{path.name}: column and row adjacency lists disagree")
    if not (np.array_equal(H.sum(0), col_deg) and np.array_equal(H.sum(1), row_deg)):
        raise ValueError(f"{path.name}: degree header does not match adjacency lists")
    return H


def gf2_rank(H: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) (rows packed into Python ints)."""
    rows = [int("".join(map(str, r[::-1])), 2) for r in H.astype(np.uint8)]
    rank = 0
    for bit in range(H.shape[1]):
        mask = 1 << bit
        pivot = next((r for r in range(rank, len(rows)) if rows[r] & mask), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r] & mask:
                rows[r] ^= rows[rank]
        rank += 1
    return rank


@dataclass(frozen=True)
class LDPCCode:
    name: str
    H: np.ndarray  # [m, n] uint8
    k: int

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def n(self) -> int:
        return self.H.shape[1]

    @property
    def rate(self) -> float:
        return self.k / self.n

    @property
    def num_edges(self) -> int:
        return int(self.H.sum())

    def edges(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """``(edge_vn, edge_cn)``, one entry per nonzero of H, ordered by check."""
        cn, vn = np.nonzero(self.H)  # row-major, i.e. grouped by check
        return (
            torch.as_tensor(vn, dtype=torch.long, device=device),
            torch.as_tensor(cn, dtype=torch.long, device=device),
        )

    def __repr__(self) -> str:
        return (f"LDPCCode({self.name}: n={self.n}, k={self.k}, m={self.m}, "
                f"rate={self.rate:.3f}, edges={self.num_edges})")


def load_code(name_or_path: str | Path) -> LDPCCode:
    """Load ``codes/<name>.alist`` (or an explicit path)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CODES_DIR / f"{path.name}.alist"
    H = read_alist(path)
    return LDPCCode(name=path.stem, H=H, k=H.shape[1] - gf2_rank(H))


def available_codes() -> list[str]:
    return sorted(p.stem for p in CODES_DIR.glob("*.alist"))
