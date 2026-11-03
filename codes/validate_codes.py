"""Validate the alist parity-check matrices in this directory.

Pure Python (no numpy) so it runs anywhere. For every ``*.alist`` it checks:

* the header (n, m, max degrees, degree lists) matches the adjacency lists,
* the column-wise and row-wise adjacency lists describe the same matrix,
* no duplicate entries (a repeated index would be a non-binary H),
* the true dimension ``k = n - rank_GF2(H)`` (alist files do not store k,
  and some H matrices have redundant rows, so ``n - m`` can be wrong),
* the girth of the Tanner graph (4-cycles hurt BP; girth >= 6 is expected).

Usage::

    python codes/validate_codes.py            # all codes, prints a table
    python codes/validate_codes.py --manifest # also checks manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_alist(path: Path):
    """Return ``(n, m, cols)`` where ``cols[j]`` is the sorted check list of VN j.

    Handles both alist dialects in the wild: MacKay pads short rows with
    zeros, AFF3CT does not. Indices in the file are 1-based. Lines starting
    with ``#`` (provenance comments in some AFF3CT files) are skipped.
    """
    lines = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
    tokens = [int(t) for ln in lines for t in ln.split()]
    pos = 0

    def take(count):
        nonlocal pos
        chunk = tokens[pos:pos + count]
        if len(chunk) != count:
            raise ValueError(f"{path.name}: truncated file")
        pos += count
        return chunk

    n, m = take(2)
    max_col_deg, max_row_deg = take(2)
    col_deg = take(n)
    row_deg = take(m)

    # Padded files store max_deg entries per line, unpadded store deg entries.
    padded_len = 4 + n + m + n * max_col_deg + m * max_row_deg
    unpadded_len = 4 + n + m + sum(col_deg) + sum(row_deg)
    if len(tokens) == padded_len:
        col_width = [max_col_deg] * n
        row_width = [max_row_deg] * m
    elif len(tokens) == unpadded_len:
        col_width, row_width = col_deg, row_deg
    else:
        raise ValueError(
            f"{path.name}: {len(tokens)} tokens, expected {padded_len} (padded) "
            f"or {unpadded_len} (unpadded)"
        )

    cols, rows = [], []
    for j in range(n):
        entries = [e for e in take(col_width[j]) if e != 0]
        if len(entries) != col_deg[j]:
            raise ValueError(f"{path.name}: column {j} degree mismatch")
        cols.append(sorted(e - 1 for e in entries))
    for i in range(m):
        entries = [e for e in take(row_width[i]) if e != 0]
        if len(entries) != row_deg[i]:
            raise ValueError(f"{path.name}: row {i} degree mismatch")
        rows.append(sorted(e - 1 for e in entries))

    if max(col_deg) != max_col_deg or max(row_deg) != max_row_deg:
        raise ValueError(f"{path.name}: header max degree mismatch")
    for j, c in enumerate(cols):
        if len(set(c)) != len(c) or (c and not 0 <= c[0] <= c[-1] < m):
            raise ValueError(f"{path.name}: bad entries in column {j}")
    for i, r in enumerate(rows):
        if len(set(r)) != len(r) or (r and not 0 <= r[0] <= r[-1] < n):
            raise ValueError(f"{path.name}: bad entries in row {i}")

    edges_from_cols = {(i, j) for j, c in enumerate(cols) for i in c}
    edges_from_rows = {(i, j) for i, r in enumerate(rows) for j in r}
    if edges_from_cols != edges_from_rows:
        raise ValueError(f"{path.name}: column and row lists disagree")
    return n, m, cols


def gf2_rank(n: int, m: int, cols) -> int:
    """Rank of H over GF(2), rows packed as Python ints."""
    rows = [0] * m
    for j, c in enumerate(cols):
        for i in c:
            rows[i] |= 1 << j
    rank = 0
    for bit in range(n):
        mask = 1 << bit
        pivot = next((r for r in range(rank, m) if rows[r] & mask), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(m):
            if r != rank and rows[r] & mask:
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def girth(n: int, m: int, cols) -> int:
    """Shortest cycle in the bipartite Tanner graph (BFS from every VN)."""
    adj = [[] for _ in range(n + m)]  # VNs are 0..n-1, CNs are n..n+m-1
    for j, c in enumerate(cols):
        for i in c:
            adj[j].append(n + i)
            adj[n + i].append(j)
    best = float("inf")
    for src in range(n):
        dist = [-1] * (n + m)
        parent = [-1] * (n + m)
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if 2 * dist[u] >= best:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v:
                    best = min(best, dist[u] + dist[v] + 1)
        if best == 4:  # bipartite minimum
            break
    return int(best) if best != float("inf") else 0


def describe(path: Path) -> dict:
    n, m, cols = read_alist(path)
    rank = gf2_rank(n, m, cols)
    col_deg = [len(c) for c in cols]
    row_deg = [0] * m
    for c in cols:
        for i in c:
            row_deg[i] += 1
    return {
        "file": path.name,
        "n": n,
        "m": m,
        "rank": rank,
        "k": n - rank,
        "rate": round((n - rank) / n, 4),
        "edges": sum(col_deg),
        "vn_degrees": sorted(set(col_deg)),
        "cn_degrees": sorted(set(row_deg)),
        "girth": girth(n, m, cols),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", action="store_true",
                        help="Check results against manifest.json.")
    args = parser.parse_args()

    manifest = {}
    if args.manifest:
        manifest = {c["file"]: c for c in json.loads((HERE / "manifest.json").read_text())["codes"]}

    failures = 0
    header = f"{'file':34s} {'n':>5s} {'m':>5s} {'k':>5s} {'rate':>6s} {'edges':>6s} {'girth':>5s}  VN deg / CN deg"
    print(header)
    print("-" * len(header))
    for path in sorted(HERE.glob("*.alist")):
        try:
            info = describe(path)
        except ValueError as exc:
            print(f"FAIL {exc}")
            failures += 1
            continue
        print(f"{info['file']:34s} {info['n']:5d} {info['m']:5d} {info['k']:5d} "
              f"{info['rate']:6.3f} {info['edges']:6d} {info['girth']:5d}  "
              f"{info['vn_degrees']} / {info['cn_degrees']}")
        expected = manifest.get(path.name)
        if args.manifest:
            if expected is None:
                print(f"  FAIL not listed in manifest.json")
                failures += 1
                continue
            for key in ("n", "k", "girth", "sha256"):
                if expected[key] != info[key]:
                    print(f"  FAIL {key}: manifest {expected[key]} != file {info[key]}")
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
