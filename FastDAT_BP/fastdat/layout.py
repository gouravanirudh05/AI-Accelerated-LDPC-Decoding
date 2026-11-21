"""Padded (ELL) layout of a Tanner graph: every scatter-sum becomes a gather + sum
over a fixed-width neighbour table. No atomics, deterministic, fixed shapes, so
it is CUDA-graph and torch.compile friendly.

Edges are those of TannerGraph (check-major). Tables:
  c_edges [m, dc_max]  edge ids of every check (pad = E, a zero slot)
  v_edges [n, dv_max]  edge ids of every bit    (pad = E)
  c_bits  [m, dc_max]  bit ids of every check   (pad = n, a zero column)
"""

import torch


def _pad_table(index, size, pad_value, values=None):
    E = index.numel()
    dev = index.device
    order = torch.argsort(index, stable=True)
    cnt = torch.bincount(index, minlength=size)
    assert int(cnt.min()) >= 1, "every node needs at least one edge"
    width = int(cnt.max())
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(E, device=dev) - start[index[order]]
    tab = torch.full((size, width), pad_value, dtype=torch.long, device=dev)
    src = order if values is None else values[order]
    tab[index[order], pos] = src
    return tab, tab != pad_value


class Layout:
    def __init__(self, g, agg="padded"):
        assert agg in ("padded", "index_add")
        self.g, self.agg, self.n, self.m, self.E = g, agg, g.n, g.m, g.num_edges
        self.c_edges, self.c_mask = _pad_table(g.e_c, g.m, self.E)
        self.v_edges, self.v_mask = _pad_table(g.e_v, g.n, self.E)
        self.c_bits, _ = _pad_table(g.e_c, g.m, self.n, values=g.e_v)

    # ---- sums over edges grouped by node: [B, E, ...] -> [B, size, ...]
    def _gsum(self, src, table, index, size):
        if self.agg == "index_add":
            return src.new_zeros((src.shape[0], size) + tuple(src.shape[2:])).index_add_(1, index, src)
        pad = (0, 0, 0, 1) if src.dim() == 3 else (0, 1)
        return torch.nn.functional.pad(src, pad)[:, table].sum(2)

    def sum_c(self, src):
        return self._gsum(src, self.c_edges, self.g.e_c, self.m)

    def sum_v(self, src):
        return self._gsum(src, self.v_edges, self.g.e_v, self.n)

    # ---- per-receiver padded view of an edge tensor [B, E, ...] -> [B, size, width, ...]
    def per_c(self, src, fill=0.0):
        return self._padded(src, self.c_edges, fill)

    def per_v(self, src, fill=0.0):
        return self._padded(src, self.v_edges, fill)

    @staticmethod
    def _padded(src, table, fill):
        shape = (src.shape[0], 1) + tuple(src.shape[2:])
        return torch.cat([src, src.new_full(shape, fill)], 1)[:, table]

    def syndrome(self, hard):
        """[B, n] {0,1} -> [B, m] {0,1}, as TannerGraph.syndrome."""
        if self.agg == "index_add":
            return self.g.syndrome(hard)
        h = torch.nn.functional.pad(hard, (0, 1))
        return torch.remainder(h[:, self.c_bits].sum(2), 2.0)
