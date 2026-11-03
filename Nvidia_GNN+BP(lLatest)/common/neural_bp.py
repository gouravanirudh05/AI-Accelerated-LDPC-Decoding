"""Unrolled neural BP (weighted sum-product, Nachmani et al. 2016/2018).

Each of the ``iters`` unrolled iterations multiplies the CN->VN messages by
learned weights (initialised to 1, i.e. plain BP). Two parametrisations:

* per_edge=True : one weight per edge and iteration (the classic "neural BP",
  tied to one specific parity-check matrix, cannot be applied to another code);
* per_edge=False: one weight per edge *type* and iteration; edge types are 0 for
  classical codes (-> a learned scalar per iteration) and base-graph entries for
  5G (protograph weights, transferable across lifting sizes Z).
"""

import torch
import torch.nn as nn

from .bp import _cn_spa
from .graph import scatter_sum


class NeuralBPDecoder(nn.Module):
    def __init__(self, num_edges, num_edge_types, iters=10, per_edge=True, **_):
        super().__init__()
        self.iters, self.per_edge = iters, per_edge
        self.num_edges = num_edges
        self.w = nn.Parameter(torch.ones(iters, num_edges if per_edge else num_edge_types))
        self.trained_on = None  # set by the experiment runner (per-edge weights are code specific)

    def compatible(self, code):
        return not self.per_edge or (self.trained_on is not None and code.name in self.trained_on)

    def _weights(self, t, g):
        return self.w[t] if self.per_edge else self.w[t][g.edge_type]

    def forward(self, llr, g):
        v2c = llr[:, g.e_v]
        outs = []
        for t in range(self.iters):
            c2v = _cn_spa(v2c, g, eps=1e-6) * self._weights(t, g)
            total = llr + scatter_sum(c2v, g.e_v, g.n)
            v2c = total[:, g.e_v] - c2v
            outs.append(total)
        return outs

    @torch.no_grad()
    def decode(self, llr, g, iters=None, early_stop=True):
        B = llr.shape[0]
        out = llr.clone()
        used = torch.full((B,), self.iters, device=llr.device, dtype=torch.long)
        active = torch.arange(B, device=llr.device)
        x = llr
        v2c = x[:, g.e_v]
        for t in range(self.iters):
            c2v = _cn_spa(v2c, g) * self._weights(t, g)
            total = x + scatter_sum(c2v, g.e_v, g.n)
            v2c = total[:, g.e_v] - c2v
            out[active] = total
            if early_stop and t < self.iters - 1:
                ok = (g.syndrome((total < 0).float()) == 0).all(1)
                if ok.any():
                    used[active[ok]] = t + 1
                    keep = ~ok
                    if not keep.any():
                        break
                    active, x, v2c = active[keep], x[keep], v2c[keep]
        return out, used

    def macs_per_iteration(self, g):
        return int(g.num_edges)  # one extra multiply per edge over plain BP
