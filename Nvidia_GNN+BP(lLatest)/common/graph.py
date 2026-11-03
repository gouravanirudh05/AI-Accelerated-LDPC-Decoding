"""Tanner graph representation shared by the GNN and BP decoders."""

import numpy as np
import torch


class TannerGraph:
    """Edge-list view of a parity-check matrix.

    Node/edge "types" index learned embeddings in the GNN. For classical codes
    they are node degrees; for 5G codes they are base-graph rows/columns/edges
    (protograph-aware embedding, independent of the lifting size Z).
    """

    def __init__(self, H, vn_type=None, cn_type=None, edge_type=None, device="cpu"):
        H = np.asarray(H, dtype=np.uint8)
        self.m, self.n = H.shape
        cn, vn = np.nonzero(H)  # edges in check-major order
        self.num_edges = len(vn)
        vn_deg = H.sum(0).astype(np.int64)
        cn_deg = H.sum(1).astype(np.int64)
        if vn_type is None:
            vn_type = vn_deg
        if cn_type is None:
            cn_type = cn_deg
        if edge_type is None:
            edge_type = np.zeros(self.num_edges, dtype=np.int64)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
        self.e_v, self.e_c = t(vn), t(cn)
        self.vn_deg = torch.as_tensor(vn_deg, dtype=torch.float32, device=device)
        self.cn_deg = torch.as_tensor(cn_deg, dtype=torch.float32, device=device)
        self.vn_type, self.cn_type, self.edge_type = t(vn_type), t(cn_type), t(edge_type)
        self.H = torch.as_tensor(H, dtype=torch.float32, device=device)

    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def syndrome(self, hard):
        """hard: [B, n] {0,1} float -> [B, m] syndrome (0/1)."""
        s = torch.zeros(hard.shape[0], self.m, device=hard.device, dtype=hard.dtype)
        s.index_add_(1, self.e_c, hard[:, self.e_v])
        return torch.remainder(s, 2.0)


def scatter_sum(src, index, size):
    """src: [B, E, ...] -> [B, size, ...] summed by index along dim 1."""
    out = src.new_zeros((src.shape[0], size) + tuple(src.shape[2:]))
    return out.index_add_(1, index, src)
