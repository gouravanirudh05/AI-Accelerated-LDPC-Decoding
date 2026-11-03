"""A code = parity-check matrix + Tanner graph + encoder + transmission pattern."""

import numpy as np
import torch

from .channel import awgn_llr
from .gf2 import LinearEncoder
from .graph import TannerGraph


class Code:
    def __init__(self, name, H, k=None, tx_mask=None, info_pos=None, vn_type=None,
                 cn_type=None, edge_type=None, col_order=None, device="cpu", meta=None):
        H = np.asarray(H, dtype=np.uint8)
        self.name = name
        self.H = H
        self.m, self.n = H.shape
        self.encoder = LinearEncoder(H, col_order=col_order, device=device)
        self.rank = self.encoder.rank
        self.k = self.encoder.k if k is None else k
        assert self.k == self.encoder.k, f"{name}: dimension {self.encoder.k} != {self.k}"
        self.graph = TannerGraph(H, vn_type, cn_type, edge_type, device=device)
        self.tx_mask = None if tx_mask is None else torch.as_tensor(tx_mask, dtype=torch.bool, device=device)
        self.n_tx = self.n if tx_mask is None else int(np.sum(tx_mask))
        self.rate = self.k / self.n_tx
        pos = self.encoder.info_pos if info_pos is None else info_pos
        self.info_pos = torch.as_tensor(pos, device=device)
        self.meta = meta or {}
        self.device = device

    def to(self, device):
        self.encoder.to(device)
        self.graph.to(device)
        self.info_pos = self.info_pos.to(device)
        if self.tx_mask is not None:
            self.tx_mask = self.tx_mask.to(device)
        self.device = device
        return self

    def sample(self, batch, ebno_db, generator_seed=None, punct_scale=0.0):
        """Random codewords through BPSK/AWGN. ebno_db: float or (lo, hi) range.
        punct_scale: fraction of the channel LLR kept on punctured bits (training hint)."""
        dev = self.device
        if isinstance(ebno_db, (tuple, list)):
            lo, hi = ebno_db
            ebno = lo + (hi - lo) * torch.rand(batch, device=dev)
        else:
            ebno = torch.full((batch,), float(ebno_db), device=dev)
        c = self.encoder.random_codewords(batch, dev)
        return c, awgn_llr(c, ebno, self.rate, self.tx_mask, punct_scale)

    def summary(self):
        return dict(name=self.name, n=self.n_tx, k=self.k, rate=round(self.rate, 4),
                    H_shape=[self.m, self.n], edges=self.graph.num_edges, rank=self.rank,
                    **self.meta)
