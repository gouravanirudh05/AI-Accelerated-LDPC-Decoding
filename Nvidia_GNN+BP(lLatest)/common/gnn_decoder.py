"""GNN message-passing decoder on the Tanner graph.

Follows the architecture of Gong, Cammerer & Renes, "Graph Neural Networks for
Enhanced Decoding of Quantum LDPC Codes" (NVIDIA, 2023) and its classical
predecessor (Cammerer et al., "Graph Neural Networks for Channel Decoding"):

* every variable node (VN) and check node (CN) carries a d-dimensional embedding,
* VN->CN and CN->VN messages are produced by learned MLPs of (sender, receiver,
  edge) embeddings, aggregated by a (degree-normalised) sum,
* node updates are learned MLPs, the weights are shared over all iterations,
* training uses a loss on the soft output of *every* iteration.

Adaptations for classical / 5G LDPC codes (the quantum version decodes from a
syndrome only, here we have soft channel observations):

1. VN input = channel LLR features; CN input = differentiable *soft syndrome*
   prod_v tanh(L_v/2) of the current estimate, recomputed every iteration, so
   the CNs see how well their parity is satisfied (classical counterpart of the
   paper's syndrome conditioning).
2. Residual readout L_v = llr_v + f(h_v): the network learns a correction on top
   of the channel LLR, so it starts from the "uncoded" decision.
3. Node/edge type embeddings: node degrees for classical codes, base-graph
   row/column/edge indices for 5G NR (protograph-aware and independent of the
   lifting size Z, so one model decodes several block lengths).
4. Syndrome-check early stopping with batch compaction at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import scatter_sum


def llr_features(llr):
    return torch.stack([llr.clamp(-20, 20) / 8.0, torch.tanh(llr / 2.0)], dim=-1)


def soft_syndrome(L, g):
    """prod_{v in N(c)} tanh(L_v / 2) per check, computed in the log domain."""
    t = torch.tanh((L / 2.0).clamp(-15, 15))[:, g.e_v]
    logmag = scatter_sum(torch.log(t.abs().clamp_min(1e-12)), g.e_c, g.m)
    neg = scatter_sum((t < 0).to(L.dtype), g.e_c, g.m)
    sign = 1.0 - 2.0 * torch.remainder(neg, 2.0)
    return sign * torch.exp(logmag)


def soft_syndrome_loss(L, g):
    """-log P(check satisfied) averaged over checks; P = (1 + prod tanh) / 2."""
    p = 0.5 * (1.0 + soft_syndrome(L, g))
    return -torch.log(p.clamp_min(1e-7)).mean()


class EdgeMLP(nn.Module):
    """MLP on concat(sender, receiver, edge). The first layer is split per input
    so it is evaluated per node and then gathered (much cheaper than per edge)."""

    def __init__(self, d, d_e, hidden, out):
        super().__init__()
        self.ws = nn.Linear(d, hidden)
        self.wr = nn.Linear(d, hidden, bias=False)
        self.we = nn.Linear(d_e, hidden, bias=False)
        self.out = nn.Linear(hidden, out)

    def forward(self, h_send, idx_send, h_recv, idx_recv, e):
        z = self.ws(h_send)[:, idx_send] + self.wr(h_recv)[:, idx_recv] + self.we(e)
        return self.out(F.silu(z))


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class GNNDecoder(nn.Module):
    def __init__(self, num_vn_types, num_cn_types, num_edge_types, d=16, d_e=8,
                 hidden=48, iters=10, use_soft_syndrome=True, readout_init="zero"):
        super().__init__()
        self.iters = iters
        self.use_soft_syndrome = use_soft_syndrome
        self.vn_emb = nn.Embedding(num_vn_types, d)
        self.cn_emb = nn.Embedding(num_cn_types, d)
        self.edge_emb = nn.Embedding(num_edge_types, d_e)
        self.vn_in = nn.Linear(2, d)
        self.cn_in = nn.Linear(1, d)
        self.msg_vc = EdgeMLP(d, d_e, hidden, d)
        self.msg_cv = EdgeMLP(d, d_e, hidden, d)
        self.upd_c = mlp(2 * d + 1, hidden, d)
        self.upd_v = mlp(2 * d + 2, hidden, d)
        self.readout = mlp(d, hidden, 1)
        # "zero": start as the channel decision; "default": random init (avoids the
        # zero-gradient saddle on 5G, where every check touches a punctured bit)
        if readout_init == "zero":
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)

    def init_state(self, llr, g):
        f_v = llr_features(llr)
        h_v = self.vn_in(f_v) + self.vn_emb(g.vn_type)
        s = soft_syndrome(llr, g) if self.use_soft_syndrome else llr.new_zeros(llr.shape[0], g.m)
        h_c = self.cn_in(s[..., None]) + self.cn_emb(g.cn_type)
        return f_v, h_v, h_c, s

    def step(self, llr, f_v, h_v, h_c, s, g, e, inv_dc, inv_dv):
        m = self.msg_vc(h_v, g.e_v, h_c, g.e_c, e)
        agg_c = scatter_sum(m, g.e_c, g.m) * inv_dc
        h_c = h_c + self.upd_c(torch.cat([h_c, agg_c, s[..., None]], -1))
        m = self.msg_cv(h_c, g.e_c, h_v, g.e_v, e)
        agg_v = scatter_sum(m, g.e_v, g.n) * inv_dv
        h_v = h_v + self.upd_v(torch.cat([h_v, agg_v, f_v], -1))
        L = llr + self.readout(h_v).squeeze(-1)
        if self.use_soft_syndrome:
            s = soft_syndrome(L, g)
        return h_v, h_c, s, L

    def forward(self, llr, g, iters=None):
        """Training forward: returns the list of output LLRs of every iteration."""
        iters = iters or self.iters
        f_v, h_v, h_c, s = self.init_state(llr, g)
        e = self.edge_emb(g.edge_type)
        inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
        outs = []
        for _ in range(iters):
            h_v, h_c, s, L = self.step(llr, f_v, h_v, h_c, s, g, e, inv_dc, inv_dv)
            outs.append(L)
        return outs

    @torch.no_grad()
    def decode(self, llr, g, iters=None, early_stop=True):
        """Inference. Returns (final LLR [B, n], iterations used per frame [B]).

        With ``early_stop`` frames whose hard decision satisfies all checks are
        frozen and removed from the active batch (saves compute and energy).
        """
        iters = iters or self.iters
        B = llr.shape[0]
        out = llr.clone()
        used = torch.full((B,), iters, device=llr.device, dtype=torch.long)
        f_v, h_v, h_c, s = self.init_state(llr, g)
        e = self.edge_emb(g.edge_type)
        inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
        active = torch.arange(B, device=llr.device)
        x = llr
        for t in range(iters):
            h_v, h_c, s, L = self.step(x, f_v, h_v, h_c, s, g, e, inv_dc, inv_dv)
            out[active] = L
            if early_stop and t < iters - 1:
                ok = (g.syndrome((L < 0).float()) == 0).all(1)
                if ok.any():
                    used[active[ok]] = t + 1
                    keep = ~ok
                    if not keep.any():
                        break
                    active, x, f_v, h_v, h_c, s = (a[keep] for a in (active, x, f_v, h_v, h_c, s))
        return out, used

    def macs_per_iteration(self, g):
        """Multiply-accumulates per codeword and iteration (for energy discussion)."""
        d = self.vn_in.out_features
        h = self.upd_c[0].out_features
        d_e = self.edge_emb.embedding_dim
        node_first = (g.n + g.m) * 2 * d * h      # split first layers, both messages
        edge = 2 * g.num_edges * (d_e * h + h * d)  # edge term + second layer
        upd = g.m * ((2 * d + 1) * h + h * d) + g.n * ((2 * d + 2) * h + h * d)
        ro = g.n * (d * h + h)
        return int(node_first + edge + upd + ro)
