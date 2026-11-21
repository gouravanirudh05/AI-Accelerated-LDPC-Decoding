"""BP-DAT hybrid: a dual-attention bipartite transformer between BP runs.

Same skeleton as BP-GNN, BP-SGT and BP-PGNN:

    llr -> BP(T) -> DAT -> BP(T) -> DAT -> ... -> BP(T)      (S DAT stages)

The module between two BP runs adapts the *dual-attention* backbone for
bipartite variable-constraint graphs (Huang, Wu, Ma, Wu, Zhang & Song, "A General
Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention", ICML
2026) to the Tanner graph. A Tanner graph is exactly such a graph: bits are the
variables, checks are the (parity) constraints, H is the constraint matrix.
Every layer has two branches:

* **cross branch, sparse** (only on the E edges of H): bits -> checks, then
  checks -> bits. ``cross="attn"`` (the backbone's coefficient-aware cross
  attention): a per-head softmax over the neighbours of the receiver, with the
  score modulated by the edge feature (here the BP messages v2c, c2v and the
  base-graph edge type, in place of the MILP coefficient A_ij).
  ``cross="nb"``: non-backtracking message passing (Park et al., "Non-backtracking
  Graph Neural Networks", NBA-GNN): directed edge states, and the message a
  node sends along an edge excludes what it received along that edge
  (sum minus self, the extrinsic rule of BP, in a d-dimensional learned space);
* **self branch, global and linear** (``self_attn=True``): linear attention of
  every bit over *all* bits and of every check over *all* checks, with the
  backbone's sigmoid kernel  sigma(Q) (sigma(K)^T V) / (sigma(Q) sum sigma(K)).
  Cost O(N d^2) instead of O(N^2 d): no top-k selection, every node takes part.

The two branches are concatenated with the node state and its BP features and
mixed by an MLP, followed by an FFN, with pre-LayerNorm, residuals and dropout
(the GNN+ recipe used by BP-PGNN). The input features, RWSE structural encoding
and zero-initialised head are BP-PGNN's, so an untrained model is exactly BP
with restarts, and the only difference to BP-PGNN is the module itself.
"""

import math

import torch
import torch.nn as nn

from common.bp import bp_iterations, staged_decode
from common.gnn_decoder import llr_features, mlp, soft_syndrome
from common.graph import scatter_sum
from pgnn.pgnn_decoder import FFN, NUM_CN_FEATURES, NUM_VN_FEATURES, rw_encoding


class LinearSelfAttention(nn.Module):
    """Global linear attention within one node type (sigmoid kernel), O(N d^2)."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).unbind(2)
        q, k = torch.sigmoid(q), torch.sigmoid(k)
        kv = torch.einsum("bnhi,bnhj->bhij", k, v) / N           # scaled by 1/N for fp16 safety
        z = k.mean(1)                                             # [B, H, dh]
        num = torch.einsum("bnhi,bhij->bnhj", q, kv)
        den = torch.einsum("bnhi,bhi->bnh", q, z)[..., None]
        return self.out((num / (den + 1e-6)).reshape(B, N, d))

    @staticmethod
    def macs(d, heads, N):
        dh = d // heads
        return N * (3 * d * d + d * d) + 2 * N * heads * dh * dh + N * d


class SparseCrossAttention(nn.Module):
    """Edge-feature-aware attention of every receiver over its Tanner neighbours."""

    def __init__(self, d, d_e, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.e = nn.Linear(d_e, 2 * d)  # score modulation and value term of the edge
        self.out = nn.Linear(d, d)

    def forward(self, h_recv, idx_recv, n_recv, h_send, idx_send, e):
        B, E = e.shape[0], idx_recv.shape[0]
        q = self.q(h_recv)[:, idx_recv]
        k, v = self.kv(h_send)[:, idx_send].chunk(2, -1)
        ew, ev = self.e(e).chunk(2, -1)
        p = (q * k * (1.0 + ew)).view(B, E, self.h, self.dh).sum(-1) / math.sqrt(self.dh)   # [B, E, H]
        idx = idx_recv[None, :, None].expand(B, E, self.h)
        pmax = p.new_full((B, n_recv, self.h), -1e4).scatter_reduce(1, idx, p.detach(), "amax")
        a = torch.exp(p - pmax[:, idx_recv])
        a = a / scatter_sum(a, idx_recv, n_recv)[:, idx_recv].clamp_min(1e-9)
        msg = ((v + ev).view(B, E, self.h, self.dh) * a[..., None]).reshape(B, E, -1)
        return self.out(scatter_sum(msg, idx_recv, n_recv))

    @staticmethod
    def macs(d, d_e, heads, n_recv, n_send, E):
        return n_recv * d * d + n_send * 2 * d * d + E * d_e * 2 * d + E * 4 * d + E * heads + n_recv * d * d


class NonBacktrackingMessage(nn.Module):
    """New directed edge state s_{a->b} = MLP(sum_{x in N(a), x != b} s_{x->a}, h_a, e_ab).

    The extrinsic sum is S_a - s_{b->a}; its linear first layer is evaluated once
    per node and once per edge (W S_a gathered, minus W s_{b->a})."""

    def __init__(self, d, d_e, hidden):
        super().__init__()
        self.wa = nn.Linear(d, hidden)
        self.wh = nn.Linear(d, hidden, bias=False)
        self.we = nn.Linear(d_e, hidden, bias=False)
        self.out = nn.Linear(hidden, d)

    def forward(self, S_send, h_send, idx_send, s_back, e):
        # wa is affine, so wa(S_a - s_{b->a}) = wa(S_a) - W_a s_{b->a}
        z = (self.wa(S_send) + self.wh(h_send))[:, idx_send] - s_back @ self.wa.weight.t() + self.we(e)
        return self.out(torch.nn.functional.silu(z))

    @staticmethod
    def macs(d, d_e, h, n_send, E):
        return n_send * 2 * d * h + E * (d * h + d_e * h + h * d)


class DATLayer(nn.Module):
    """One dual-attention layer: checks update from bits, then bits from checks.

    Within a layer the two Tanner half-steps run in sequence (as in BP-GNN /
    BP-PGNN), so one layer covers bit -> check -> bit; the MILP backbone updates
    both types in parallel."""

    def __init__(self, d, d_e, hidden, heads, dropout, cross, self_attn, nb_hidden, last=False):
        super().__init__()
        self.cross, self.self_attn, self.last = cross, self_attn, last
        self.ln_v, self.ln_c, self.ln_c2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        if cross == "attn":
            self.x_c = SparseCrossAttention(d, d_e, heads)
            self.x_v = SparseCrossAttention(d, d_e, heads)
        else:
            self.nb_cv = NonBacktrackingMessage(d, d_e, nb_hidden)
            # the bit -> check states of the last layer are not read any more
            self.nb_vc = None if last else NonBacktrackingMessage(d, d_e, nb_hidden)
        if self_attn:
            self.sa_c, self.sa_v = LinearSelfAttention(d, heads), LinearSelfAttention(d, heads)
        w = 3 if self_attn else 2
        self.upd_c = mlp(w * d + NUM_CN_FEATURES, hidden, d)
        self.upd_v = mlp(w * d + NUM_VN_FEATURES, hidden, d)
        self.ffn_c, self.ffn_v = FFN(d, 2 * d, dropout), FFN(d, 2 * d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_v, h_c, s_vc, f_v, f_c, e, g, inv_dc, inv_dv):
        hv, hc = self.ln_v(h_v), self.ln_c(h_c)
        # ---- checks <- bits
        if self.cross == "attn":
            x_c = self.x_c(hc, g.e_c, g.m, hv, g.e_v, e)
        else:
            S_c = scatter_sum(s_vc, g.e_c, g.m)
            x_c = S_c * inv_dc
        parts = [hc, x_c, f_c] + ([self.sa_c(hc)] if self.self_attn else [])
        h_c = h_c + self.drop(self.upd_c(torch.cat(parts, -1)))
        h_c = h_c + self.drop(self.ffn_c(h_c))
        hc = self.ln_c2(h_c)
        # ---- bits <- checks
        if self.cross == "attn":
            x_v = self.x_v(hv, g.e_v, g.n, hc, g.e_c, e)
        else:
            s_cv = self.nb_cv(S_c, hc, g.e_c, s_vc, e)            # extrinsic: excludes s_{v->c}
            S_v = scatter_sum(s_cv, g.e_v, g.n)
            x_v = S_v * inv_dv
            s_vc = None if self.nb_vc is None else self.nb_vc(S_v, hv, g.e_v, s_cv, e)
        parts = [hv, x_v, f_v] + ([self.sa_v(hv)] if self.self_attn else [])
        h_v = h_v + self.drop(self.upd_v(torch.cat(parts, -1)))
        h_v = h_v + self.drop(self.ffn_v(h_v))
        return h_v, h_c, s_vc

    @staticmethod
    def macs(d, d_e, h, heads, g, cross, self_attn, nb_h, last=False):
        n, m, E = g.n, g.m, g.num_edges
        if cross == "attn":
            t = SparseCrossAttention.macs(d, d_e, heads, m, n, E) + SparseCrossAttention.macs(d, d_e, heads, n, m, E)
        else:
            t = NonBacktrackingMessage.macs(d, d_e, nb_h, m, E) + E * d * 2   # + the two scatter sums
            if not last:
                t += NonBacktrackingMessage.macs(d, d_e, nb_h, n, E)
        if self_attn:
            t += LinearSelfAttention.macs(d, heads, n) + LinearSelfAttention.macs(d, heads, m)
        w = 3 if self_attn else 2
        t += m * ((w * d + NUM_CN_FEATURES) * h + h * d) + n * ((w * d + NUM_VN_FEATURES) * h + h * d)
        return t + (n + m) * 2 * d * 2 * d


class DualAttnBPDecoder(nn.Module):
    def __init__(self, num_vn_types, num_cn_types, num_edge_types, d=16, d_e=8, hidden=48, heads=2,
                 layers=2, nb_hidden=32, cross="attn", self_attn=True, rw_steps=8, dropout=0.1, stages=4,
                 bp_iters=10, base="prior"):
        super().__init__()
        assert cross in ("attn", "nb") and base in ("prior", "llr") and dropout <= 0.2
        self.stages, self.bp_iters = stages, bp_iters
        self.iters = (stages + 1) * bp_iters
        self.cross, self.self_attn, self.base, self.rw_steps, self.heads = cross, self_attn, base, rw_steps, heads
        self.vn_emb = nn.Embedding(num_vn_types, d)
        self.cn_emb = nn.Embedding(num_cn_types, d)
        self.edge_emb = nn.Embedding(num_edge_types, d_e)
        self.vn_in = nn.Linear(NUM_VN_FEATURES + rw_steps, d)
        self.cn_in = nn.Linear(NUM_CN_FEATURES + rw_steps, d)
        self.edge_in = nn.Linear(2, d_e)
        if cross == "nb":  # initial bit -> check edge states: linear in (bit state, edge feature)
            self.s0_v, self.s0_e = nn.Linear(d, d), nn.Linear(d_e, d, bias=False)
        self.nb_hidden = nb_hidden
        self.layers = nn.ModuleList(DATLayer(d, d_e, hidden, heads, dropout, cross, self_attn, nb_hidden,
                                             last=i == layers - 1) for i in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = mlp(d, hidden, 1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def refine(self, llr, prior, L, v2c, c2v, g):
        """Dual-attention module between two BP runs: stuck BP state -> prior of the next run."""
        hard = (L < 0).to(L.dtype)
        unsat = g.syndrome(hard)
        frac_unsat = scatter_sum(unsat[:, g.e_c], g.e_v, g.n) / g.vn_deg
        f_v = torch.cat([llr_features(llr), llr_features(prior), llr_features(L), frac_unsat[..., None]], -1)
        f_c = torch.stack([soft_syndrome(L, g), 1.0 - 2.0 * unsat], -1)
        B = L.shape[0]
        if self.rw_steps:
            rw_v, rw_c = rw_encoding(g, self.rw_steps)
            xin_v = torch.cat([f_v, rw_v.expand(B, -1, -1)], -1)
            xin_c = torch.cat([f_c, rw_c.expand(B, -1, -1)], -1)
        else:
            xin_v, xin_c = f_v, f_c
        h_v = self.vn_in(xin_v) + self.vn_emb(g.vn_type)
        h_c = self.cn_in(xin_c) + self.cn_emb(g.cn_type)
        e = self.edge_in(torch.stack([torch.tanh((v2c / 2).clamp(-15, 15)),
                                      torch.tanh((c2v / 2).clamp(-15, 15))], -1)) + self.edge_emb(g.edge_type)
        inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
        s_vc = self.s0_v(h_v)[:, g.e_v] + self.s0_e(e) if self.cross == "nb" else None
        for layer in self.layers:
            h_v, h_c, s_vc = layer(h_v, h_c, s_vc, f_v, f_c, e, g, inv_dc, inv_dv)
        delta = self.head(self.ln_out(h_v)).squeeze(-1)
        base = prior if self.base == "prior" else llr
        return (base + delta).clamp(-30, 30)

    def forward(self, llr, g):
        """Differentiable end-to-end pass: marginals of every BP run after a DAT stage."""
        prior, outs = llr, []
        for s in range(self.stages + 1):
            L, v2c, c2v = bp_iterations(prior, g, self.bp_iters)
            if s > 0:
                outs.append(L)
            if s < self.stages:
                prior = self.refine(llr, prior, L, v2c, c2v, g)
        return outs

    @torch.no_grad()
    def decode(self, llr, g, iters=None, early_stop=True):
        return staged_decode(llr, g, self.stages, self.bp_iters, self.refine, early_stop)

    def macs_per_refinement(self, g):
        """Multiply-accumulates of one DAT stage per codeword."""
        d = self.vn_in.out_features
        d_e = self.edge_emb.embedding_dim
        h = self.head[0].out_features
        n, m, E = g.n, g.m, g.num_edges
        total = n * (NUM_VN_FEATURES + self.rw_steps) * d + m * (NUM_CN_FEATURES + self.rw_steps) * d + E * 2 * d_e
        if self.cross == "nb":
            total += n * d * d + E * d_e * d
        total += sum(DATLayer.macs(d, d_e, h, self.heads, g, self.cross, self.self_attn,
                                   self.nb_hidden, l.last)
                     for l in self.layers)
        return int(total + n * (d * h + h))
