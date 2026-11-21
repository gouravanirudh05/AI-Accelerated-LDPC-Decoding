"""BP-PGNN hybrid: a pooled bipartite GNN between BP runs (replaces BP-SGT's sparse attention).

Same skeleton as BP-GNN (Nvidia_GNN+BP(lLatest)/common/bp_gnn_decoder.py) and
BP-SGT (SparseGT_BP/sgt/sgt_decoder.py):

    llr -> BP(T) -> PGNN -> BP(T) -> PGNN -> ... -> BP(T)      (S PGNN stages)

The module between two BP runs is a graph U-Net on the Tanner graph that keeps
the two node types (bits, checks) apart at every level:

1. fine encoder: ``pre_layers`` bipartite message-passing layers on the full
   Tanner graph (bit -> check -> bit), every bit and every check;
2. pool (``pool``):
   * "diff"  -- bipartite DiffPool (BiDiffPool): separate soft assignments
     S_v [n, K_v] for bits and S_c [m, K_c] for checks. Pooled features
     X_v = S_v^T h_v, X_c = S_c^T h_c (membership-weighted means), pooled
     bipartite adjacency A' = S_c^T H S_v [K_c, K_v] plus the same product
     restricted to the *unsatisfied* checks, A'_u = S_c^T diag(u) H S_v, used
     as an edge feature of the coarse graph (fraction of violated edges between
     two clusters). ``assign="proto"`` (5G NR): S is the protograph lifting
     itself, bit cluster = base-graph column, check cluster = base-graph row, so
     the coarse graph is the base graph, exact and independent of Z;
     ``assign="learned"``: S = softmax(Linear(h)) per frame (DiffPool);
   * "topk"  -- dual Top-K / SAGPool (ablation): keep the K_c highest-scoring
     checks and the K_v highest-scoring bits (separate ratios), gate them by
     their score, coarse adjacency = H restricted to the kept nodes;
   * "none"  -- no hierarchy (ablation): only the global readout below;
3. concatenated readout: g = MLP([mean_v h_v || max_v h_v || mean_c h_c ||
   max_c h_c]), one context vector per frame, fed to every coarse node and
   added back to every fine node;
4. ``coarse_layers`` dense bipartite layers on the small graph (K_c x K_v);
5. unpool: h_v += Linear(S_v X_v' + g), h_c += Linear(S_c X_c' + g) (U-Net skip;
   for Top-K only kept nodes receive X', every node receives g), so *every*
   bit gets a correction although the coarse graph is small;
6. fine decoder: ``post_layers`` bipartite layers on the full graph, then a
   zero-initialised head gives one correction per bit for the next BP prior
   (base="prior": prior + delta as BP-SGT's tuned config; base="llr": llr +
   delta as BP-GNN). An untrained model is exactly BP with restarts.

Training tricks of the enhanced classic GNN (GNN+, Luo, Shi & Wu, ICML 2025),
where they apply to a node-level decoder:
* edge features enter every message (BP messages tanh(v2c/2), tanh(c2v/2) +
  base-graph edge embedding on the fine graph; violated-edge fraction on the
  coarse graph);
* normalisation: LayerNorm (pre-norm). Not BatchNorm: at inference only frames
  BP failed reach the module and early stopping shrinks the batch to as little
  as one frame, so batch statistics would differ from training;
* dropout <= 0.2 on the residual branches (default 0.1), off at inference;
* residual connections on every layer;
* a feed-forward network (FFN) after every message-passing layer;
* positional encoding: random-walk structural encoding (RWSE) of every bit and
  check, the return probabilities of a 2, 4, ..., 2*rw_steps step random walk on
  the Tanner graph. Return probabilities at step 4, 6, 8 grow with the number of
  4-, 6-, 8-cycles through the node, i.e. short cycles, which trapping sets
  are made of.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.bp import bp_iterations, staged_decode
from common.gnn_decoder import EdgeMLP, llr_features, mlp, soft_syndrome
from common.graph import scatter_sum

NUM_VN_FEATURES = 7  # llr, prior, L (2 each) + fraction of unsatisfied checks
NUM_CN_FEATURES = 2  # soft syndrome, 1 - 2 * hard syndrome
NUM_COARSE_FEATURES = 1  # violated fraction of the cluster


def rw_encoding(g, steps):
    """[n, steps], [m, steps]: return probability of a 2t-step random walk on the
    Tanner graph (t = 1..steps) for every bit and check. Cached on the graph."""
    key = f"_pgnn_rw{steps}"
    cached = getattr(g, key, None)
    if cached is not None and cached[0].device == g.H.device:
        return cached
    H = g.H.double()
    Pv = (H.t() / g.vn_deg.double()[:, None]) @ (H / g.cn_deg.double()[:, None])  # bit -> check -> bit
    Pc = (H / g.cn_deg.double()[:, None]) @ (H.t() / g.vn_deg.double()[:, None])  # check -> bit -> check
    out = []
    for P in (Pv, Pc):
        cur, diags = P, []
        for _ in range(steps):
            diags.append(torch.diagonal(cur))
            cur = cur @ P
        out.append(torch.stack(diags, 1).float())
    setattr(g, key, tuple(out))
    return tuple(out)


def _proto_cache(g):
    """Protograph pooling for 5G NR: pair index of every edge, cluster sizes and
    the fixed coarse adjacency (edge count between base-graph row / column)."""
    c = getattr(g, "_pgnn_proto", None)
    if c is not None and c["pair"].device == g.H.device:
        return c
    kv, kc = int(g.vn_type.max()) + 1, int(g.cn_type.max()) + 1
    pair = g.cn_type[g.e_c] * kv + g.vn_type[g.e_v]                      # [E]
    A = torch.zeros(kc * kv, device=g.H.device).index_add_(0, pair, torch.ones_like(pair, dtype=torch.float))
    c = dict(kv=kv, kc=kc, pair=pair, A=A.view(kc, kv),
             cnt_v=torch.zeros(kv, device=g.H.device).index_add_(0, g.vn_type, torch.ones(g.n, device=g.H.device)),
             cnt_c=torch.zeros(kc, device=g.H.device).index_add_(0, g.cn_type, torch.ones(g.m, device=g.H.device)))
    g._pgnn_proto = c
    return c


class FFN(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(self.ln(x))


class FineLayer(nn.Module):
    """One bipartite message-passing layer on the full Tanner graph (bit -> check
    -> bit) with pre-LayerNorm, edge features, dropout, residuals and an FFN."""

    def __init__(self, d, d_e, hidden, dropout, last=False):
        super().__init__()
        self.ln_v, self.ln_c, self.ln_c2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.msg_vc = EdgeMLP(d, d_e, hidden, d)
        self.msg_cv = EdgeMLP(d, d_e, hidden, d)
        self.upd_c = mlp(2 * d + NUM_CN_FEATURES, hidden, d)
        self.upd_v = mlp(2 * d + NUM_VN_FEATURES, hidden, d)
        self.ffn_v = FFN(d, 2 * d, dropout)
        # the check states of the last layer are not read any more: no check FFN there
        self.ffn_c = None if last else FFN(d, 2 * d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_v, h_c, f_v, f_c, e, g, inv_dc, inv_dv):
        hv, hc = self.ln_v(h_v), self.ln_c(h_c)
        m = self.msg_vc(hv, g.e_v, hc, g.e_c, e)
        h_c = h_c + self.drop(self.upd_c(torch.cat([hc, scatter_sum(m, g.e_c, g.m) * inv_dc, f_c], -1)))
        hc = self.ln_c2(h_c)
        m = self.msg_cv(hc, g.e_c, hv, g.e_v, e)
        h_v = h_v + self.drop(self.upd_v(torch.cat([hv, scatter_sum(m, g.e_v, g.n) * inv_dv, f_v], -1)))
        h_v = h_v + self.drop(self.ffn_v(h_v))
        if self.ffn_c is not None:
            h_c = h_c + self.drop(self.ffn_c(h_c))
        return h_v, h_c

    @staticmethod
    def macs(d, d_e, h, g):
        n, m, E = g.n, g.m, g.num_edges
        msg = (n + m) * 2 * d * h + 2 * E * (d_e * h + h * d)
        upd = m * ((2 * d + NUM_CN_FEATURES) * h + h * d) + n * ((2 * d + NUM_VN_FEATURES) * h + h * d)
        ffn = (n + m) * 2 * d * 2 * d
        return msg + upd + ffn


class CoarseLayer(nn.Module):
    """Dense bipartite layer on the pooled graph. Check cluster i aggregates bit
    clusters j with weights A[i, j] (row-normalised) and, separately, weighted by
    the violated-edge fraction r[i, j] (edge feature of the coarse graph)."""

    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.ln_v, self.ln_c, self.ln_c2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.upd_c = mlp(4 * d + NUM_COARSE_FEATURES, hidden, d)
        self.upd_v = mlp(4 * d + NUM_COARSE_FEATURES, hidden, d)
        self.ffn_v, self.ffn_c = FFN(d, 2 * d, dropout), FFN(d, 2 * d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_v, x_c, A_cv, Ar_cv, A_vc, Ar_vc, u_v, u_c, glob):
        xv, xc = self.ln_v(x_v), self.ln_c(x_c)
        gc, gv = glob[:, None].expand_as(x_c), glob[:, None].expand_as(x_v)
        x_c = x_c + self.drop(self.upd_c(torch.cat([xc, A_cv @ xv, Ar_cv @ xv, gc, u_c[..., None]], -1)))
        xc = self.ln_c2(x_c)
        x_v = x_v + self.drop(self.upd_v(torch.cat([xv, A_vc @ xc, Ar_vc @ xc, gv, u_v[..., None]], -1)))
        x_v = x_v + self.drop(self.ffn_v(x_v))
        x_c = x_c + self.drop(self.ffn_c(x_c))
        return x_v, x_c

    @staticmethod
    def macs(d, h, kv, kc):
        upd = (kv + kc) * ((4 * d + NUM_COARSE_FEATURES) * h + h * d)
        agg = 4 * kc * kv * d
        return upd + agg + (kv + kc) * 2 * d * 2 * d


def _rownorm(A, eps=1e-6):
    return A / A.sum(-1, keepdim=True).clamp_min(eps)


class PooledGNNBPDecoder(nn.Module):
    def __init__(self, num_vn_types, num_cn_types, num_edge_types, d=16, d_e=8, hidden=48,
                 pre_layers=1, post_layers=1, coarse_layers=2, pool="diff", assign="learned",
                 k_v=16, k_c=8, topk_v=0.125, topk_c=0.25, readout=True, rw_steps=8, dropout=0.1,
                 stages=4, bp_iters=10, base="prior"):
        super().__init__()
        assert pool in ("diff", "topk", "none") and assign in ("learned", "proto")
        assert base in ("prior", "llr") and dropout <= 0.2
        self.stages, self.bp_iters = stages, bp_iters
        self.iters = (stages + 1) * bp_iters
        self.pool, self.assign, self.base, self.readout_on = pool, assign, base, readout
        self.rw_steps = rw_steps
        self.topk_v, self.topk_c = topk_v, topk_c
        if pool == "diff" and assign == "proto":
            k_v, k_c = num_vn_types, num_cn_types  # base-graph columns / rows
        self.k_v, self.k_c = k_v, k_c
        self.vn_emb = nn.Embedding(num_vn_types, d)
        self.cn_emb = nn.Embedding(num_cn_types, d)
        self.edge_emb = nn.Embedding(num_edge_types, d_e)
        self.vn_in = nn.Linear(NUM_VN_FEATURES + rw_steps, d)
        self.cn_in = nn.Linear(NUM_CN_FEATURES + rw_steps, d)
        self.edge_in = nn.Linear(2, d_e)
        self.pre = nn.ModuleList(FineLayer(d, d_e, hidden, dropout) for _ in range(pre_layers))
        self.post = nn.ModuleList(FineLayer(d, d_e, hidden, dropout, last=i == post_layers - 1)
                                  for i in range(post_layers))
        if readout:
            self.glob = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden), nn.SiLU(),
                                      nn.Linear(hidden, d))
        if pool != "none":
            if pool == "topk" or assign == "learned":
                self.ln_pool_v, self.ln_pool_c = nn.LayerNorm(d), nn.LayerNorm(d)
            if pool == "diff" and assign == "learned":
                self.s_v, self.s_c = nn.Linear(d, k_v), nn.Linear(d, k_c)
            if pool == "topk":
                self.score_v, self.score_c = nn.Linear(d, 1), nn.Linear(d + 1, 1)
            self.coarse = nn.ModuleList(CoarseLayer(d, hidden, dropout) for _ in range(coarse_layers))
        if pool != "none" or readout:
            self.up_v, self.up_c = nn.Linear(d, d), nn.Linear(d, d)
        self.ln_out = nn.LayerNorm(d)
        self.head = mlp(d, hidden, 1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ----------------------------------------------------------------- pooling
    def _pool_diff(self, h_v, h_c, unsat, g):
        """BiDiffPool. Returns coarse features, normalised coarse adjacencies,
        violated fractions and an unpool function."""
        B = h_v.shape[0]
        if self.assign == "proto":
            pc = _proto_cache(g)
            kv, kc = pc["kv"], pc["kc"]
            x_v = h_v.new_zeros(B, kv, h_v.shape[-1]).index_add_(1, g.vn_type, h_v) / pc["cnt_v"].clamp_min(1)[:, None]
            x_c = h_c.new_zeros(B, kc, h_c.shape[-1]).index_add_(1, g.cn_type, h_c) / pc["cnt_c"].clamp_min(1)[:, None]
            A = pc["A"].expand(B, kc, kv)
            Au = unsat.new_zeros(B, kc * kv).index_add_(1, pc["pair"], unsat[:, g.e_c]).view(B, kc, kv)
            u_c = unsat.new_zeros(B, kc).index_add_(1, g.cn_type, unsat) / pc["cnt_c"].clamp_min(1)
            unpool_v = lambda x: x[:, g.vn_type]
            unpool_c = lambda x: x[:, g.cn_type]
        else:
            S_v = torch.softmax(self.s_v(self.ln_pool_v(h_v)), -1)          # [B, n, Kv]
            S_c = torch.softmax(self.s_c(self.ln_pool_c(h_c)), -1)          # [B, m, Kc]
            cnt_v, cnt_c = S_v.sum(1).clamp_min(1e-6), S_c.sum(1).clamp_min(1e-6)
            x_v = S_v.transpose(1, 2) @ h_v / cnt_v[..., None]
            x_c = S_c.transpose(1, 2) @ h_c / cnt_c[..., None]
            AS = scatter_sum(S_v[:, g.e_v], g.e_c, g.m)                     # H S_v      [B, m, Kv]
            A = S_c.transpose(1, 2) @ AS                                    # S_c^T H S_v
            Au = S_c.transpose(1, 2) @ (AS * unsat[..., None])
            u_c = (S_c * unsat[..., None]).sum(1) / cnt_c
            unpool_v = lambda x: S_v @ x
            unpool_c = lambda x: S_c @ x
        r = Au / A.clamp_min(1e-6)                                          # violated-edge fraction
        u_v = (r * A).sum(1) / A.sum(1).clamp_min(1e-6)                     # per bit cluster
        At = A.transpose(1, 2)
        return x_v, x_c, _rownorm(A), _rownorm(A * r), _rownorm(At), _rownorm(At * r.transpose(1, 2)), \
            u_v, u_c, unpool_v, unpool_c

    def num_topk(self, g):
        return max(4, math.ceil(self.topk_v * g.n)), max(2, math.ceil(self.topk_c * g.m))

    def _pool_topk(self, h_v, h_c, unsat, g):
        """Dual Top-K / SAGPool with separate ratios for bits and checks. Check
        scores also see the hard syndrome, so violated checks are preferred."""
        kv, kc = self.num_topk(g)
        z_v = torch.sigmoid(self.score_v(self.ln_pool_v(h_v))).squeeze(-1)                        # [B, n]
        z_c = torch.sigmoid(self.score_c(torch.cat([self.ln_pool_c(h_c), unsat[..., None]], -1))).squeeze(-1)
        iv = torch.topk(z_v, kv, 1).indices
        ic = torch.topk(z_c, kc, 1).indices
        x_v = h_v.gather(1, iv[..., None].expand(-1, -1, h_v.shape[-1])) * z_v.gather(1, iv)[..., None]
        x_c = h_c.gather(1, ic[..., None].expand(-1, -1, h_c.shape[-1])) * z_c.gather(1, ic)[..., None]
        Hc = g.H[ic]                                                        # [B, Kc, n]
        A = Hc.gather(2, iv[:, None, :].expand(-1, kc, -1))                 # [B, Kc, Kv]
        uc = unsat.gather(1, ic)
        r = A * uc[..., None]
        u_v = r.sum(1) / A.sum(1).clamp_min(1e-6)
        At = A.transpose(1, 2)

        def scatter_back(idx, n):
            return lambda x: x.new_zeros(x.shape[0], n, x.shape[-1]).scatter(
                1, idx[..., None].expand(-1, -1, x.shape[-1]), x)
        return x_v, x_c, _rownorm(A), _rownorm(r), _rownorm(At), _rownorm(r.transpose(1, 2)), \
            u_v, uc, scatter_back(iv, g.n), scatter_back(ic, g.m)

    # ------------------------------------------------------------- refinement
    def refine(self, llr, prior, L, v2c, c2v, g):
        """Pooled GNN between two BP runs: stuck BP state -> prior of the next run."""
        hard = (L < 0).to(L.dtype)
        unsat = g.syndrome(hard)                                            # [B, m]
        frac_unsat = scatter_sum(unsat[:, g.e_c], g.e_v, g.n) / g.vn_deg
        f_v = torch.cat([llr_features(llr), llr_features(prior), llr_features(L), frac_unsat[..., None]], -1)
        f_c = torch.stack([soft_syndrome(L, g), 1.0 - 2.0 * unsat], -1)
        rw_v, rw_c = rw_encoding(g, self.rw_steps) if self.rw_steps else (None, None)
        B = L.shape[0]
        xin_v = torch.cat([f_v, rw_v.expand(B, -1, -1)], -1) if self.rw_steps else f_v
        xin_c = torch.cat([f_c, rw_c.expand(B, -1, -1)], -1) if self.rw_steps else f_c
        h_v = self.vn_in(xin_v) + self.vn_emb(g.vn_type)
        h_c = self.cn_in(xin_c) + self.cn_emb(g.cn_type)
        e = self.edge_in(torch.stack([torch.tanh((v2c / 2).clamp(-15, 15)),
                                      torch.tanh((c2v / 2).clamp(-15, 15))], -1)) + self.edge_emb(g.edge_type)
        inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
        for layer in self.pre:
            h_v, h_c = layer(h_v, h_c, f_v, f_c, e, g, inv_dc, inv_dv)

        glob = None
        if self.readout_on:  # concatenated readout, bits and checks kept apart
            glob = self.glob(torch.cat([h_v.mean(1), h_v.amax(1), h_c.mean(1), h_c.amax(1)], -1))
        if self.pool != "none":
            pooled = (self._pool_diff if self.pool == "diff" else self._pool_topk)(h_v, h_c, unsat, g)
            x_v, x_c, A_cv, Ar_cv, A_vc, Ar_vc, u_v, u_c, unpool_v, unpool_c = pooled
            gl = glob if glob is not None else x_v.new_zeros(B, x_v.shape[-1])
            for layer in self.coarse:
                x_v, x_c = layer(x_v, x_c, A_cv, Ar_cv, A_vc, Ar_vc, u_v, u_c, gl)
            up_v, up_c = unpool_v(x_v), unpool_c(x_c)
            if glob is not None:
                up_v, up_c = up_v + glob[:, None], up_c + glob[:, None]
            h_v, h_c = h_v + self.up_v(up_v), h_c + self.up_c(up_c)
        elif glob is not None:
            h_v, h_c = h_v + self.up_v(glob)[:, None], h_c + self.up_c(glob)[:, None]

        for layer in self.post:
            h_v, h_c = layer(h_v, h_c, f_v, f_c, e, g, inv_dc, inv_dv)
        delta = self.head(self.ln_out(h_v)).squeeze(-1)
        base = prior if self.base == "prior" else llr
        return (base + delta).clamp(-30, 30)

    def forward(self, llr, g):
        """Differentiable end-to-end pass: marginals of every BP run after a PGNN stage."""
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
        """Multiply-accumulates of one PGNN stage per codeword."""
        d = self.vn_in.out_features
        d_e = self.edge_emb.embedding_dim
        h = self.head[0].out_features
        n, m, E = g.n, g.m, g.num_edges
        total = (len(self.pre) + len(self.post)) * FineLayer.macs(d, d_e, h, g)
        total += n * (NUM_VN_FEATURES + self.rw_steps) * d + m * (NUM_CN_FEATURES + self.rw_steps) * d + E * 2 * d_e
        total += n * (d * h + h)                                            # head
        if self.readout_on:
            total += 4 * d * h + h * d
        if self.pool != "none" or self.readout_on:
            total += (n + m) * d * d                                        # up_v / up_c
        if self.pool == "diff" and self.assign == "learned":
            kv, kc = self.k_v, self.k_c
            total += n * d * kv + m * d * kc                                # assignment logits
            total += n * kv * d + m * kc * d                                # S^T h
            total += E * kv + 2 * m * kc * kv                               # S_c^T H S_v (+ unsat)
            total += n * kv * d + m * kc * d                                # unpool S X
        elif self.pool == "diff":
            kv, kc = self.k_v, self.k_c
            total += (n + m) * d + E                                        # index pooling, unsat counts
        elif self.pool == "topk":
            kv, kc = self.num_topk(g)
            total += n * d + m * (d + 1) + (kv + kc) * d
        if self.pool != "none":
            total += len(self.coarse) * CoarseLayer.macs(d, h, kv, kc)
        return int(total)
