"""BP-SGT hybrid: a sparse graph transformer between BP runs instead of the GNN.

Same skeleton as the NVIDIA BP-GNN hybrid (Gong, Cammerer & Renes 2023, see
Nvidia_GNN+BP(lLatest)/common/bp_gnn_decoder.py):

    llr -> BP(T) -> SGT -> BP(T) -> SGT -> ... -> BP(T)      (S SGT stages)

but the GNN that reads the whole Tanner graph is replaced by a sparse graph
transformer (SGT) that only looks at the k bits the stuck BP run is least sure
about, and fixes those:

1. after a BP run, rank the bits by |L| (select="unsat": |L| / (1 + number of
   unsatisfied checks the bit touches)) and keep the k = k_frac * n least
   confident ones;
2. one token per selected bit: the stuck BP state expressed relative to the
   bit's current hard decision s = sign(L) (|L|, s*llr, s*prior, unsatisfied
   checks, disagreeing check messages, ...) plus a VN-type embedding (degree,
   or 5G base-graph column, so one model decodes several lifting sizes);
3. multi-head attention among the k tokens only, O(k^2) and independent of the
   rest of the graph, with a learned per-head bias from the Tanner graph: the
   number of checks two selected bits share and how many of those are
   unsatisfied. attn="local" masks attention to bits that share a check
   (2-hop Tanner neighbourhood), attn="global" lets all k bits attend to each other;
4. one correction per selected bit, applied to the prior of the next BP run
   (fix="residual": base + s*delta, fix="replace": s*delta, i.e. the bit's
   prior is overwritten). base="prior" keeps the corrections of earlier stages
   on bits that are no longer selected, base="llr" restarts every unselected
   bit from its channel LLR (as the BP-GNN does). The head is zero-initialised,
   so an untrained residual model is exactly BP with restarts.
5. warm=True: the next BP run continues from the check messages of the previous
   one instead of restarting (only the channel prior of the k fixed bits
   changes), so an untrained model is exactly BP(S*T + T) and the SGT only
   adds corrections on top. warm=False restarts BP as the BP-GNN does.

Everything is differentiable (gather / scatter on the selected indices; the
indices themselves are not), so the model is trained end to end through all
BP runs with the BP-GNN loss and hard-example mining (common/train.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.bp import _cn_spa, bp_iterations, bp_stage, staged_decode
from common.graph import scatter_sum

LLR_SCALE, LLR_CLIP = 8.0, 5.0
MAX_PAIR = 3  # shared-check counts are clipped here for the bias tables
NUM_FEATURES = 9


def _sign(x):
    """+1 / -1 with 0 mapped to +1 (bit 0)."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def _graph_cache(g):
    """Dense H^T [n, m], cached on the TannerGraph (pairwise shared-check counts)."""
    if getattr(g, "_sgt_Ht", None) is None or g._sgt_Ht.device != g.H.device:
        g._sgt_Ht = g.H.t().contiguous()
    return g._sgt_Ht


def bp_resume(prior, g, iters, c2v):
    """Differentiable sum-product run that continues from check messages ``c2v``
    with a new channel prior. Returns (marginal LLR, v2c, c2v)."""
    v2c = (prior + scatter_sum(c2v, g.e_v, g.n))[:, g.e_v] - c2v
    for _ in range(iters):
        c2v = _cn_spa(v2c, g, eps=1e-6)
        total = prior + scatter_sum(c2v, g.e_v, g.n)
        v2c = total[:, g.e_v] - c2v
    return total, v2c, c2v


@torch.no_grad()
def bp_stage_resume(prior, g, iters, c2v, early_stop=True):
    """Inference counterpart of bp_resume with per-iteration early stopping and
    batch compaction (same outputs as common.bp.bp_stage)."""
    B = prior.shape[0]
    TOT, C2V = prior.clone(), c2v.clone()
    V2C = torch.zeros_like(c2v)
    ok = torch.zeros(B, dtype=torch.bool, device=prior.device)
    used = torch.full((B,), iters, device=prior.device, dtype=torch.long)
    active = torch.arange(B, device=prior.device)
    x = prior
    v2c = (x + scatter_sum(c2v, g.e_v, g.n))[:, g.e_v] - c2v
    for t in range(iters):
        c2v = _cn_spa(v2c, g)
        total = x + scatter_sum(c2v, g.e_v, g.n)
        v2c = total[:, g.e_v] - c2v
        TOT[active], V2C[active], C2V[active] = total, v2c, c2v
        if early_stop:
            okk = (g.syndrome((total < 0).float()) == 0).all(1)
            if okk.any():
                used[active[okk]] = t + 1
                ok[active[okk]] = True
                keep = ~okk
                if not keep.any() or t == iters - 1:
                    break
                active, x, v2c = active[keep], x[keep], v2c[keep]
    if not early_stop:
        ok = (g.syndrome((TOT < 0).float()) == 0).all(1)
    return TOT, V2C, C2V, ok, used


@torch.no_grad()
def staged_decode_resume(llr, g, stages, iters, refine, early_stop=True):
    """common.bp.staged_decode with BP continuing from the previous check messages."""
    B = llr.shape[0]
    out = llr.clone()
    used = torch.zeros(B, device=llr.device, dtype=torch.long)
    active = torch.arange(B, device=llr.device)
    x, prior, c2v = llr, llr, None
    for s in range(stages + 1):
        if c2v is None:
            L, v2c, c2v, ok, u = bp_stage(prior, g, iters, early_stop)
        else:
            L, v2c, c2v, ok, u = bp_stage_resume(prior, g, iters, c2v, early_stop)
        out[active] = L
        used[active] += u
        if s == stages:
            break
        if early_stop:
            keep = ~ok
            if not keep.any():
                break
            active, x, prior, L, v2c, c2v = (a[keep] for a in (active, x, prior, L, v2c, c2v))
        prior = refine(x, prior, L, v2c, c2v, g)
    return out, used


class Block(nn.Module):
    """Pre-LN transformer block; attention bias / mask given per batch and head."""

    def __init__(self, d, heads, hidden):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x, bias):
        b, k, d = x.shape
        q, kk, v = self.qkv(self.ln1(x)).view(b, k, 3, self.heads, d // self.heads).unbind(2)
        q, kk, v = (t.transpose(1, 2) for t in (q, kk, v))
        a = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias)
        x = x + self.proj(a.transpose(1, 2).reshape(b, k, d))
        return x + self.ff(self.ln2(x))


class SparseGTBPDecoder(nn.Module):
    def __init__(self, num_vn_types, num_cn_types=1, num_edge_types=1, d=64, heads=4, hidden=128,
                 gt_layers=2, k_frac=0.1, k_min=8, stages=4, bp_iters=10, attn="global",
                 select="llr", fix="residual", base="prior", struct_bias=True, warm=False):
        super().__init__()
        assert attn in ("global", "local") and select in ("llr", "unsat")
        assert fix in ("residual", "replace") and base in ("prior", "llr")
        self.stages, self.bp_iters = stages, bp_iters
        self.iters = (stages + 1) * bp_iters
        self.k_frac, self.k_min = k_frac, k_min
        self.attn, self.select, self.fix, self.base = attn, select, fix, base
        self.struct_bias, self.warm = struct_bias, warm
        self.heads = heads
        self.vn_emb = nn.Embedding(num_vn_types, d)
        self.inp = nn.Linear(NUM_FEATURES, d)
        self.blocks = nn.ModuleList(Block(d, heads, hidden) for _ in range(gt_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if struct_bias:  # one bias per head for every (shared, shared-unsatisfied) count pair
            self.pair_bias = nn.Embedding((MAX_PAIR + 1) ** 2, heads)
            nn.init.zeros_(self.pair_bias.weight)

    def num_selected(self, n):
        return min(n, max(self.k_min, int(math.ceil(self.k_frac * n))))

    def refine(self, llr, prior, L, v2c, c2v, g):
        """SGT between two BP runs: stuck BP state -> prior of the next run."""
        B, n = L.shape
        k = self.num_selected(n)
        s_all = _sign(L)
        hard = (L < 0).to(L.dtype)
        unsat = g.syndrome(hard)                                     # [B, m]
        deg = g.vn_deg
        n_unsat = scatter_sum(unsat[:, g.e_c], g.e_v, n)             # [B, n]
        s_e = s_all[:, g.e_v]
        disagree = scatter_sum((c2v * s_e < 0).to(L.dtype), g.e_v, n) / deg
        support = scatter_sum(torch.tanh((c2v / 2).clamp(-15, 15)) * s_e, g.e_v, n) / deg
        with torch.no_grad():
            score = L.abs() if self.select == "llr" else L.abs() / (1.0 + n_unsat)
            idx = torch.topk(score, k, dim=1, largest=False, sorted=True).indices  # [B, k]
        s = s_all.gather(1, idx)
        rel = lambda x: (x.gather(1, idx) * s / LLR_SCALE).clamp(-LLR_CLIP, LLR_CLIP)
        dk = deg[idx]
        nu = n_unsat.gather(1, idx)
        rank = torch.arange(k, device=L.device, dtype=L.dtype).expand(B, k) / k
        syn_w = unsat.mean(1, keepdim=True).expand(B, k)
        f = torch.stack([rel(L), rel(llr), rel(prior), nu / dk, nu / 4.0, disagree.gather(1, idx),
                         support.gather(1, idx), syn_w, rank], -1)
        x = self.inp(f) + self.vn_emb(g.vn_type[idx])

        bias = None
        if self.struct_bias or self.attn == "local":
            Hk = _graph_cache(g)[idx]                                # [B, k, m]
            shared = torch.bmm(Hk, Hk.transpose(1, 2))               # checks shared by two selected bits
            bias = x.new_zeros(B, k, k, self.heads)
            if self.struct_bias:
                sh_u = torch.bmm(Hk * unsat[:, None, :], Hk.transpose(1, 2))
                pair = shared.clamp(max=MAX_PAIR) * (MAX_PAIR + 1) + sh_u.clamp(max=MAX_PAIR)
                bias = self.pair_bias(pair.long())
            if self.attn == "local":  # diagonal always allowed (deg >= 1), no empty rows
                bias = bias.masked_fill((shared == 0)[..., None], float("-inf"))
            bias = bias.permute(0, 3, 1, 2)                          # [B, h, k, k]
        for blk in self.blocks:
            x = blk(x, bias)
        delta = self.head(self.ln(x)).squeeze(-1) * s                # [B, k], sign-relative
        base = prior if self.base == "prior" else llr
        new = delta if self.fix == "replace" else base.gather(1, idx) + delta
        return base.scatter(1, idx, new).clamp(-30, 30)

    def forward(self, llr, g):
        """Differentiable end-to-end pass: marginals of every BP run after an SGT stage."""
        prior, outs, c2v = llr, [], None
        for s in range(self.stages + 1):
            if self.warm and c2v is not None:
                L, v2c, c2v = bp_resume(prior, g, self.bp_iters, c2v)
            else:
                L, v2c, c2v = bp_iterations(prior, g, self.bp_iters)
            if s > 0:
                outs.append(L)
            if s < self.stages:
                prior = self.refine(llr, prior, L, v2c, c2v, g)
        return outs

    def refine_chunked(self, llr, prior, L, v2c, c2v, g, budget=1.5e8):
        """refine() over batch chunks of at most ``budget`` attention-bias elements
        (B * k^2 * heads), so large k on long codes fits in GPU memory at inference."""
        k = self.num_selected(L.shape[1])
        step = max(1, int(budget // (k * k * self.heads)))
        if L.shape[0] <= step:
            return self.refine(llr, prior, L, v2c, c2v, g)
        return torch.cat([self.refine(*(a[i:i + step] for a in (llr, prior, L, v2c, c2v)), g)
                          for i in range(0, L.shape[0], step)])

    @torch.no_grad()
    def decode(self, llr, g, iters=None, early_stop=True):
        dec = staged_decode_resume if self.warm else staged_decode
        return dec(llr, g, self.stages, self.bp_iters, self.refine_chunked, early_stop)

    def macs_per_refinement(self, g):
        """Multiply-accumulates of one SGT stage per codeword (incl. the k x k x m
        structure products and the per-edge feature sums)."""
        k = self.num_selected(g.n)
        d = self.inp.out_features
        h = self.blocks[0].ff[0].out_features if len(self.blocks) else 0
        per_layer = k * (3 * d * d + d * d + 2 * d * h) + 2 * k * k * d
        struct = (2 if self.struct_bias else 1) * k * k * g.m if (self.struct_bias or self.attn == "local") else 0
        feats = 4 * g.num_edges + k * NUM_FEATURES * d + k * d
        return int(len(self.blocks) * per_layer + struct + feats)
