"""Instrumented copy of SparseGTBPDecoder.refine (SparseGT_BP/sgt/sgt_decoder.py),
same operations in the same order, split into regions:

  features     syndrome, unsatisfied-check count per bit, disagreeing / supporting
               check messages (gathers + 4 scatter_sum over all E edges)
  select       topk of the k = k_frac * n least reliable bits
  tokens       gather the 9 per-token features, input Linear + VN-type embedding
  bias/Hk      gather the dense rows of H^T for the selected bits  [B, k, m]
  bias/shared  bmm  Hk @ Hk^T                     (checks shared by two bits)   B*k*k*m MACs
  bias/shared_unsat  (Hk * unsat) @ Hk^T          (shared unsatisfied checks)   B*k*k*m MACs
  bias/embed   pair index, Embedding lookup [B,k,k,heads], permute
  block<l>/attn  LayerNorm, qkv Linear, scaled_dot_product_attention with the
                 float bias as mask, output projection
  block<l>/ffn   LayerNorm + 2-layer GELU MLP
  head         final LayerNorm + Linear, scatter of the corrections into the prior

``variant`` selects equivalent but cheaper formulations of the bias, to test
whether the bias is the bottleneck (tests/test_equivalence.py checks they give
the same output):
  "orig"      as trained
  "gather"    shared = (H H^T)[idx, idx] gathered from a precomputed n x n
              matrix (no k*k*m bmm), shared_unsat still a bmm
  "gather16"  as "gather", and shared_unsat as an fp16 bmm (tensor cores on a T4;
              exact because the operands are 0/1 and the counts are < 2048)
  "nobias"    struct_bias off (changes the model output: timing only)
"""

import torch
import torch.nn.functional as F

from common.graph import scatter_sum
from sgt.sgt_decoder import LLR_CLIP, LLR_SCALE, MAX_PAIR, _graph_cache, _sign

from .regions import OFF

VARIANTS = ("orig", "gather", "gather16", "nobias")


def _hht(g):
    if getattr(g, "_lat_HHt", None) is None or g._lat_HHt.device != g.H.device:
        g._lat_HHt = (g.H.t() @ g.H).contiguous()  # [n, n] shared-check counts
    return g._lat_HHt


def block_forward(blk, x, bias, R):
    b, k, d = x.shape
    with R("attn"):
        q, kk, v = blk.qkv(blk.ln1(x)).view(b, k, 3, blk.heads, d // blk.heads).unbind(2)
        q, kk, v = (t.transpose(1, 2) for t in (q, kk, v))
        a = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias)
        x = x + blk.proj(a.transpose(1, 2).reshape(b, k, d))
    with R("ffn"):
        return x + blk.ff(blk.ln2(x))


def refine(model, llr, prior, L, v2c, c2v, g, R=OFF, variant="orig"):
    self = model
    B, n = L.shape
    k = self.num_selected(n)
    with R("features"):
        s_all = _sign(L)
        hard = (L < 0).to(L.dtype)
        unsat = g.syndrome(hard)
        deg = g.vn_deg
        n_unsat = scatter_sum(unsat[:, g.e_c], g.e_v, n)
        s_e = s_all[:, g.e_v]
        disagree = scatter_sum((c2v * s_e < 0).to(L.dtype), g.e_v, n) / deg
        support = scatter_sum(torch.tanh((c2v / 2).clamp(-15, 15)) * s_e, g.e_v, n) / deg
    with R("select"):
        score = L.abs() if self.select == "llr" else L.abs() / (1.0 + n_unsat)
        idx = torch.topk(score, k, dim=1, largest=False, sorted=True).indices
    with R("tokens"):
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
    struct = self.struct_bias and variant != "nobias"
    if struct or self.attn == "local":
        with R("bias"):
            if variant in ("gather", "gather16"):
                with R("shared"):
                    shared = _hht(g)[idx[:, :, None], idx[:, None, :]]
                with R("Hk"):
                    Ht = _graph_cache(g)
                    Hk = Ht[idx] if variant == "gather" else Ht.half()[idx]
            else:
                with R("Hk"):
                    Hk = _graph_cache(g)[idx]
                with R("shared"):
                    shared = torch.bmm(Hk, Hk.transpose(1, 2))
            with R("embed"):
                bias = x.new_zeros(B, k, k, self.heads)
            if struct:
                with R("shared_unsat"):
                    u = unsat if variant != "gather16" else unsat.half()
                    sh_u = torch.bmm(Hk * u[:, None, :], Hk.transpose(1, 2))
                    if variant == "gather16":
                        sh_u = sh_u.float()
                with R("embed"):
                    pair = shared.clamp(max=MAX_PAIR) * (MAX_PAIR + 1) + sh_u.clamp(max=MAX_PAIR)
                    bias = self.pair_bias(pair.long())
            with R("embed"):
                if self.attn == "local":
                    bias = bias.masked_fill((shared == 0)[..., None], float("-inf"))
                bias = bias.permute(0, 3, 1, 2)
    for i, blk in enumerate(self.blocks):
        with R(f"block{i}"):
            x = block_forward(blk, x, bias, R)
    with R("head"):
        delta = self.head(self.ln(x)).squeeze(-1) * s
        base = prior if self.base == "prior" else llr
        new = delta if self.fix == "replace" else base.gather(1, idx) + delta
        return base.scatter(1, idx, new).clamp(-30, 30)


def make_refine(model, variant="orig", budget=1.5e8):
    """refine_chunked equivalent: fn(llr, prior, L, v2c, c2v, g, R)."""
    def fn(llr, prior, L, v2c, c2v, g, R=OFF):
        k = model.num_selected(L.shape[1])
        step = max(1, int(budget // (k * k * model.heads)))
        if L.shape[0] <= step:
            return refine(model, llr, prior, L, v2c, c2v, g, R, variant)
        with R("chunks"):
            return torch.cat([refine(model, *(a[i:i + step] for a in (llr, prior, L, v2c, c2v)), g, R, variant)
                              for i in range(0, L.shape[0], step)])
    return fn
