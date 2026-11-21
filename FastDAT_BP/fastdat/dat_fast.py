"""Inference-only fast path of the BP-DAT module (DualAttention_BP/dat/dat_decoder.py).

``fast_dat_refine(model, layout)`` returns a refine function with the signature
of DualAttnBPDecoder.refine that uses the *same weights* and the same
arithmetic per edge and per node, with three implementation changes:

1. every scatter-sum (syndrome, unsatisfied-check fraction, soft syndrome,
   cross-attention aggregation, NB sums) is a gather + sum over the padded
   neighbour tables of fastdat.layout (no atomics, fixed shapes);
2. the cross-attention softmax over a receiver's neighbours is one
   ``torch.softmax`` over the padded [B, nodes, width, heads] scores (pads at
   -inf), instead of scatter_reduce(amax) + exp + scatter_sum + gather;
3. optional mixed precision (``dtype=torch.float16`` / bfloat16): the module's
   matmuls run under autocast; the BP messages, the LLR inputs and the output
   prior stay fp32 (sum-product BP underflows in fp16).

With dtype=None the outputs equal model.refine up to fp32 summation order
(checked in tests/test_equivalence.py). Dropout is off (eval mode).
"""

import math

import torch
import torch.nn.functional as F

from common.gnn_decoder import llr_features
from pgnn.pgnn_decoder import rw_encoding


def _soft_syndrome(L, lay):
    """common.gnn_decoder.soft_syndrome with the layout's sums."""
    t = torch.tanh((L / 2.0).clamp(-15, 15))[:, lay.g.e_v]
    both = lay.sum_c(torch.stack([torch.log(t.abs().clamp_min(1e-12)), (t < 0).to(L.dtype)], -1))
    return (1.0 - 2.0 * torch.remainder(both[..., 1], 2.0)) * torch.exp(both[..., 0])


def _cross_attn(mod, h_recv, idx_recv, per_recv, h_send, idx_send, e):
    """SparseCrossAttention.forward with a padded softmax. per_recv: layout.per_c / per_v."""
    B, E = e.shape[0], idx_recv.shape[0]
    q = mod.q(h_recv)[:, idx_recv]
    k, v = mod.kv(h_send)[:, idx_send].chunk(2, -1)
    ew, ev = mod.e(e).chunk(2, -1)
    p = (q * k * (1.0 + ew)).view(B, E, mod.h, mod.dh).sum(-1) / math.sqrt(mod.dh)     # [B, E, H]
    a = torch.softmax(per_recv(p.float(), fill=float("-inf")), dim=2)                  # [B, Nr, W, H]
    val = per_recv((v + ev).view(B, E, mod.h, mod.dh))                                 # [B, Nr, W, H, dh]
    out = (val * a.to(val.dtype)[..., None]).sum(2)
    return mod.out(out.reshape(B, out.shape[1], -1))


def _layer(layer, h_v, h_c, s_vc, f_v, f_c, e, g, lay, inv_dc, inv_dv):
    """DATLayer.forward (eval) with the layout's operations."""
    hv, hc = layer.ln_v(h_v), layer.ln_c(h_c)
    if layer.cross == "attn":
        x_c = _cross_attn(layer.x_c, hc, g.e_c, lay.per_c, hv, g.e_v, e)
    else:
        S_c = lay.sum_c(s_vc)
        x_c = S_c * inv_dc
    parts = [hc, x_c, f_c.to(hc.dtype)] + ([layer.sa_c(hc)] if layer.self_attn else [])
    h_c = h_c + layer.upd_c(torch.cat(parts, -1))
    h_c = h_c + layer.ffn_c(h_c)
    hc = layer.ln_c2(h_c)
    if layer.cross == "attn":
        x_v = _cross_attn(layer.x_v, hv, g.e_v, lay.per_v, hc, g.e_c, e)
    else:
        s_cv = layer.nb_cv(S_c, hc, g.e_c, s_vc, e)
        S_v = lay.sum_v(s_cv)
        x_v = S_v * inv_dv
        s_vc = None if layer.nb_vc is None else layer.nb_vc(S_v, hv, g.e_v, s_cv, e)
    parts = [hv, x_v, f_v.to(hv.dtype)] + ([layer.sa_v(hv)] if layer.self_attn else [])
    h_v = h_v + layer.upd_v(torch.cat(parts, -1))
    h_v = h_v + layer.ffn_v(h_v)
    return h_v, h_c, s_vc


def fast_dat_refine(model, lay, dtype=None):
    """Refine function for StagedEngine (or staged_decode) equivalent to model.refine."""
    model.eval()
    g = lay.g
    dev = g.e_v.device
    inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
    rw = rw_encoding(g, model.rw_steps) if model.rw_steps else None
    # node/edge type embeddings do not depend on the frame: precompute once
    with torch.no_grad():
        emb_v, emb_c, emb_e = model.vn_emb(g.vn_type), model.cn_emb(g.cn_type), model.edge_emb(g.edge_type)
    use_amp = dtype is not None

    def refine(llr, prior, L, v2c, c2v, g_):
        hard = (L < 0).to(L.dtype)
        unsat = lay.syndrome(hard)
        frac_unsat = lay.sum_v(unsat[:, g.e_c]) / g.vn_deg
        f_v = torch.cat([llr_features(llr), llr_features(prior), llr_features(L), frac_unsat[..., None]], -1)
        f_c = torch.stack([_soft_syndrome(L, lay), 1.0 - 2.0 * unsat], -1)
        B = L.shape[0]
        if rw is not None:
            xin_v = torch.cat([f_v, rw[0].expand(B, -1, -1)], -1)
            xin_c = torch.cat([f_c, rw[1].expand(B, -1, -1)], -1)
        else:
            xin_v, xin_c = f_v, f_c
        ein = torch.stack([torch.tanh((v2c / 2).clamp(-15, 15)), torch.tanh((c2v / 2).clamp(-15, 15))], -1)
        with torch.autocast(dev.type, dtype=dtype, enabled=use_amp):
            h_v = model.vn_in(xin_v) + emb_v
            h_c = model.cn_in(xin_c) + emb_c
            e = model.edge_in(ein) + emb_e
            s_vc = model.s0_v(h_v)[:, g.e_v] + model.s0_e(e) if model.cross == "nb" else None
            for layer in model.layers:
                h_v, h_c, s_vc = _layer(layer, h_v, h_c, s_vc, f_v, f_c, e, g, lay, inv_dc, inv_dv)
            delta = model.head(model.ln_out(h_v)).squeeze(-1)
        base = prior if model.base == "prior" else llr
        return (base + delta.float()).clamp(-30, 30)

    return refine
