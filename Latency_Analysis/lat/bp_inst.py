"""Instrumented copies of common/bp.py: same tensor operations in the same order
(tests/test_equivalence.py checks bit-exact outputs), split into named regions.

Per BP iteration the regions are
  cn        check-node update (sum-product)
    cn/tanh_log    tanh(v2c/2), log|.|                      (elementwise, E per frame)
    cn/mag_scatter scatter_sum of log|.| to checks + gather back (index_add, E)
    cn/sign        sign product: scatter_sum of negatives + gather + remainder
    cn/exp_atanh   exp, clamp, 2 atanh                       (elementwise, E)
  vn        variable-node update: scatter_sum(c2v -> bits) + channel prior
  v2c       gather total to edges, subtract c2v
  writeback write the active frames back into the full-batch output
  syndrome  hard decision + syndrome (gather + index_add + remainder + all)
  host_sync ``ok.any()`` / ``keep.any()``: copies a bool to the host and blocks
            until the GPU has finished everything queued so far
  compact   index the still-active frames (active[keep], x[keep], v2c[keep])
"""

import torch

from common.bp import _check_sign, _cn_nms
from common.graph import scatter_sum

from .regions import OFF


def cn_spa(v2c, g, R=OFF, eps=1e-7):
    with R("tanh_log"):
        t = torch.tanh((v2c / 2.0).clamp(-15, 15))
        logm = torch.log(t.abs().clamp_min(1e-12))
    with R("mag_scatter"):
        s = scatter_sum(logm, g.e_c, g.m)[:, g.e_c] - logm
    with R("sign"):
        sg = _check_sign(v2c, g)
    with R("exp_atanh"):
        p = (sg * torch.exp(s)).clamp(-1 + eps, 1 - eps)
        return 2.0 * torch.atanh(p)


def cn_update(v2c, g, mode, alpha, R):
    with R("cn"):
        return cn_spa(v2c, g, R) if mode == "spa" else _cn_nms(v2c, g, alpha)


@torch.no_grad()
def bp_decode(llr, g, iters=20, mode="spa", alpha=0.75, early_stop=True, R=OFF):
    """common.bp.bp_decode with regions."""
    B = llr.shape[0]
    out = llr.clone()
    used = torch.full((B,), iters, device=llr.device, dtype=torch.long)
    active = torch.arange(B, device=llr.device)
    x = llr
    v2c = x[:, g.e_v]
    for t in range(iters):
        c2v = cn_update(v2c, g, mode, alpha, R)
        with R("vn"):
            total = x + scatter_sum(c2v, g.e_v, g.n)
        with R("v2c"):
            v2c = total[:, g.e_v] - c2v
        with R("writeback"):
            out[active] = total
        if early_stop and t < iters - 1:
            with R("syndrome"):
                ok = (g.syndrome((total < 0).float()) == 0).all(1)
            with R("host_sync"):
                any_ok = bool(ok.any())
            if any_ok:
                with R("compact"):
                    used[active[ok]] = t + 1
                    keep = ~ok
                with R("host_sync"):
                    any_keep = bool(keep.any())
                if not any_keep:
                    break
                with R("compact"):
                    active, x, v2c = active[keep], x[keep], v2c[keep]
    return out, used


@torch.no_grad()
def bp_stage(prior, g, iters, early_stop=True, R=OFF):
    """common.bp.bp_stage with regions."""
    B = prior.shape[0]
    with R("init"):
        TOT = prior.clone()
        V2C = prior[:, g.e_v].clone()
        C2V = torch.zeros_like(V2C)
        ok = torch.zeros(B, dtype=torch.bool, device=prior.device)
        used = torch.full((B,), iters, device=prior.device, dtype=torch.long)
        active = torch.arange(B, device=prior.device)
    x, v2c = prior, V2C
    for t in range(iters):
        c2v = cn_update(v2c, g, "spa", None, R)
        with R("vn"):
            total = x + scatter_sum(c2v, g.e_v, g.n)
        with R("v2c"):
            v2c = total[:, g.e_v] - c2v
        with R("writeback"):
            TOT[active], V2C[active], C2V[active] = total, v2c, c2v
        if early_stop:
            with R("syndrome"):
                okk = (g.syndrome((total < 0).float()) == 0).all(1)
            with R("host_sync"):
                any_ok = bool(okk.any())
            if any_ok:
                with R("compact"):
                    used[active[okk]] = t + 1
                    ok[active[okk]] = True
                    keep = ~okk
                with R("host_sync"):
                    any_keep = bool(keep.any())
                if not any_keep or t == iters - 1:
                    break
                with R("compact"):
                    active, x, v2c = active[keep], x[keep], v2c[keep]
    if not early_stop:
        with R("syndrome"):
            ok = (g.syndrome((TOT < 0).float()) == 0).all(1)
    return TOT, V2C, C2V, ok, used


@torch.no_grad()
def staged_decode(llr, g, stages, iters, refine, early_stop=True, R=OFF, stats=None):
    """common.bp.staged_decode with regions: stage<s>/bp, stage<s>/refine and
    stage<s>/compact. ``refine(..., R)`` gets the region object so the SGT can
    split itself further. ``stats`` (list) receives the active batch size per stage."""
    B = llr.shape[0]
    out = llr.clone()
    used = torch.zeros(B, device=llr.device, dtype=torch.long)
    active = torch.arange(B, device=llr.device)
    x, prior = llr, llr
    for s in range(stages + 1):
        with R(f"stage{s}"):
            if stats is not None:
                stats.append(int(active.numel()))
            with R("bp"):
                L, v2c, c2v, ok, u = bp_stage(prior, g, iters, early_stop, R)
            with R("writeback"):
                out[active] = L
                used[active] += u
            if s == stages:
                break
            if early_stop:
                with R("host_sync"):
                    keep = ~ok
                    any_keep = bool(keep.any())
                if not any_keep:
                    break
                with R("compact"):
                    active, x, prior, L, v2c, c2v = (a[keep] for a in (active, x, prior, L, v2c, c2v))
            with R("refine"):
                prior = refine(x, prior, L, v2c, c2v, g, R)
    return out, used
