"""Flooding belief-propagation baselines: sum-product (SPA) and normalised min-sum."""

import torch

from .graph import scatter_sum


def _check_sign(v2c, g):
    neg = (v2c < 0).to(v2c.dtype)
    n_other = scatter_sum(neg, g.e_c, g.m)[:, g.e_c] - neg
    return 1.0 - 2.0 * torch.remainder(n_other, 2.0)


def _cn_spa(v2c, g, eps=1e-7):
    t = torch.tanh((v2c / 2.0).clamp(-15, 15))
    logm = torch.log(t.abs().clamp_min(1e-12))
    s = scatter_sum(logm, g.e_c, g.m)[:, g.e_c] - logm
    p = (_check_sign(v2c, g) * torch.exp(s)).clamp(-1 + eps, 1 - eps)
    return 2.0 * torch.atanh(p)


def bp_iterations(prior, g, iters):
    """Differentiable sum-product run started from ``prior`` (used for training
    the BP-GNN hybrid). Returns (marginal LLR, v2c, c2v)."""
    v2c = prior[:, g.e_v]
    for _ in range(iters):
        c2v = _cn_spa(v2c, g, eps=1e-6)  # milder clamp keeps atanh gradients bounded
        total = prior + scatter_sum(c2v, g.e_v, g.n)
        v2c = total[:, g.e_v] - c2v
    return total, v2c, c2v


@torch.no_grad()
def bp_stage(prior, g, iters, early_stop=True):
    """One inference BP run with per-iteration early stopping (batch compaction).

    Returns full-batch (marginal, v2c, c2v, converged mask, iterations used)."""
    B = prior.shape[0]
    TOT = prior.clone()
    V2C = prior[:, g.e_v].clone()
    C2V = torch.zeros_like(V2C)
    ok = torch.zeros(B, dtype=torch.bool, device=prior.device)
    used = torch.full((B,), iters, device=prior.device, dtype=torch.long)
    active = torch.arange(B, device=prior.device)
    x, v2c = prior, V2C
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
def staged_decode(llr, g, stages, iters, refine, early_stop=True):
    """BP(iters) -> refine -> BP(iters) -> ... with ``stages`` refinements.

    ``refine(llr, prior, L, v2c, c2v, g)`` returns the prior of the next BP run
    (a GNN for the BP-GNN hybrid, random perturbation for the baseline). With
    early stopping only frames that are still unsolved are refined and re-run.
    Returns (final LLR, total BP iterations used per frame)."""
    B = llr.shape[0]
    out = llr.clone()
    used = torch.zeros(B, device=llr.device, dtype=torch.long)
    active = torch.arange(B, device=llr.device)
    x, prior = llr, llr
    for s in range(stages + 1):
        L, v2c, c2v, ok, u = bp_stage(prior, g, iters, early_stop)
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


def random_perturbation(scale=0.5):
    """Baseline refinement (cf. random perturbation post-processing compared in
    the paper): restart BP with channel LLRs of bits touching unsatisfied checks
    multiplicatively perturbed."""
    def refine(llr, prior, L, v2c, c2v, g):
        unsat = g.syndrome((L < 0).float())
        touched = scatter_sum(unsat[:, g.e_c], g.e_v, g.n) > 0
        noisy = llr * (1.0 + scale * torch.randn_like(llr))
        return torch.where(touched, noisy, llr)
    return refine


def _cn_nms(v2c, g, alpha):
    a = v2c.abs()
    B = a.shape[0]
    idx = g.e_c.expand(B, -1)
    inf = torch.full((B, g.m), float("inf"), device=a.device, dtype=a.dtype)
    min1 = inf.scatter_reduce(1, idx, a, "amin")
    is_min = a == min1[:, g.e_c]
    cnt = scatter_sum(is_min.to(a.dtype), g.e_c, g.m)
    min2 = inf.scatter_reduce(1, idx, torch.where(is_min, torch.full_like(a, float("inf")), a), "amin")
    min2 = torch.where(cnt >= 2, min1, min2)
    mag = torch.where(is_min, min2[:, g.e_c], min1[:, g.e_c])
    return alpha * _check_sign(v2c, g) * mag


@torch.no_grad()
def bp_decode(llr, g, iters=20, mode="spa", alpha=0.75, early_stop=True):
    """Returns (final a-posteriori LLR [B, n], iterations used [B])."""
    B = llr.shape[0]
    out = llr.clone()
    used = torch.full((B,), iters, device=llr.device, dtype=torch.long)
    active = torch.arange(B, device=llr.device)
    x = llr
    v2c = x[:, g.e_v]
    for t in range(iters):
        c2v = _cn_spa(v2c, g) if mode == "spa" else _cn_nms(v2c, g, alpha)
        total = x + scatter_sum(c2v, g.e_v, g.n)
        v2c = total[:, g.e_v] - c2v
        out[active] = total
        if early_stop and t < iters - 1:
            ok = (g.syndrome((total < 0).float()) == 0).all(1)
            if ok.any():
                used[active[ok]] = t + 1
                keep = ~ok
                if not keep.any():
                    break
                active, x, v2c = active[keep], x[keep], v2c[keep]
    return out, used
