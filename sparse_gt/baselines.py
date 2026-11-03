"""Non-learned decoders the sparse-GT pipeline must beat at equal GPU latency.

* :func:`bp_compact` -- BP with syndrome early stopping that *removes*
  converged frames from the batch, so a batch costs its mean iteration count
  rather than its slowest frame's. This is the fair "BP as deployed" baseline
  (the freeze-based early stop in :class:`SumProductBP` keeps every frame in
  the batch until the last one converges).
* :func:`chase_bp` -- classical post-processing on the *same* least-reliable
  set the transformer sees: after BP(T), try all 2^p sign patterns on the p
  least-reliable bits (pinning them to +-``pin`` in the channel LLR), resume
  BP for R iterations on every pattern in parallel, and keep the first
  pattern that satisfies the syndrome (else the one with the lowest syndrome
  weight). If this already recovers most of the oracle gain, a learned
  corrector adds little.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

from .bp import SumProductBP
from .model import sign_pm1


@torch.no_grad()
def bp_compact(bp: SumProductBP, channel_llr: Tensor, iterations: int) -> tuple[Tensor, Tensor, Tensor]:
    """Early-stopped BP with converged-frame removal. Returns (llr, iterations_used, converged)."""
    batch = channel_llr.shape[0]
    device = channel_llr.device
    out = channel_llr.clone()
    used = torch.full((batch,), iterations, dtype=torch.long, device=device)
    converged = bp.syndrome_weight(channel_llr) == 0
    used[converged] = 0
    active = torch.nonzero(~converged).squeeze(1)
    L = channel_llr[active]
    c2v = L.new_zeros(L.shape[0], bp.edge_vn.numel())
    for t in range(1, iterations + 1):
        if active.numel() == 0:
            break
        posterior = L + bp._vn_sum(c2v)
        c2v = bp.check_update(posterior[:, bp.edge_vn] - c2v)
        posterior = L + bp._vn_sum(c2v)
        ok = bp.syndrome_weight(posterior) == 0
        if t == iterations:
            out[active] = posterior
            converged[active] = ok
            used[active[ok]] = t
            break
        if bool(ok.any()):
            done = active[ok]
            out[done] = posterior[ok]
            used[done] = t
            converged[done] = True
            keep = ~ok
            active, L, c2v = active[keep], L[keep], c2v[keep]
    return out, used, converged


@torch.no_grad()
def chase_bp(bp: SumProductBP, channel_llr: Tensor, T: int, p: int = 3, R: int = 5,
             pin: float = 20.0) -> Tensor:
    """BP(T), then Chase-style test patterns on the p least-reliable bits + BP(R)."""
    first = bp(channel_llr, T, early_stop=True)
    out = first.llr.clone()
    rows = torch.nonzero(~first.converged).squeeze(1)
    if rows.numel() == 0:
        return out
    L_T, ch, c2v = first.llr[rows], channel_llr[rows], first.c2v[rows]
    b, n = L_T.shape
    idx = torch.topk(L_T.abs(), p, dim=1, largest=False).indices  # [b, p]
    s = sign_pm1(L_T).gather(1, idx)  # current hard decisions (+1 = bit 0)
    patterns = torch.tensor(list(itertools.product((1.0, -1.0), repeat=p)),
                            device=ch.device, dtype=ch.dtype)  # [P, p], +1 keeps, -1 flips
    P = patterns.shape[0]
    pinned = (s[:, None, :] * patterns[None] * pin)  # [b, P, p]
    test = ch[:, None, :].expand(b, P, n).clone()
    test.scatter_(2, idx[:, None, :].expand(b, P, p), pinned)
    res = bp(test.reshape(b * P, n), R, early_stop=True,
             c2v_init=c2v[:, None, :].expand(b, P, -1).reshape(b * P, -1))
    weight = bp.syndrome_weight(res.llr).view(b, P)
    # First satisfied pattern wins (pattern 0 = no flips); else lowest weight.
    score = weight * (P + 1) + torch.arange(P, device=ch.device)
    best = score.argmin(1)
    chosen = res.llr.view(b, P, n)[torch.arange(b, device=ch.device), best]
    out[rows] = chosen
    return out
