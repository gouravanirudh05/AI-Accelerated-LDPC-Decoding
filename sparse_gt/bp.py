"""Batched, differentiable sum-product BP (flooding schedule).

All messages live on the edge list of :class:`~sparse_gt.codes.LDPCCode`
(``[batch, E]``); node sums are ``index_add`` over ``edge_vn`` / ``edge_cn``.
One iteration is therefore a fixed number of dense kernels regardless of the
code, which is what makes n ~ 1000-2000 cheap on a GPU.

Check-node update, in the numerically stable "phi" form::

    phi(x) = log((e^x + 1) / (e^x - 1)) = log1p(2 / expm1(x)),   phi(phi(x)) = x
    |c2v_e| = phi( sum_{e' in N(c)} phi(|v2c_e'|) - phi(|v2c_e|) )
    sgn c2v_e = prod_{e' in N(c), e' != e} sgn v2c_e'

which is exactly ``2 atanh(prod tanh(v2c/2))`` over the other edges of the
check. Extrinsic exclusion is done by subtraction from the per-check total.
In float32 that subtraction loses precision only once the true extrinsic
magnitude exceeds ~12, i.e. for messages that are already decisive.

Early stopping follows AFF3CT's ``--dec-synd-depth 1``: a frame's output is
frozen at the first iteration whose hard decision satisfies every check. The
freeze uses ``torch.where`` so the whole decoder stays differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .codes import LDPCCode

# Bounds on the argument of phi. Below MIN_ABS a message carries no
# information (phi -> ~14.5); above MAX_ABS it is certain (phi ~ 1e-13).
MIN_ABS = 1e-6
MAX_ABS = 30.0
# phi(MAX_ABS): floor on the extrinsic phi-sum, so |c2v| <= MAX_ABS.
PHI_FLOOR = math.log1p(2.0 / math.expm1(MAX_ABS))


def phi(x: Tensor) -> Tensor:
    x = x.clamp(MIN_ABS, MAX_ABS)
    return torch.log1p(2.0 / torch.expm1(x))


@dataclass
class BPResult:
    llr: Tensor  # [B, n] output posterior LLR (frozen at convergence)
    c2v: Tensor  # [B, E] last check-to-variable messages, for resuming BP
    converged: Tensor  # [B] bool, syndrome satisfied within the budget
    iterations: Tensor  # [B] iterations actually used (budget if never converged)
    # [T+1, B, n]: trace[t] is the output BP would give with a budget of t
    # iterations (trace[0] = channel LLR). Only when return_trace=True.
    trace: Optional[Tensor] = None


class SumProductBP(nn.Module):
    def __init__(self, code: LDPCCode):
        super().__init__()
        self.code = code
        edge_vn, edge_cn = code.edges()
        self.register_buffer("edge_vn", edge_vn, persistent=False)
        self.register_buffer("edge_cn", edge_cn, persistent=False)
        self.n, self.m = code.n, code.m

    # ------------------------------------------------------------ helpers

    def _vn_sum(self, edge_values: Tensor) -> Tensor:
        out = edge_values.new_zeros(edge_values.shape[0], self.n)
        return out.index_add(1, self.edge_vn, edge_values)

    def _cn_sum(self, edge_values: Tensor) -> Tensor:
        out = edge_values.new_zeros(edge_values.shape[0], self.m)
        return out.index_add(1, self.edge_cn, edge_values)

    def syndrome_weight(self, llr: Tensor) -> Tensor:
        """Number of unsatisfied checks per frame for the hard decision of ``llr``."""
        return self.unsatisfied_checks(llr).sum(1)

    def unsatisfied_checks(self, llr: Tensor) -> Tensor:
        """``[B, m]`` bool, True where the hard decision violates the check."""
        hard = (llr < 0).to(llr.dtype)
        parity = self._cn_sum(hard[:, self.edge_vn])
        return torch.remainder(parity, 2.0) > 0.5

    def check_update(self, v2c: Tensor) -> Tensor:
        """Exact sum-product check-node update over the edge list."""
        p = phi(v2c.abs())
        total = self._cn_sum(p)
        magnitude = phi((total[:, self.edge_cn] - p).clamp_min(PHI_FLOOR))

        negative = (v2c < 0).to(v2c.dtype)
        neg_count = self._cn_sum(negative)
        extrinsic_neg = neg_count[:, self.edge_cn] - negative
        sign = 1.0 - 2.0 * torch.remainder(extrinsic_neg, 2.0)
        return sign * magnitude

    # ------------------------------------------------------------ forward

    def forward(
        self,
        channel_llr: Tensor,
        iterations: int,
        early_stop: bool = True,
        return_trace: bool = False,
        c2v_init: Optional[Tensor] = None,
    ) -> BPResult:
        """Run up to ``iterations`` flooding iterations.

        ``c2v_init`` resumes from earlier check-to-variable messages (e.g.
        BP -> correction -> BP); ``None`` starts from zero messages.
        """
        batch = channel_llr.shape[0]
        device = channel_llr.device
        c2v = (channel_llr.new_zeros(batch, self.edge_vn.numel())
               if c2v_init is None else c2v_init)

        output = channel_llr + self._vn_sum(c2v)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        used = torch.full((batch,), iterations, dtype=torch.long, device=device)
        if early_stop:
            done = self.syndrome_weight(output) == 0
            used = torch.where(done, torch.zeros_like(used), used)
        trace = [output] if return_trace else None

        for t in range(1, iterations + 1):
            if early_stop and bool(done.all()):
                if trace is not None:
                    trace.extend([output] * (iterations + 1 - len(trace)))
                break
            posterior = channel_llr + self._vn_sum(c2v)
            v2c = posterior[:, self.edge_vn] - c2v
            c2v = self.check_update(v2c)
            posterior = channel_llr + self._vn_sum(c2v)

            if early_stop:
                output = torch.where(done.unsqueeze(1), output, posterior)
                newly = (~done) & (self.syndrome_weight(posterior) == 0)
                used = torch.where(newly, torch.full_like(used, t), used)
                done = done | newly
            else:
                output = posterior
            if trace is not None:
                trace.append(output)

        converged = done if early_stop else self.syndrome_weight(output) == 0
        return BPResult(
            llr=output,
            c2v=c2v,
            converged=converged,
            iterations=used,
            trace=torch.stack(trace) if trace is not None else None,
        )
