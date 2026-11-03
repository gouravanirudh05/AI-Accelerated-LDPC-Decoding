"""GPU-structured message-passing decoders on the Tanner graph (BP family + learned GNN).

A decoder here *is* a graph neural network on the Tanner graph: T rounds of
variable->check and check->variable messages. What changes between variants
is the check-node rule and which scalars are learned:

=================  ====================  =========================================
variant            check rule            learned parameters
=================  ====================  =========================================
``spa``            sum-product (exact)   none (the accuracy reference)
``ms``             min-sum               none
``nms``            min-sum               none, fixed normalisation beta (0.75)
``noms``           min-sum               beta_t, alpha_t per iteration (2T scalars,
                                         code-length independent)
``wms``            min-sum               a weight per edge per iteration (T*E) and
                                         alpha_t (weighted / "neural" min-sum)
=================  ====================  =========================================

Every learned operation is an elementwise scale or offset on a message, so a
learned decoder costs the same per iteration as min-sum. The point is to
reach a target FER in fewer, cheaper iterations, not to add a network.

GPU layout. Messages live on the edge list ``[B, E]`` (edges grouped by
check), but every node computation runs on a **dense padded view**:
``[B, m, dc_max]`` per check and ``[B, n, dv_max]`` per variable, built with
one gather each. Reductions are plain ``min/sum`` over the last axis:
no scatter atomics, fixed shapes and no host syncs, so a fixed-T decode can
be captured in a CUDA graph and run in fp16 (min-sum variants; sum-product's
log/exp is kept in fp32).

Codeword symmetry. All rules act on message magnitudes with sign products,
and learned scalars multiply magnitudes, so every variant is equivariant to
the transmitted codeword: training on the all-zero codeword is exact.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .codes import LDPCCode

VARIANTS = ("spa", "ms", "nms", "noms", "wms")
NMS_BETA = 0.75
SPA_MIN, SPA_MAX = 1e-6, 30.0
PHI_FLOOR = math.log1p(2.0 / math.expm1(SPA_MAX))


def _phi(x: Tensor) -> Tensor:
    x = x.clamp(SPA_MIN, SPA_MAX)
    return torch.log1p(2.0 / torch.expm1(x))


class TannerLayout(nn.Module):
    """Index tensors for the dense per-check / per-variable views of the edge list."""

    def __init__(self, code: LDPCCode):
        super().__init__()
        cn, vn = np.nonzero(code.H)  # edges grouped by check (row-major)
        E, m, n = cn.size, code.m, code.n
        dc, dv = code.H.sum(1).astype(int), code.H.sum(0).astype(int)
        if dc.min() < 2:
            raise ValueError("min-sum needs every check to have degree >= 2")
        self.E, self.m, self.n = E, m, n
        self.dc_max, self.dv_max = int(dc.max()), int(dv.max())

        cn_slots = np.full((m, self.dc_max), E, dtype=np.int64)  # E = padding slot
        rank_in_check = np.zeros(E, dtype=np.int64)
        start = 0
        for i in range(m):
            cn_slots[i, :dc[i]] = np.arange(start, start + dc[i])
            rank_in_check[start:start + dc[i]] = np.arange(dc[i])
            start += dc[i]
        vn_slots = np.full((n, dv.max()), E, dtype=np.int64)
        fill = np.zeros(n, dtype=np.int64)
        for e in range(E):
            vn_slots[vn[e], fill[vn[e]]] = e
            fill[vn[e]] += 1

        self.register_buffer("edge_vn", torch.as_tensor(vn), persistent=False)
        self.register_buffer("cn_slots", torch.as_tensor(cn_slots), persistent=False)
        self.register_buffer("vn_slots", torch.as_tensor(vn_slots), persistent=False)
        # Flat position of every edge inside the [m, dc_max] check view.
        self.register_buffer("edge_pos", torch.as_tensor(cn * self.dc_max + rank_in_check),
                             persistent=False)
        self.regular_checks = bool((dc == self.dc_max).all())

    def to_checks(self, edge_values: Tensor, pad: float) -> Tensor:
        padded = F.pad(edge_values, (0, 1), value=pad)
        return padded[:, self.cn_slots]  # [B, m, dc_max]

    def from_checks(self, check_values: Tensor) -> Tensor:
        flat = check_values.flatten(1)
        return flat if self.regular_checks else flat[:, self.edge_pos]  # [B, E]

    def vn_sum(self, edge_values: Tensor) -> Tensor:
        padded = F.pad(edge_values, (0, 1), value=0.0)
        return padded[:, self.vn_slots].sum(-1)  # [B, n]

    def syndrome_ok(self, llr: Tensor) -> Tensor:
        """[B] bool: hard decision satisfies every check."""
        hard = (llr < 0).to(torch.int8)
        per_edge = F.pad(hard[:, self.edge_vn], (0, 1), value=0)[:, self.cn_slots]
        return (per_edge.sum(-1, dtype=torch.int32) % 2 == 0).all(-1)


class MessagePassingDecoder(nn.Module):
    """Flooding message passing on the Tanner graph with a configurable check rule."""

    def __init__(self, code: LDPCCode, variant: str = "noms", iterations: int = 10):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.code, self.variant, self.T = code, variant, iterations
        self.layout = TannerLayout(code)
        if variant == "noms":
            self.beta = nn.Parameter(torch.full((iterations,), NMS_BETA))
            self.alpha = nn.Parameter(torch.zeros(iterations))
        elif variant == "wms":
            self.edge_weight = nn.Parameter(torch.full((iterations, self.layout.E), NMS_BETA))
            self.alpha = nn.Parameter(torch.zeros(iterations))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"variant={self.variant}, T={self.T}, params={self.num_parameters}"

    # ------------------------------------------------------------ check rules

    def _check_spa(self, v2c: Tensor) -> Tensor:
        lay = self.layout
        # log/exp are not fp16-safe: compute in fp32 (or fp64 if given).
        x = lay.to_checks(v2c.float() if v2c.dtype == torch.float16 else v2c, pad=SPA_MAX)
        p = _phi(x.abs())
        mag = _phi((p.sum(-1, keepdim=True) - p).clamp_min(PHI_FLOOR))
        neg = (x < 0)
        parity = (neg.sum(-1, keepdim=True, dtype=torch.int32) - neg.to(torch.int32)) % 2
        return lay.from_checks((1 - 2 * parity).to(mag.dtype) * mag).to(v2c.dtype)

    def _check_minsum(self, v2c: Tensor, t: int) -> Tensor:
        lay = self.layout
        x = lay.to_checks(v2c, pad=float("inf"))
        mag = x.abs()
        # Two plain reductions instead of topk(2): topk along a short axis is a
        # slow GPU kernel (measured ~3x the cost of the whole SPA update).
        min1, idx1 = mag.min(-1, keepdim=True)
        slot = torch.arange(lay.dc_max, device=x.device)
        is_min = slot == idx1
        min2 = mag.masked_fill(is_min, float("inf")).amin(-1, keepdim=True)
        ext = torch.where(is_min, min2, min1)  # min over the other edges of the check
        neg = (x < 0)
        parity = (neg.sum(-1, keepdim=True, dtype=torch.int32) - neg.to(torch.int32)) % 2
        sign = (1 - 2 * parity).to(x.dtype)
        if self.variant == "nms":
            ext = NMS_BETA * ext
        elif self.variant == "noms":
            ext = self.beta[t] * F.relu(ext - self.alpha[t])
        elif self.variant == "wms":
            ext = F.relu(ext - self.alpha[t])
        c2v = lay.from_checks(sign * ext)
        if self.variant == "wms":
            c2v = c2v * self.edge_weight[t]
        return c2v

    # ---------------------------------------------------------------- forward

    def forward(self, channel_llr: Tensor, iterations: int | None = None,
                return_trace: bool = True) -> Tensor:
        """Fixed-iteration decode: the trace ``[T, B, n]`` of posteriors, or only
        the final posterior ``[B, n]`` when ``return_trace=False``.

        No early stopping and no host synchronisation: shapes are static, so
        this is what gets captured in a CUDA graph (with ``return_trace=False``,
        the deployed form). Early-stopping *outputs* are recovered from the
        trace by :func:`first_valid`.
        """
        T = iterations or self.T
        if self.variant in ("noms", "wms") and T > self.T:
            raise ValueError(f"trained for {self.T} iterations, asked for {T}")
        lay = self.layout
        c2v = channel_llr.new_zeros(channel_llr.shape[0], lay.E)
        trace = []
        posterior = channel_llr
        for t in range(T):
            # The previous posterior is exactly this iteration's VN total.
            v2c = posterior[:, lay.edge_vn] - c2v
            c2v = self._check_spa(v2c) if self.variant == "spa" else self._check_minsum(v2c, t)
            posterior = channel_llr + lay.vn_sum(c2v)
            if return_trace:
                trace.append(posterior)
        return torch.stack(trace) if return_trace else posterior

    def decode(self, channel_llr: Tensor, iterations: int | None = None) -> Tensor:
        """Output with syndrome early-stopping semantics: ``[B, n]``."""
        return first_valid(self.layout, self.forward(channel_llr, iterations))


def first_valid(layout: TannerLayout, trace: Tensor) -> Tensor:
    """Per frame, the first iteration's output that satisfies every check (else the last).

    Equal to the output of a decoder that stops at the first valid syndrome,
    computed without host synchronisation.
    """
    T, B, n = trace.shape
    ok = torch.stack([layout.syndrome_ok(trace[t]) for t in range(T)])  # [T, B]
    ok[-1] = True
    first = ok.to(torch.int8).argmax(0)  # first True per frame
    return trace[first, torch.arange(B, device=trace.device)]


def multiloss(trace: Tensor) -> Tensor:
    """Mean BCE over all iterations' outputs (all-zero codeword targets)."""
    return F.binary_cross_entropy_with_logits(-trace.float(), torch.zeros_like(trace, dtype=torch.float32))
