"""BP(T) -> top-k least-confident bits -> sparse global transformer -> BP(R).

Pipeline (:class:`SparseGTDecoder`):

1. Run ``T`` BP iterations with syndrome early stopping. Frames whose hard
   decision already satisfies every check are **done**: the transformer never
   runs on them (early exit, also at inference).
2. For the remaining frames, select the ``k`` bits with the smallest
   ``|LLR|``.
3. A transformer with full attention **among those k bits only** (cost
   O(k^2), independent of n) predicts a correction ``delta`` per selected bit.
   Attention logits get a learned bias from the code structure between the
   selected bits: how many checks each pair shares, and how many of those
   are unsatisfied.
4. The correction is injected into the selected bits' channel LLRs and BP
   resumes for ``R`` iterations from its saved check messages. With R = 0 the
   output is ``L_T + sign * delta`` on the selected bits and ``L_T`` elsewhere.

Codeword symmetry. Every input is expressed relative to the bit's current
hard decision ``s = sign(L_T)``: LLRs enter as ``s * L``, plus syndrome-based
counts that depend only on the error pattern. The correction is applied as
``s * delta``. For symmetric BP, ``L(c) = (-1)^c L(0)``, so the whole pipeline
is equivariant to the transmitted codeword and training on the all-zero
codeword is exact, not a shortcut the model could exploit.

The output head is zero-initialised: an untrained decoder is exactly BP(T)
followed by BP(R), i.e. BP(T + R).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bp import SumProductBP
from .codes import LDPCCode

LLR_SCALE = 8.0  # LLR features are divided by this and clipped to +-LLR_CLIP
LLR_CLIP = 5.0
HISTORY = 4  # BP iterations of LLR history fed per bit (most recent first)
MAX_PAIR_COUNT = 3  # shared-check counts are clipped here for the bias tables


def sign_pm1(x: Tensor) -> Tensor:
    """+1 / -1 sign with 0 mapped to +1 (bit 0)."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


@dataclass
class GTConfig:
    T: int = 5
    k: int = 32
    R: int = 0
    d_model: int = 64
    heads: int = 4
    layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.0
    structure_bias: bool = True


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.dropout = dropout

    def forward(self, x: Tensor, bias: Tensor | None) -> Tensor:
        b, k, d = x.shape
        q, kk, v = self.qkv(self.ln1(x)).view(b, k, 3, self.heads, d // self.heads).unbind(2)
        q, kk, v = (t.transpose(1, 2) for t in (q, kk, v))  # [b, h, k, d_h]
        att = F.scaled_dot_product_attention(q, kk, v, attn_mask=bias,
                                             dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(att.transpose(1, 2).reshape(b, k, d))
        return x + self.ff(self.ln2(x))


class SparseGlobalTransformer(nn.Module):
    """Full attention over the k selected bits; outputs one delta per bit."""

    NUM_FEATURES = 2 + HISTORY + 6

    def __init__(self, cfg: GTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(self.NUM_FEATURES, cfg.d_model)
        self.blocks = nn.ModuleList(
            _Block(cfg.d_model, cfg.heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if cfg.structure_bias:
            n_vals = MAX_PAIR_COUNT + 1
            self.shared_bias = nn.Embedding(n_vals, cfg.heads)
            self.unsat_bias = nn.Embedding(n_vals, cfg.heads)
            nn.init.zeros_(self.shared_bias.weight)
            nn.init.zeros_(self.unsat_bias.weight)

    def forward(self, feats: Tensor, shared: Tensor | None, shared_unsat: Tensor | None) -> Tensor:
        bias = None
        if self.cfg.structure_bias:
            bias = (self.shared_bias(shared.clamp(max=MAX_PAIR_COUNT))
                    + self.unsat_bias(shared_unsat.clamp(max=MAX_PAIR_COUNT)))
            bias = bias.permute(0, 3, 1, 2).to(feats.dtype)  # [b, h, k, k]
        x = self.embed(feats)
        for block in self.blocks:
            x = block(x, bias)
        return self.head(self.ln(x)).squeeze(-1)  # [b, k]


class SparseGTDecoder(nn.Module):
    def __init__(self, code: LDPCCode, cfg: GTConfig):
        super().__init__()
        self.code, self.cfg = code, cfg
        self.bp = SumProductBP(code)
        self.gt = SparseGlobalTransformer(cfg)
        # Check list of every variable node, padded with -1: [n, dv_max].
        dv = code.H.sum(0)
        checks = np.full((code.n, int(dv.max())), -1, dtype=np.int64)
        for j in range(code.n):
            c = np.nonzero(code.H[:, j])[0]
            checks[j, :c.size] = c
        self.register_buffer("vn_checks", torch.as_tensor(checks), persistent=False)
        self.register_buffer("vn_degree", torch.as_tensor(dv, dtype=torch.float32), persistent=False)

    # ---------------------------------------------------------------- inputs

    def select(self, llr: Tensor) -> Tensor:
        return torch.topk(llr.abs(), self.cfg.k, dim=1, largest=False, sorted=True).indices

    def features(self, trace: Tensor, channel: Tensor, idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-bit features ``[b, k, F]`` and pairwise counts ``[b, k, k]``.

        ``trace`` is ``[T+1, b, n]`` for the active frames only.
        """
        L_T = trace[-1]
        b, k = idx.shape
        s = sign_pm1(L_T).gather(1, idx)

        def llr_feat(x: Tensor) -> Tensor:
            return (x.gather(1, idx) * s / LLR_SCALE).clamp(-LLR_CLIP, LLR_CLIP)

        unsat_checks = self.bp.unsatisfied_checks(L_T)  # [b, m] bool
        unsat_edges = unsat_checks[:, self.bp.edge_cn].to(L_T.dtype)
        unsat_per_bit = self.bp._vn_sum(unsat_edges).gather(1, idx)
        degree = self.vn_degree[idx]
        signs = trace < 0
        flips = (signs[1:] != signs[:-1]).sum(0).to(L_T.dtype).gather(1, idx)
        syndrome_w = unsat_checks.sum(1, keepdim=True).to(L_T.dtype) / self.code.m
        T = trace.shape[0] - 1
        history = [llr_feat(trace[max(T - h, 0)]) for h in range(1, HISTORY + 1)]
        rank = torch.arange(k, device=idx.device, dtype=L_T.dtype).expand(b, k) / k
        feats = torch.stack(
            [(L_T.gather(1, idx).abs() / LLR_SCALE).clamp(max=LLR_CLIP), llr_feat(channel), *history,
             flips / max(T, 1), unsat_per_bit / degree, unsat_per_bit, degree / degree.max(),
             syndrome_w.expand(b, k), rank],
            dim=-1,
        )

        shared = shared_unsat = None
        if self.cfg.structure_bias:
            chk = self.vn_checks[idx]  # [b, k, d]
            valid = chk >= 0
            unsat_of = torch.where(valid, unsat_checks.gather(1, chk.clamp(min=0).flatten(1)).view_as(chk),
                                   torch.zeros_like(valid))
            eq = (chk[:, :, None, :, None] == chk[:, None, :, None, :]) \
                & valid[:, :, None, :, None] & valid[:, None, :, None, :]
            shared = eq.sum((-1, -2))
            shared_unsat = (eq & unsat_of[:, :, None, :, None]).sum((-1, -2))
        return feats, shared, shared_unsat

    # --------------------------------------------------------------- forward

    def correct(self, trace: Tensor, channel: Tensor, c2v: Tensor) -> tuple[Tensor, Tensor]:
        """Corrected output LLR for active frames (differentiable w.r.t. the GT)."""
        L_T = trace[-1]
        idx = self.select(L_T)
        feats, shared, shared_unsat = self.features(trace, channel, idx)
        delta = self.gt(feats, shared, shared_unsat)
        signed = sign_pm1(L_T).gather(1, idx) * delta
        if self.cfg.R == 0:
            return L_T.scatter_add(1, idx, signed), idx
        new_channel = channel.scatter_add(1, idx, signed)
        resumed = self.bp(new_channel, self.cfg.R, early_stop=not self.training, c2v_init=c2v)
        return resumed.llr, idx

    def forward(self, channel_llr: Tensor) -> tuple[Tensor, dict]:
        """Decode a batch. Returns the output LLR ``[B, n]`` and aux info.

        aux["active"] marks frames the transformer ran on; aux["active_llr"]
        is their corrected output (what the training loss is computed on).
        """
        with torch.no_grad():
            first = self.bp(channel_llr, self.cfg.T, early_stop=True, return_trace=True)
        active = ~first.converged
        out = first.llr
        aux = {"active": active, "active_llr": None}
        if bool(active.any()):
            rows = torch.nonzero(active).squeeze(1)
            corrected, _ = self.correct(first.trace[:, rows], channel_llr[rows], first.c2v[rows])
            out = out.index_put((rows,), corrected)
            aux["active_llr"] = corrected
        return out, aux
