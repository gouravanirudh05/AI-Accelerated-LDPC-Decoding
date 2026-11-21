"""Ablation variants of BP, one change at a time, to measure what each suspected
bottleneck costs (they are experiments, not a replacement for common/bp.py):

* aggregation backend for the scatter sums (``agg``):
    "index_add"  as in common/graph.py (atomic adds)
    "padded"     per-node padded edge lists: pad + gather + sum (no atomics, deterministic)
    "dense"      matmul with a one-hot [E, m] / [E, n] matrix
    "csr"        torch.sparse CSR matmul
* fused sign: the sign count and log-magnitude of the check update share one
  aggregation over a [B, E, 2] tensor instead of two (``fuse_sign``)
* early stopping without host synchronisation (``masked``): converged frames are
  frozen with torch.where instead of removed from the batch, so there is no
  ``.any()`` per iteration and no index compaction; the host checks
  ``done.all()`` once every ``check_every`` iterations
* CUDA graphs (``graph``): each block of ``check_every`` masked iterations is one
  captured graph, i.e. one launch instead of ~25 per iteration
* check-node rule (``cn``): sum-product (as trained) or normalised min-sum
* ``compile``: torch.compile the block of iterations (kernel fusion)
* dtype: fp32, or fp16 (min-sum only: the SPA log-domain clamp underflows in fp16)

With agg="index_add", fuse_sign=False, cn="spa", fp32 the masked decoder returns
the same LLRs and iteration counts as common.bp.bp_decode (tests/test_equivalence.py).
"""

import math

import torch

from common.bp import _cn_nms

AGGS = ("index_add", "padded", "dense", "csr")


class Aggregator:
    """sum over edges grouped by node: src [B, E(, c)] -> [B, size(, c)]."""

    def __init__(self, index, size, kind, dtype, device):
        self.kind, self.size = kind, size
        E = index.numel()
        self.index = index
        if kind == "padded":
            order = torch.argsort(index, stable=True)
            cnt = torch.bincount(index, minlength=size)
            dmax = int(cnt.max())
            start = torch.cumsum(cnt, 0) - cnt
            pos = torch.arange(E, device=device) - start[index[order]]
            pad = torch.full((size, dmax), E, dtype=torch.long, device=device)
            pad[index[order], pos] = order
            self.pad = pad                                        # [size, dmax], E = zero slot
        elif kind in ("dense", "csr"):
            S = torch.zeros(E, size, dtype=dtype, device=device)
            S[torch.arange(E, device=device), index] = 1
            self.S = S
            if kind == "csr":
                self.St = S.t().contiguous().to_sparse_csr()

    def __call__(self, src):
        if self.kind == "index_add":
            out = src.new_zeros((src.shape[0], self.size) + tuple(src.shape[2:]))
            return out.index_add_(1, self.index, src)
        if self.kind == "padded":
            p = torch.nn.functional.pad(src, (0, 0, 0, 1) if src.dim() == 3 else (0, 1))
            return p[:, self.pad].sum(2)
        if src.dim() == 3:  # [B, E, c] -> [B, size, c]
            return torch.stack([self(src[..., i]) for i in range(src.shape[2])], -1)
        if self.kind == "dense":
            return src @ self.S
        return (self.St @ src.t()).t()


class FastBP:
    def __init__(self, g, iters, agg="index_add", fuse_sign=False, cn="spa", alpha=0.75,
                 dtype=torch.float32, check_every=1, graph=False, compile=False):
        assert cn in ("spa", "nms") and agg in AGGS
        assert not (cn == "spa" and dtype == torch.float16), "SPA underflows in fp16"
        dev = g.e_v.device
        self.g, self.iters, self.cn, self.alpha, self.dtype = g, iters, cn, alpha, dtype
        self.fuse_sign, self.K = fuse_sign, max(1, min(check_every, iters))
        self.graph = graph and dev.type == "cuda"
        self.agg_c = Aggregator(g.e_c, g.m, agg, dtype, dev)
        self.agg_v = Aggregator(g.e_v, g.n, agg, dtype, dev)
        self._graphs = {}
        # torch.compile fuses the ~30 elementwise / gather passes of an iteration into a
        # few kernels: tests whether BP at large batch is bound by memory passes
        self._run = torch.compile(self._block, dynamic=False) if compile else self._block

    # one BP iteration on the full (masked) batch, all on-device ---------------
    def _cn_spa(self, v2c, eps=1e-7):
        g = self.g
        t = torch.tanh((v2c / 2.0).clamp(-15, 15))
        logm = torch.log(t.abs().clamp_min(1e-12))
        neg = (v2c < 0).to(v2c.dtype)
        if self.fuse_sign:
            both = torch.stack([logm, neg], -1)
            r = self.agg_c(both)[:, g.e_c] - both
            s, n_other = r[..., 0], r[..., 1]
        else:
            s = self.agg_c(logm)[:, g.e_c] - logm
            n_other = self.agg_c(neg)[:, g.e_c] - neg
        sg = 1.0 - 2.0 * torch.remainder(n_other, 2.0)
        p = (sg * torch.exp(s)).clamp(-1 + eps, 1 - eps)
        return 2.0 * torch.atanh(p)

    def _syndrome_ok(self, total):
        hard = (total < 0).to(total.dtype)
        return (torch.remainder(self.agg_c(hard[:, self.g.e_v]), 2.0) == 0).all(1)

    def _iter(self, st):
        g = self.g
        c2v = self._cn_spa(st["v2c"]) if self.cn == "spa" else _cn_nms(st["v2c"], g, self.alpha)
        total = st["x"] + self.agg_v(c2v)
        st["v2c"].copy_(total[:, g.e_v] - c2v)
        done = st["done"]
        st["out"].copy_(torch.where(done[:, None], st["out"], total))
        st["t"].add_(1)
        ok = self._syndrome_ok(total) & ~done & (st["t"] < self.iters)
        st["used"].copy_(torch.where(ok, st["t"], st["used"]))
        done.logical_or_(ok)

    def _block(self, st, n):
        for _ in range(n):
            self._iter(st)

    def _state(self, llr):
        x = llr.to(self.dtype)
        B = x.shape[0]
        return dict(x=x.clone(), v2c=x[:, self.g.e_v].clone(), out=x.clone(),
                    used=torch.full((B,), self.iters, device=x.device, dtype=torch.long),
                    done=torch.zeros(B, dtype=torch.bool, device=x.device),
                    t=torch.zeros((), dtype=torch.long, device=x.device))

    def _graphed(self, llr, n):
        key = (tuple(llr.shape), n)
        if key not in self._graphs:
            st = self._state(llr)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                snap = {k: v.clone() for k, v in st.items()}
                self._run(st, n)                                    # warm-up
                for k in st:
                    st[k].copy_(snap[k])
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                self._run(st, n)
            self._graphs[key] = (gr, st)
        return self._graphs[key]

    @torch.no_grad()
    def __call__(self, llr):
        """Returns (LLR [B, n] in fp32, iterations used [B])."""
        nblk = math.ceil(self.iters / self.K)
        sizes = [self.K] * (self.iters // self.K) + ([self.iters % self.K] if self.iters % self.K else [])
        if self.graph:
            st = None
            for i, n in enumerate(sizes):
                gr, s = self._graphed(llr, n)
                if st is None:
                    fresh = self._state(llr)
                    st = s
                    for k in st:
                        st[k].copy_(fresh[k])
                elif s is not st:
                    for k in s:
                        s[k].copy_(st[k])
                    st = s
                gr.replay()
                if i < nblk - 1 and bool(st["done"].all()):
                    break
            return st["out"].float().clone(), st["used"].clone()
        st = self._state(llr)
        for i, n in enumerate(sizes):
            self._run(st, n)
            if i < nblk - 1 and bool(st["done"].all()):
                break
        return st["out"].float(), st["used"]
