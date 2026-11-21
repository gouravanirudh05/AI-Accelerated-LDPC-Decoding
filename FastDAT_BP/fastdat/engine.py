"""Sync-free staged BP -> module -> BP decoder (inference only).

Same schedule and early-stop semantics as common.bp.staged_decode (and, with
``warm=True``, SparseGT_BP's staged_decode_resume):

* a frame whose hard decision satisfies all checks after any BP iteration is
  frozen: its output is the LLR of that iteration and its iteration count stops;
* frames that are not done after a BP run are refined by the module, and the
  next BP run starts from the module's prior.

What changes is *how* it runs on a GPU:

1. **Masking instead of batch compaction.** The original removes converged
   frames from the batch, which needs ``ok.any()`` (a device -> host copy) after
   every iteration and index_select/copy of every state tensor. Here every
   frame stays in the batch and converged frames are frozen with
   ``torch.where``, so the shapes never change and nothing waits for the host.
2. **One host check per block.** The host reads ``done.all()`` once every
   ``check_every`` iterations (default 5) to skip the rest of a BP run when the
   whole (sub-)batch is solved. ``check_every=0``: no checks inside a run.
2b. **Compaction only at stage boundaries** (``compact="pow2"``, default). Before
   the module runs, the frames that are not done are gathered into a smaller
   batch whose size is rounded up to a power of two (padding rows are marked
   done), so the module and the next BP run cost what the *failing* frames
   need, as in the original, but with a handful of fixed shapes that CUDA graphs
   can replay. This costs one device -> host read per stage (at most
   ``stages``), instead of one per BP iteration. ``compact="none"`` keeps the
   whole batch to the end; with check_every=0 that is a fixed schedule with no
   host synchronisation at all.
3. **CUDA graphs.** Every block (``check_every`` BP iterations, plus the module
   and the restart of BP at the end of a stage) is captured once per batch size
   and replayed: one launch instead of ~30 per BP iteration and a few hundred
   per module call. All graphs share one memory pool.
4. **Padded layout** (``agg="padded"``, fastdat.layout): gather + sum instead of
   index_add scatters in BP and the syndrome.

Frames that are already done but share a (sub-)batch with unsolved frames still
go through BP (masked out): extra arithmetic, bounded by the power-of-two
rounding after the first stage.

With ``agg="index_add"`` and the original refine function the outputs are
bit-identical to staged_decode (tests/test_equivalence.py).
"""

import math

import torch

from common.bp import _cn_spa


class StagedEngine:
    def __init__(self, g, stages, iters, refine=None, agg="padded", check_every=5, graph=True,
                 warm=False, compile=False, layout=None, compact="pow2"):
        from .layout import Layout
        assert compact in ("pow2", "exact", "none")
        self.g, self.stages, self.iters, self.refine, self.warm = g, stages, iters, refine, warm
        assert stages == 0 or refine is not None
        self.lay = layout or Layout(g, agg)
        self.K = iters if check_every in (0, None) else max(1, min(check_every, iters))
        self.host_checks = check_every not in (0, None)
        self.graph = graph and g.e_v.device.type == "cuda"
        self.compact = compact if not (self.graph and compact == "exact") else "pow2"
        self._graphs, self._bufs, self._pool = {}, {}, None
        self.chunks = [self.K] * (iters // self.K) + ([iters % self.K] if iters % self.K else [])
        self._block_fn = torch.compile(self._block, dynamic=False) if compile else self._block
        self.stats = dict(calls=0, host_syncs=0, stage_rows=[0] * (stages + 1))

    # ------------------------------------------------------------ BP pieces
    def _cn(self, v2c, eps=1e-7):
        """_cn_spa of common.bp with the layout's sums (identical for agg=index_add)."""
        if self.lay.agg == "index_add":
            return _cn_spa(v2c, self.g, eps)
        e_c = self.g.e_c
        t = torch.tanh((v2c / 2.0).clamp(-15, 15))
        logm = torch.log(t.abs().clamp_min(1e-12))
        neg = (v2c < 0).to(v2c.dtype)
        both = self.lay.sum_c(torch.stack([logm, neg], -1))[:, e_c]
        s = both[..., 0] - logm
        n_other = both[..., 1] - neg
        sign = 1.0 - 2.0 * torch.remainder(n_other, 2.0)
        p = (sign * torch.exp(s)).clamp(-1 + eps, 1 - eps)
        return 2.0 * torch.atanh(p)

    def _iter(self, st):
        e_v = self.g.e_v
        c2v = self._cn(st["v2c"])
        total = st["prior"] + self.lay.sum_v(c2v)
        st["v2c"].copy_(total[:, e_v] - c2v)
        st["c2v"].copy_(c2v)
        st["L"].copy_(total)
        act = ~st["done"]
        st["out"].copy_(torch.where(act[:, None], total, st["out"]))
        st["used"].add_(act.to(st["used"].dtype))
        ok = (self.lay.syndrome((total < 0).to(total.dtype)) == 0).all(1) & act
        st["done"].logical_or_(ok)

    def _block(self, st, n, refine_before):
        """[module + BP restart] + n BP iterations, all on the device."""
        if refine_before:
            new = self.refine(st["llr"], st["prior"], st["L"], st["v2c"], st["c2v"], self.g)
            st["prior"].copy_(torch.where(st["done"][:, None], st["prior"], new.to(st["prior"].dtype)))
            e_v = self.g.e_v
            if self.warm:  # SparseGT_BP staged_decode_resume: continue from the last check messages
                st["v2c"].copy_((st["prior"] + self.lay.sum_v(st["c2v"]))[:, e_v] - st["c2v"])
            else:          # common.bp.bp_stage: cold restart from the new prior
                st["v2c"].copy_(st["prior"][:, e_v])
        for _ in range(n):
            self._iter(st)

    # ------------------------------------------------------------ state
    KEYS = ("llr", "prior", "L", "out", "v2c", "c2v", "used", "done")

    def _alloc(self, B, dev):
        n, E = self.g.n, self.g.num_edges
        f = lambda *s: torch.zeros(*s, device=dev)
        return dict(llr=f(B, n), prior=f(B, n), L=f(B, n), out=f(B, n), v2c=f(B, E), c2v=f(B, E),
                    used=torch.zeros(B, dtype=torch.long, device=dev),
                    done=torch.zeros(B, dtype=torch.bool, device=dev))

    @staticmethod
    def _fresh(llr):
        x = llr.float()
        B = x.shape[0]
        return dict(llr=x, prior=x.clone(), L=x.clone(), out=x.clone(), v2c=None, c2v=None,
                    used=torch.zeros(B, dtype=torch.long, device=x.device),
                    done=torch.zeros(B, dtype=torch.bool, device=x.device))

    def _fill(self, st, full, rows, nb):
        """st[:nb] = full[rows] (all rows if rows is None); padding rows copy row 0 and
        are marked done, so they never change the result."""
        size = st["done"].shape[0]
        for k in self.KEYS:
            dst, src = st[k][:nb], full[k]
            if src is None:  # fresh decode: BP starts from v2c = prior, c2v = 0
                if k == "c2v":
                    dst.zero_()
                else:
                    dst.copy_(full["prior"][:, self.g.e_v])
            elif rows is None:
                dst.copy_(src)
            else:
                torch.index_select(src, 0, rows, out=dst)
            if nb < size:
                st[k][nb:].copy_(st[k][:1].expand_as(st[k][nb:]))
        if nb < size:
            st["done"][nb:] = True

    def _store(self, st, full, rows, nb):
        for k in self.KEYS:
            if rows is None:
                full[k] = st[k][:nb].clone()
            else:
                full[k].index_copy_(0, rows, st[k][:nb])

    def _bucket(self, nb, B):
        if self.compact == "exact":
            return nb
        return min(B, 1 << max(0, math.ceil(math.log2(nb))))

    # ------------------------------------------------------------ graphs
    def _graph(self, st, size, n, rb):
        key = (size, n, rb)
        if key not in self._graphs:
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            snap = {k: v.clone() for k, v in st.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # warm-up (lazy caches, cuBLAS workspaces, compile)
                self._block_fn(st, n, rb)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=self._pool):
                self._block_fn(st, n, rb)
            for k in st:  # capture also executed the block: restore the inputs
                st[k].copy_(snap[k])
            self._graphs[key] = gr
        return self._graphs[key]

    def _run(self, st, size, n, rb):
        if self.graph:
            self._graph(st, size, n, rb).replay()
        else:
            self._block_fn(st, n, rb)

    def _buf(self, size, dev):
        if not self.graph:
            return self._alloc(size, dev)
        if size not in self._bufs:
            self._bufs[size] = self._alloc(size, dev)
        return self._bufs[size]

    # ------------------------------------------------------------ decode
    def _sync(self, t):
        self.stats["host_syncs"] += 1
        return t.item() if t.numel() == 1 else t.cpu()

    @torch.no_grad()
    def __call__(self, llr):
        """Returns (final LLR [B, n] fp32, BP iterations used per frame [B])."""
        B, dev = llr.shape[0], llr.device
        self.stats["calls"] += 1
        full = self._fresh(llr)
        rows, nb, size = None, B, B
        st = self._buf(B, dev)
        self._fill(st, full, None, B)
        for s in range(self.stages + 1):
            if s > 0 and self.compact != "none":
                rows = torch.nonzero(~full["done"]).squeeze(1)   # one device -> host read per stage
                self.stats["host_syncs"] += 1
                nb = rows.numel()
                if nb == 0:
                    break
                size = self._bucket(nb, B)
                st = self._buf(size, dev)
                self._fill(st, full, rows, nb)
            self.stats["stage_rows"][s] += size
            for i, n in enumerate(self.chunks):
                self._run(st, size, n, s > 0 and i == 0)
                if self.host_checks and i < len(self.chunks) - 1 and bool(self._sync(st["done"].all())):
                    break
            all_done = (self.compact == "none" and self.host_checks and s < self.stages
                        and bool(self._sync(st["done"].all())))
            if self.compact != "none" or s == self.stages or all_done:
                self._store(st, full, rows, nb)
            if all_done:
                break
        return full["out"], full["used"]

    def host_syncs_per_decode(self):
        """Upper bound on device -> host reads of one decode."""
        per_stage = len(self.chunks) - 1 if self.host_checks else 0
        between = self.stages if (self.compact != "none" or self.host_checks) else 0
        return (self.stages + 1) * per_stage + between
