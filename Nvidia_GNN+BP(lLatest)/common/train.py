"""Training loop: multi-iteration BCE + soft-syndrome loss, random Eb/N0 per frame.

Options (cfg keys, all optional unless noted):
* schedule="cosine" (default; cosine decay over the step / time budget) or
  "plateau" (halve the LR when the validation score has not improved for
  ``patience`` evaluations, restore the best weights, stop below ``lr_min``);
* hard_frac > 0: hard-example mining for the BP-GNN hybrid. With ``pool_bins``
  set, frames that plain BP(mining_bp_iters) fails on are collected into one pool
  per Eb/N0 band (so rare high-SNR trapping-set failures are not swamped by
  low-SNR ones) and each batch draws hard_frac of its frames evenly from the pools;
* val_mode="hard": model selection on a fixed set of BP failures per band
  (score = mean frame error rate over bands of the frames the model still gets
  wrong), otherwise BLER + BER at ``val_ebno``;
* punct_hint=a, punct_hint_frac=f: curriculum for punctured codes (5G NR). Training
  frames keep a fraction of the channel LLR on punctured bits, decaying linearly
  from a to 0 over the first f of the time budget (validation always uses true
  puncturing; the plateau schedule only starts once the hint is gone).
"""

import copy
import math
import random
import time

import torch
import torch.nn.functional as F

from .bp import bp_decode
from .gnn_decoder import soft_syndrome_loss


def gnn_loss(outs, c, g, lam_syn):
    bce = sum(F.binary_cross_entropy_with_logits(-L, c) for L in outs) / len(outs)
    syn = soft_syndrome_loss(outs[-1], g) if lam_syn > 0 else bce.new_zeros(())
    return bce + lam_syn * syn, bce, syn


class _Seeded:
    """Context manager: run with a fixed seed, then restore the RNG state."""

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.state = torch.random.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)

    def __exit__(self, *a):
        torch.random.set_rng_state(self.state)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


@torch.no_grad()
def validate(model, codes, ebno, frames, batch, seed=1234):
    """BER/BLER of model.decode on a fixed (seeded) validation set."""
    be = fe = nb = nf = 0
    with _Seeded(seed):
        for code in codes:
            for _ in range(max(1, frames // batch)):
                c, llr = code.sample(batch, ebno)
                L, _ = model.decode(llr, code.graph)
                err = ((L < 0).float() != c)
                be += err[:, code.info_pos].sum().item()
                fe += err.any(1).sum().item()
                nb += batch * code.k
                nf += batch
    return be / nb, fe / nf


@torch.no_grad()
def collect_failures(code, ebno, bp_iters, want, chunk, max_frames):
    """Frames (c, llr) at Eb/N0 range ``ebno`` on which BP(bp_iters) fails."""
    cs, ls, got, seen = [], [], 0, 0
    while got < want and seen < max_frames:
        c, llr = code.sample(chunk, ebno)
        L, _ = bp_decode(llr, code.graph, bp_iters)
        fail = ((L < 0).float() != c).any(1)
        if fail.any():
            cs.append(c[fail])
            ls.append(llr[fail])
            got += int(fail.sum())
        seen += chunk
    if not cs:
        return None
    return torch.cat(cs)[:want], torch.cat(ls)[:want]


def _bands(rng, n):
    lo, hi = rng
    edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


class HardPools:
    """One FIFO pool of BP-failure frames per Eb/N0 band."""

    def __init__(self, code, cfg):
        self.code, self.cfg = code, cfg
        self.bands = _bands(cfg["train_ebno"], cfg["pool_bins"])
        self.pools = [None] * len(self.bands)

    def refresh(self, want=None):
        cfg = self.cfg
        for i, band in enumerate(self.bands):
            new = collect_failures(self.code, band, cfg["mining_bp_iters"], want or cfg["pool_new"],
                                   cfg["mining_chunk"], cfg["mining_max_frames"])
            if new is None:
                continue
            if self.pools[i] is None:
                self.pools[i] = new
            else:
                self.pools[i] = tuple(torch.cat([p, q])[-cfg["pool_size"]:] for p, q in zip(self.pools[i], new))

    def sample(self, k):
        live = [p for p in self.pools if p is not None]
        if not live or k == 0:
            return None
        per = [k // len(live) + (1 if i < k % len(live) else 0) for i in range(len(live))]
        cs, ls = [], []
        for (c, l), n in zip(live, per):
            idx = torch.randint(0, c.shape[0], (n,), device=c.device)
            cs.append(c[idx])
            ls.append(l[idx])
        return torch.cat(cs), torch.cat(ls)

    def sizes(self):
        return [0 if p is None else p[0].shape[0] for p in self.pools]


@torch.no_grad()
def mine_hard(code, batch, cfg):
    """Unstratified mining: sample cfg['mining_factor'] x batch frames, run plain BP
    and fill the batch with failing frames first (up to cfg['hard_frac'])."""
    c, llr = code.sample(batch * cfg["mining_factor"], cfg["train_ebno"])
    L, _ = bp_decode(llr, code.graph, cfg["mining_bp_iters"])
    fail = ((L < 0).float() != c).any(1)
    hard = torch.nonzero(fail).squeeze(1)
    easy = torch.nonzero(~fail).squeeze(1)
    n_hard = min(len(hard), int(batch * cfg["hard_frac"]))
    idx = torch.cat([hard[:n_hard], easy[:batch - n_hard]])
    return c[idx], llr[idx]


class HardValidation:
    """Fixed set of BP failures per band; score = mean FER of the model on them."""

    def __init__(self, code, cfg, log):
        self.code = code
        self.sets = []
        with _Seeded(4321):
            for band in _bands(cfg["val_hard_ebno"], cfg.get("val_hard_bins", 3)):
                s = collect_failures(code, band, cfg["mining_bp_iters"], cfg["val_hard_frames"],
                                     cfg["mining_chunk"], cfg["val_hard_max_frames"])
                if s is not None:
                    self.sets.append((band, s))
        log(f"  hard validation set: " + ", ".join(
            f"{a:.2f}-{b:.2f} dB: {s[0].shape[0]}" for (a, b), s in self.sets))

    @torch.no_grad()
    def __call__(self, model, batch):
        fers = []
        for _, (c, llr) in self.sets:
            fe = 0
            for i in range(0, c.shape[0], batch):
                L, _ = model.decode(llr[i:i + batch], self.code.graph)
                fe += ((L < 0).float() != c[i:i + batch]).any(1).sum().item()
            fers.append(fe / c.shape[0])
        return sum(fers) / len(fers), fers


def train_gnn(model, codes, cfg, log=print):
    """codes: list of Code objects (one is sampled per step, e.g. several 5G sizes).

    cfg keys: steps, max_minutes, batch, lr, lr_min, train_ebno, lam_syn,
    eval_every, val_ebno, val_frames, grad_clip (+ the options in the module doc).
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    steps = cfg["steps"]
    plateau = cfg.get("schedule", "cosine") == "plateau"
    t0 = time.time()
    history = []
    best = (math.inf, None)
    rng = random.Random(0)
    batches = cfg["batch"] if isinstance(cfg["batch"], (list, tuple)) else [cfg["batch"]] * len(codes)
    pools = None
    if cfg.get("hard_frac", 0) > 0 and cfg.get("pool_bins"):
        pools = [HardPools(c, cfg) for c in codes]
        for p in pools:
            p.refresh(want=cfg.get("pool_init", cfg["pool_new"]))
        log(f"  hard pools (per band): {[p.sizes() for p in pools]}")
    hard_val = HardValidation(codes[0], cfg, log) if cfg.get("val_mode") == "hard" else None
    lr, bad = cfg["lr"], 0
    for step in range(1, steps + 1):
        if not plateau:
            # cosine decay over progress = whichever of step / time budget runs out first
            progress = min(1.0, max(step / steps, (time.time() - t0) / 60 / cfg["max_minutes"]))
            lr = cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = lr
        i = rng.randrange(len(codes))
        code = codes[i]
        hint = 0.0
        if cfg.get("punct_hint"):
            frac = (time.time() - t0) / 60 / (cfg["punct_hint_frac"] * cfg["max_minutes"])
            hint = cfg["punct_hint"] * max(0.0, 1.0 - frac)
        if pools is not None:
            if step % cfg["pool_refresh_every"] == 0:
                pools[i].refresh()
            n_hard = int(batches[i] * cfg["hard_frac"])
            hard = pools[i].sample(n_hard)
            c, llr = code.sample(batches[i] - (0 if hard is None else hard[0].shape[0]), cfg["train_ebno"])
            if hard is not None:
                c, llr = torch.cat([hard[0], c]), torch.cat([hard[1], llr])
        elif cfg.get("hard_frac", 0) > 0:
            c, llr = mine_hard(code, batches[i], cfg)
        else:
            c, llr = code.sample(batches[i], cfg["train_ebno"], punct_scale=hint)
        model.train()
        outs = model(llr, code.graph)
        loss, bce, syn = gnn_loss(outs, c, code.graph, cfg["lam_syn"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        elapsed = (time.time() - t0) / 60
        last = step == steps or elapsed > cfg["max_minutes"]
        if step % cfg["eval_every"] == 0 or last:
            model.eval()
            ber, bler = validate(model, codes, cfg["val_ebno"], cfg["val_frames"], max(batches))
            rec = dict(step=step, minutes=round(elapsed, 2), loss=loss.item(), bce=bce.item(),
                       syn=syn.item(), val_ber=ber, val_bler=bler, lr=lr)
            score = bler + ber  # BLER first, BER breaks ties at very low BLER
            msg = ""
            if hard_val is not None:
                score, fers = hard_val(model, max(batches))
                rec.update(hard_fer=fers, hard_score=score)
                msg = " | hard-set FER " + " ".join(f"{f:.3f}" for f in fers)
            if pools is not None:
                rec["pool_sizes"] = pools[0].sizes()
            history.append(rec)
            log(f"  step {step:6d} | {elapsed:6.1f} min | lr {lr:.1e} | loss {loss.item():.4f} "
                f"(bce {bce.item():.4f}, syn {syn.item():.4f}) | val BER {ber:.2e} BLER {bler:.2e}{msg}")
            if hint > 0:
                best, bad = (math.inf, None), 0  # model selection only on the final task
            elif score < best[0] - 1e-12:
                best, bad = (score, copy.deepcopy(model.state_dict())), 0
            else:
                bad += 1
            if plateau and bad >= cfg.get("patience", 5):
                lr *= 0.5
                bad = 0
                model.load_state_dict(best[1])
                log(f"  plateau: lr -> {lr:.1e} (restored best, score {best[0]:.4g})")
                if lr < cfg["lr_min"]:
                    log("  converged (lr below lr_min)")
                    last = True
        if last:
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return history
