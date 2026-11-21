"""CPU equivalence tests of the fast decoders (seconds to a minute; no GPU needed).

1. StagedEngine with agg="index_add" and the *original* module refine gives the
   same hard decisions and iteration counts as the original decoders on every
   frame: BP(50) (common.bp.bp_stage, i.e. one stage), BP-DAT, BP-GNN and BP-SGT
   (trained checkpoints from git when found), with host checks every 1, 5 and
   10 iterations and with none, and with compaction at stage boundaries
   (power-of-two buckets, exact) and without. The LLRs agree up to float rounding: on CPU the
   original decoder itself is not batch-invariant (one frame decoded alone vs.
   in a batch of 64 differs by ~3e-3 after 50 iterations, because the batch
   compaction changes tensor sizes and with them the vectorised summation).
2. The padded layout (agg="padded") and the fast DAT refine agree with the
   originals up to fp32 summation order: the same frames converge, after the
   same number of iterations, with the same decisions, and the same frames are
   in error (frames BP never solves oscillate, so their wrong bits can differ).
3. Mixed precision (bf16 on CPU; fp16 on GPU in the benchmark) runs and its
   hard decisions are reported (not asserted: it is a different arithmetic).

The DAT model is untrained but its zero-initialised head is given random weights,
so the module really changes the BP prior (an untrained DAT is plain BP with
restarts, which would make the test trivial).

Usage: python FastDAT_BP/tests/test_equivalence.py [--codes 5G_BG1_k528_n704,...] [--frames 64]
"""

import argparse
import glob
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fastdat  # noqa: E402,F401
from fastdat.engine import StagedEngine  # noqa: E402
from fastdat.dat_fast import fast_dat_refine  # noqa: E402
from fastdat.layout import Layout  # noqa: E402
from fastdat.models import build_bpgnn, build_dat, build_sgt  # noqa: E402
from common.bp import bp_stage, staged_decode  # noqa: E402
from dat.specs import BANDS, load_any_code  # noqa: E402


def bp_ref(llr, g, iters):
    L, _, _, _, used = bp_stage(llr, g, iters)
    return L, used


def rel_diff(L1, L2):
    return float(((L1 - L2).abs() / L1.abs().clamp_min(1.0)).max())


def check_same(name, ref, got, tol=1e-2, quiet=True):
    (L1, u1), (L2, u2) = ref, got
    flips = int(((L1 < 0) != (L2 < 0)).any(1).sum())
    d = rel_diff(L1, L2)
    if not quiet:
        print(f"    {name:44s} max rel |dLLR| {d:.2e}  frames with other hard decision {flips}  "
              f"iteration counts equal {bool(torch.equal(u1, u2))}")
    assert flips == 0, f"{name}: {flips} frames decide differently"
    assert torch.equal(u1, u2), f"{name}: iteration counts differ"
    assert d < tol, f"{name}: LLR differs by {d:.3g} (relative)"
    return d


def check_close(name, ref, got, g, c):
    """Equivalent for FER: same set of converged frames, same iteration counts, same
    decisions on every converged frame and the same frames in error. Frames BP does
    not solve are chaotic (oscillating messages), so their bits may differ in the
    last digits' consequences; both outputs are frame errors."""
    (L1, u1), (L2, u2) = ref, got
    conv1 = (g.syndrome((L1 < 0).float()) == 0).all(1)
    conv2 = (g.syndrome((L2 < 0).float()) == 0).all(1)
    err1 = ((L1 < 0).float() != c).any(1)
    err2 = ((L2 < 0).float() != c).any(1)
    flips_conv = int(((L1 < 0) != (L2 < 0))[conv1].any(1).sum())
    flips_fail = int(((L1 < 0) != (L2 < 0))[~conv1].any(1).sum())
    d = rel_diff(L1[conv1], L2[conv1]) if conv1.any() else 0.0
    print(f"    {name:48s} converged {int(conv1.sum())}/{len(conv1)} | frame errors {int(err1.sum())} vs "
          f"{int(err2.sum())} | max rel |dLLR| on converged {d:.1e} | unsolved frames with other bits "
          f"{flips_fail}")
    assert torch.equal(conv1, conv2) and torch.equal(u1, u2), f"{name}: convergence / iterations differ"
    assert flips_conv == 0 and torch.equal(err1, err2), f"{name}: decisions or frame errors differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="5G_BG2_k64_n128,5G_BG1_k528_n704,5G_BG1_k1056_n1408")
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cpu")
    for name in args.codes.split(","):
        t0 = time.time()
        code = load_any_code(name, dev)
        g = code.graph
        # a mix of easy and hard frames: just below and at the validation Eb/N0
        band = BANDS[name]
        ca, a = code.sample(args.frames // 2, band["val_ebno"] - 1.5)
        cb, b = code.sample(args.frames - args.frames // 2, band["val_ebno"] - 0.5)
        llr, c = torch.cat([a, b]), torch.cat([ca, cb]).float()
        print(f"{name}: n={g.n} m={g.m} E={g.num_edges}")
        dat_m = build_dat(code, dev, randomise_head=True)
        models = dict(DAT=dat_m, **{"DAT-NB": build_dat(code, dev, randomise_head=True, cross="nb")})
        gnn = build_bpgnn(code, dev)
        if gnn is not None:
            models["BP-GNN"] = gnn
        sgt = build_sgt(code, dev)
        if sgt is not None:
            models["SGT"] = sgt
        print("  trained checkpoints:", {k: getattr(m, "trained_from", None) is not None for k, m in models.items()})

        # 1. same decisions with index_add and the original refine
        ref = bp_ref(llr, g, 50)
        worst = 0.0
        for ce in (1, 5, 10, 0):
            worst = max(worst, check_same(f"BP(50) check_every={ce}", ref,
                                          StagedEngine(g, 0, 50, agg="index_add", check_every=ce)(llr)))
        modes = [(1, "pow2"), (5, "pow2"), (5, "exact"), (10, "none"), (0, "none"), (0, "pow2")]
        for mname, m in models.items():
            refine = m.refine_chunked if hasattr(m, "refine_chunked") else m.refine
            ref = m.decode(llr, g)
            for ce, cp in modes:
                eng = StagedEngine(g, m.stages, m.bp_iters, refine, agg="index_add", check_every=ce,
                                   warm=getattr(m, "warm", False), compact=cp)
                worst = max(worst, check_same(f"{mname} check_every={ce} compact={cp}", ref, eng(llr)))
        print(f"  1. engine + original refine, agg=index_add: same decisions and iteration counts as the "
              f"original decoders (max rel |dLLR| {worst:.1e})")

        # 2. padded layout and fast DAT refine
        lay = Layout(g, "padded")
        lay_i = Layout(g, "index_add")
        check_close("BP(50) padded layout", bp_ref(llr, g, 50), StagedEngine(g, 0, 50, layout=lay)(llr), g, c)
        for mname in ("DAT", "DAT-NB"):
            m = models[mname]
            ref = m.decode(llr, g)
            # module alone, same inputs
            Lb, v2c, c2v, _, _ = bp_stage(llr, g, 10, early_stop=False)
            with torch.no_grad():
                p1 = m.refine(llr, llr, Lb, v2c, c2v, g)
                p2 = fast_dat_refine(m, lay)(llr, llr, Lb, v2c, c2v, g)
                p3 = fast_dat_refine(m, lay_i)(llr, llr, Lb, v2c, c2v, g)
            print(f"    {mname} module alone: max|dprior| padded {float((p1 - p2).abs().max()):.2e}, "
                  f"index_add {float((p1 - p3).abs().max()):.2e}")
            assert float((p1 - p2).abs().max()) < 1e-3
            check_close(f"{mname} fast refine, index_add engine",
                        ref, StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay_i), layout=lay_i)(llr), g, c)
            check_close(f"{mname} fast refine, padded engine, check_every=5",
                        ref, StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay), layout=lay)(llr), g, c)
            check_close(f"{mname} fast refine, padded engine, fixed schedule",
                        ref, StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay), layout=lay,
                                          check_every=0, compact="none")(llr), g, c)
            # 3. mixed precision (bf16 on CPU), reported only
            L16, u16 = StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay, torch.bfloat16), layout=lay)(llr)
            flips = int(((ref[0] < 0) != (L16 < 0)).any(1).sum())
            print(f"    {mname} bf16 module: frames with other hard decision {flips} / {llr.shape[0]} "
                  f"(not asserted)")
        print(f"  2. padded layout + fast DAT refine: same converged frames, iteration counts and frame errors "
              f"({time.time() - t0:.0f} s)")
    print("equivalence tests passed")


if __name__ == "__main__":
    main()
