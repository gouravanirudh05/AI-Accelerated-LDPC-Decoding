"""CPU checks that the instrumented / variant decoders compute what the originals do.

  python tests/test_equivalence.py
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lat  # noqa: E402,F401
from common.bp import bp_decode, staged_decode  # noqa: E402
from lat import bp_inst, sgt_inst  # noqa: E402
from lat.bp_fast import AGGS, FastBP  # noqa: E402
from lat.codes import build_model, load  # noqa: E402
from lat.regions import Regions  # noqa: E402


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        check.failed = True


check.failed = False


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    for cname, ebno in [("mackay_204_102", 2.0), ("5G_BG2_k64_n128", 2.0)]:
        code = load(cname, dev)
        g = code.graph
        _, llr = code.sample(96, ebno)
        ref, ref_u = bp_decode(llr, g, 30)
        for mode in ("off", "sync", "profiler"):
            out, u = bp_inst.bp_decode(llr, g, 30, R=Regions(mode, dev))
            check(f"{cname} instrumented bp_decode [{mode}]", torch.equal(out, ref) and torch.equal(u, ref_u))
        # batch compaction changes tensor shapes, so CPU SIMD paths may differ in the last ulp
        for K in (1, 5, 7, 30):
            out, u = FastBP(g, 30, check_every=K)(llr)
            check(f"{cname} masked sync-free BP, host check every {K} == bp_decode "
                  f"(same iterations, same hard decisions, max |dLLR| {(out - ref).abs().max():.1e})",
                  torch.equal(u, ref_u) and torch.equal(out < 0, ref < 0) and (out - ref).abs().max() < 1e-3)
        for agg in AGGS:
            for fuse in (False, True):
                out, u = FastBP(g, 30, agg=agg, fuse_sign=fuse)(llr)
                same = ((out < 0) == (ref < 0)).float().mean().item()
                check(f"{cname} agg={agg} fuse_sign={fuse}: hard decisions agree {same:.4f}, "
                      f"max |dLLR| {(out - ref).abs().max():.2e}", same > 0.999)
        out, u = FastBP(g, 30, cn="nms")(llr)
        ref2, ref2u = bp_decode(llr, g, 30, mode="nms")
        check(f"{cname} masked NMS == bp_decode(nms)", torch.equal(out, ref2) and torch.equal(u, ref2u))

        # SGT: random non-zero weights so the bias and head actually matter
        model = build_model(code, dev, k_frac=0.5)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.3)
        model.eval()
        L, v2c, c2v, ok, _ = bp_inst.bp_stage(llr, g, 5, early_stop=False)
        with torch.no_grad():
            r0 = model.refine(llr, llr, L, v2c, c2v, g)
            for v in ("orig", "gather", "gather16"):
                r = sgt_inst.refine(model, llr, llr, L, v2c, c2v, g, Regions("sync", dev), v)
                tol = 0 if v == "orig" else 1e-5
                check(f"{cname} SGT refine [{v}] == model.refine (max diff {(r - r0).abs().max():.1e})",
                      (r - r0).abs().max().item() <= tol)
            ref_d, ref_du = model.decode(llr, g)
            d, du = bp_inst.staged_decode(llr, g, model.stages, model.bp_iters,
                                          sgt_inst.make_refine(model), True, Regions("sync", dev))
            check(f"{cname} instrumented staged SGT decode == model.decode",
                  torch.equal(d, ref_d) and torch.equal(du, ref_du))
            d, du = bp_inst.staged_decode(llr, g, model.stages, model.bp_iters,
                                          sgt_inst.make_refine(model, budget=20 * model.num_selected(g.n) ** 2 * 4),
                                          False)
            ref_d, ref_du = model.decode(llr, g, early_stop=False)
            agree = ((d < 0) == (ref_d < 0)).float().mean().item()
            check(f"{cname} chunked fixed SGT decode vs model.decode: hard decisions agree {agree:.4f}, "
                  f"iterations equal {torch.equal(du, ref_du)}", agree > 0.999 and torch.equal(du, ref_du))
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
