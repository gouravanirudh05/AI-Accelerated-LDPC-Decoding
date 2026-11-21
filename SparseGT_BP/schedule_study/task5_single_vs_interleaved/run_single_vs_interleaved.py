"""Task 5: single-stage BP+SGT against interleaved BP+SGT at the same BP budget.

  interleaved   BP(10) -> SGT -> BP(10) -> SGT -> BP(10) -> SGT -> BP(10) -> SGT -> BP(10)   (5x10, 4 SGT stages)
  single-stage  BP(25) -> SGT -> BP(25)                                                    (2x25, 1 SGT stage)

both with 50 BP iterations in total (and 5x40 vs 2x100 for the 200-iteration
budget), plus plain BP(50) / BP(200).

Two ways to get the single-stage decoder:
  infer (always)      the final interleaved weights run with one SGT stage
                      (labels "... interleaved weights"). Free, but the SGT never saw a
                      BP(25) state in training, so it is a lower bound for single-stage.
  --train-minutes M   additionally trains a dedicated single-stage model (stages=1,
                      bp_iters=25, hard-example mining on BP(25) failures, every other
                      setting = final BP-SGT config: k_frac 0.5, 2 layers, d 64, ...)
                      for M minutes with the BP-SGT training recipe (common/train.py),
                      saves it, and evaluates it as 2x25 and 2x100 ("SGT-1 ... trained").
                      Use M = 80 for a like-for-like comparison with the final models.

Writes traces_<code>.json like task 1, so task 2 / task 4 scripts also work on it.
Report: python compare.py --traces <out dir>

Usage: python run_single_vs_interleaved.py [--codes a,b] [--train-minutes 80] [--quick]
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import runner  # noqa: E402
from common.train import train_gnn  # noqa: E402
from sgt.sgt_decoder import SparseGTBPDecoder  # noqa: E402
from sgt.specs import sgt_spec  # noqa: E402

FINAL_OVERRIDES = dict(k_frac=0.5)  # final/sgt_config.json (all other values = sgt/specs.py MODEL)


def train_single(code, minutes, first_iters, quick, out, log, device):
    ov = dict(FINAL_OVERRIDES, stages=1, bp_iters=first_iters, mining_bp_iters=first_iters, max_minutes=minutes)
    spec = sgt_spec(code, ov, quick, tag="sgt1", name=f"SGT-1 (2x{first_iters} BP it)")
    torch.manual_seed(0)
    model = SparseGTBPDecoder(**spec["kwargs"]).to(device)
    log(f"training single-stage model on {code.name}: kwargs {spec['kwargs']}\n  train {spec['train']}")
    hist = train_gnn(model, [code], spec["train"], log)
    path = os.path.join(out, f"single_stage_{code.name}_sgt1.pt")
    torch.save(dict(state_dict=model.state_dict(), kind="sgt", kwargs=spec["kwargs"], history=hist), path)
    log(f"saved {path}")
    model.eval()
    return model


def main():
    ap = runner.parser(__doc__)
    ap.add_argument("--train-minutes", type=float, default=0.0, help="0 = inference-only comparison")
    ap.add_argument("--single", default="25-25", help="single-stage schedule at the 50-iteration budget")
    ap.add_argument("--single-200", default="100-100", help="single-stage schedule at the 200-iteration budget")
    args = ap.parse_args()
    out, log, device = runner.setup("task5_single_vs_interleaved", args)
    s50, s200 = core.parse_schedule(args.single), core.parse_schedule(args.single_200)
    assert len(s50) == 2 and len(s200) == 2, "single-stage schedules have exactly two BP runs"
    for name in args.codes.split(","):
        model, path, kw = core.load_sgt(name, device, [args.ckpt_root] if args.ckpt_root else [])
        log(f"{name}: interleaved checkpoint {path}")
        entries = [
            ("SGT 5x10 (interleaved)", [10] * 5, model),
            (f"SGT {core.schedule_str(s50)} (single-stage, interleaved weights)", s50, model),
            ("BP (50 it)", [50], None),
            ("SGT 5x40 (interleaved)", [40] * 5, model),
            (f"SGT {core.schedule_str(s200)} (single-stage, interleaved weights)", s200, model),
            ("BP (200 it)", [200], None),
        ]
        if args.train_minutes > 0:
            code = core.load_any_code(name, device)
            m1 = train_single(code, args.train_minutes, s50[0], args.quick, out, log, device)
            m1b = copy.deepcopy(m1)
            entries.insert(2, (f"SGT-1 {core.schedule_str(s50)} (single-stage, trained)", s50, m1))
            entries.insert(6, (f"SGT-1 {core.schedule_str(s200)} (single-stage, trained)", s200, m1b))
        runner.run_code(name, entries, args, out, log, device,
                        extra=dict(checkpoint=os.path.basename(path), model_kwargs=kw,
                                   train_minutes=args.train_minutes))
    log("done")


if __name__ == "__main__":
    main()
