"""Task 1: BP iteration schedules in the interleaved BP+SGT pipeline.

The final BP-SGT models were trained with 5 BP runs of 10 iterations (5x10) and
4 SGT stages with shared weights. Because the weights are shared across stages,
the same model can run with any number of BP iterations per run and any number of
runs, so this is an inference-only study (no retraining):

  same 4 SGT stages, fewer iterations per run   5x10 (as trained), 5x5, 5x2
  same 50-iteration budget, more SGT stages     10x5 (9 SGT stages), 25x2 (24)
  reference with more iterations                5x40 (as in the final comparison)
  optional non-uniform schedules                e.g. 20-10-10-5-5 or 2-5-10-10-23

against plain BP with the same total number of iterations (BP 10, 25, 50, 200).
Caveat to state with the results: the model never saw 5- or 2-iteration BP states
in training, so a schedule that loses to 5x10 may only need retraining
(--retrain is not implemented here on purpose; see task 5 for a trained variant).

Per decoder and Eb/N0 the trace records BLER, BER, average BP iterations, average
SGT calls and the stopping-point histograms, so tasks 2 and 4 read the same files.

Usage:
  python run_schedules.py [--codes a,b] [--schedules 5x10,5x5,5x2,10x5,25x2,5x40]
                          [--bp 10,25,50,200] [--quick]
  python plot_schedules.py --traces <out dir>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import runner  # noqa: E402

DEFAULT_SCHEDULES = "5x10,5x5,5x2,10x5,25x2,5x40"
DEFAULT_BP = "10,25,50,200"


def main():
    ap = runner.parser(__doc__)
    ap.add_argument("--schedules", default=DEFAULT_SCHEDULES)
    ap.add_argument("--bp", default=DEFAULT_BP, help="plain BP iteration counts ('' for none)")
    args = ap.parse_args()
    out, log, device = runner.setup("task1_bp_schedules", args)
    scheds = [core.parse_schedule(s) for s in args.schedules.split(",")]
    for name in args.codes.split(","):
        model, path, kw = core.load_sgt(name, device, [args.ckpt_root] if args.ckpt_root else [])
        log(f"{name}: checkpoint {path} (trained as {kw['stages'] + 1}x{kw['bp_iters']}, k_frac {kw['k_frac']})")
        entries = [(f"SGT {core.schedule_str(s)}", s, model) for s in scheds]
        entries += [(f"BP ({int(t)} it)", [int(t)], None) for t in args.bp.split(",") if t]
        runner.run_code(name, entries, args, out, log, device,
                        extra=dict(checkpoint=os.path.basename(path), model_kwargs=kw))
    log("done")


if __name__ == "__main__":
    main()
