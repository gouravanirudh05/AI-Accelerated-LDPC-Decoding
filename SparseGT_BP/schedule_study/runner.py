"""Common command line and run loop for the Monte Carlo scripts of the schedule study.

A run evaluates a list of decoders on one code and writes ``traces_<code>.json``:

    {"code": {...}, "ebno": [...], "checkpoint": "...", "model_kwargs": {...},
     "decoders": {label: {"kind": "sgt"|"bp", "schedule": [...], "budget": int,
                          "points": [simulate_traced record per Eb/N0]}}}

Tasks 2 and 4 (and the plots of tasks 1 and 5) only read these files, so they run
anywhere with numpy + matplotlib, without torch or a GPU.
"""

import argparse
import json
import os

import torch

import core


def add_common_args(ap):
    ap.add_argument("--codes", default=",".join(core.ALL_CODES), help="comma-separated code names")
    ap.add_argument("--ebno", default=None, help="comma-separated Eb/N0 [dB] (default: core.default_ebno)")
    ap.add_argument("--ckpt-root", default=None, help="extra folder searched for *_<code>_sgt.pt")
    ap.add_argument("--min-errors", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=90.0, help="per decoder and Eb/N0 point")
    ap.add_argument("--max-frames", type=int, default=None, help="default: core.frame_budget(n)")
    ap.add_argument("--batch", type=int, default=None, help="default 4096 (n <= 576) or 1024")
    ap.add_argument("--quick", action="store_true", help="seconds-long CPU smoke test")
    ap.add_argument("--out", default=None)
    return ap


def quick_overrides(args):
    if args.quick:
        args.min_errors, args.max_seconds, args.max_frames, args.batch = 10, 2.0, 512, 128


def run_code(code_name, entries, args, out, log, device, extra=None):
    """entries: [(label, schedule list, model or None)]. Returns the trace dict."""
    code = core.load_any_code(code_name, device)
    ebno = [float(e) for e in args.ebno.split(",")] if args.ebno else core.default_ebno(code_name)
    if args.quick:
        ebno = ebno[-2:]
    max_frames = args.max_frames or core.frame_budget(code.n)
    # long codes: smaller batches so the per-point time cap is respected (SGT k = n/2 is O(k^2))
    batch = args.batch or (4096 if code.n <= 576 else 1024)
    log(f"\n=== {code_name} {code.summary()} | Eb/N0 {ebno} | <= {max_frames} frames, >= {args.min_errors} "
        f"errors, <= {args.max_seconds} s per point ===")
    rec = dict(code=code.summary(), ebno=ebno, decoders={}, **(extra or {}))
    for label, sched, model in entries:
        dec = core.make_decoder(sched, model)
        pts = core.simulate_traced(dec, code, sched if model is not None else [sum(sched)], ebno,
                                   batch, max_frames, args.min_errors, args.max_seconds,
                                   seed=0, log=log, name=label)
        rec["decoders"][label] = dict(kind="sgt" if model is not None else "bp",
                                      schedule=sched if model is not None else [sum(sched)],
                                      budget=sum(sched), points=pts)
        with open(os.path.join(out, f"traces_{code_name}.json"), "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def setup(task, args):
    quick_overrides(args)
    out = core.results_dir(task, args.out)
    log = core.Logger(os.path.join(out, "log.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(core.device_info(device))
    log(f"args: {vars(args)}")
    return out, log, device


def parser(doc):
    return add_common_args(argparse.ArgumentParser(description=doc,
                                                   formatter_class=argparse.RawDescriptionHelpFormatter))
