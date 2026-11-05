"""Hyper-parameter sweep of the BP-SGT decoder on one code.

Every config overrides the defaults of sgt/specs.py (MODEL: k_frac, gt_layers,
d, attn, select, fix, base, ...; TRAIN: lr, schedule, ...). All configs are
trained with the BP-GNN recipe (hard-example mining on BP(10) failures, model
selection on a fixed set of BP failures) for ``--minutes`` each, then scored by
Monte Carlo BLER on the same frames (seeded) at the high-SNR points of the code:
    score = mean over Eb/N0 of BLER(config) / BLER(BP(50))      (lower is better)
Round 1 (ROUND1) varies one factor at a time around the base config; later
rounds read their grid from a JSON file ({name: overrides}).

Usage: python run_tuning.py --code mackay_204_102 --configs base,k05,... [--grid round2.json]
                            [--minutes 20] [--tag tune1] [--quick]
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import sgt  # noqa: E402,F401  (registers the "sgt" model kind)
from common.experiment import Logger, run_experiment  # noqa: E402
from common.report import write_report  # noqa: E402
from sgt.specs import load_any_code, sgt_spec  # noqa: E402

ROUND1 = {
    "base": {},
    "k05": dict(k_frac=0.05),
    "k20": dict(k_frac=0.2),
    "k30": dict(k_frac=0.3),
    "L3": dict(gt_layers=3),
    "d32": dict(d=32, hidden=64),
    "local": dict(attn="local"),
    "unsat": dict(select="unsat"),
    "replace": dict(fix="replace"),
    "llrbase": dict(base="llr"),
    "lr5e4": dict(lr=5e-4),
    # second batch of single factors (structure bias ablation, heads, k, schedule, mining, loss)
    "nobias": dict(struct_bias=False),
    "h8": dict(heads=8),
    "k15": dict(k_frac=0.15),
    "plateau": dict(schedule="plateau"),
    "hard50": dict(hard_frac=0.5),
    "syn0": dict(lam_syn=0.0),
}
# Eb/N0 points where BP(50) is in its high-SNR regime but still measurable in minutes
EBNO = {"mackay_204_102": [3.5, 4.0, 4.5], "5G_BG2_k64_n128": [3.5, 4.0, 4.5],
        "wimax_576_288": [2.5, 2.75, 3.0], "mackay_96_48": [4.5, 5.0, 5.5],
        "ccsds_128_64": [4.0, 4.5, 5.0], "mackay_408_204": [3.0, 3.25, 3.5],
        "5G_BG2_k128_n256": [3.0, 3.25, 3.5], "5G_BG1_k528_n704": [3.25, 3.5, 3.75]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated config names")
    ap.add_argument("--grid", default=None, help="JSON file {name: overrides} (default: ROUND1)")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--tag", default="tune1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"{args.tag}_{args.code}"
    out = args.out or (f"/kaggle/working/results/{name}" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "..", "results", name))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    grid = ROUND1
    if args.grid:
        with open(args.grid if os.path.isabs(args.grid) else os.path.join(HERE, args.grid)) as f:
            grid = json.load(f)
    code = load_any_code(args.code, device)
    log(f"{code.name}: {code.summary()}")
    names = args.configs.split(",")
    specs = []
    for n in names:
        ov = dict(grid[n])
        ov.setdefault("max_minutes", args.minutes)
        specs.append(sgt_spec(code, ov, q, tag=n, name=f"SGT {n}"))
        log(f"config {n}: {grid[n]} -> kwargs {specs[-1]['kwargs']}")
    ebno = EBNO[args.code]
    eval_cfg = dict(bp_iters=(50,), perturb=None, osd=False, batch=256 if q else 4096,
                    min_frame_errors=20 if q else 100, max_frames=1000,
                    max_seconds_per_point=5 if q else 120)
    spec = dict(code=code, ebno=ebno[:2] if q else ebno, prof_ebno=ebno[0],
                max_frames=2000 if q else 3_000_000, profile=False)
    res = run_experiment(name, [code], [spec], specs, eval_cfg, {}, out, device, log)
    write_report(res, out)

    rec = res["codes"][code.name]
    bp = {r["ebno"]: r for r in rec["ber_bler"]["BP (50 it)"]}
    rows = []
    for n, ms in zip(names, specs):
        rs = rec["ber_bler"][ms["label"]]
        ratios = [r["bler"] / bp[r["ebno"]]["bler"] for r in rs
                  if r["ebno"] in bp and bp[r["ebno"]]["frame_errors"] >= 20]
        info = res["models"][ms["label"]]
        hard = [h["hard_score"] for h in info["history"] if "hard_score" in h]
        rows.append(dict(config=n, overrides=grid[n], code=code.name,
                         score=sum(ratios) / len(ratios) if ratios else float("nan"),
                         bler={r["ebno"]: r["bler"] for r in rs},
                         frame_errors={r["ebno"]: r["frame_errors"] for r in rs},
                         best_hard_fer=min(hard) if hard else None, params=info["params"],
                         train_minutes=info["train_minutes"], steps=info["history"][-1]["step"],
                         macs=rec["macs"][ms["label"]]))
    rows.sort(key=lambda r: (r["score"] != r["score"], r["score"]))
    log("\nranking (mean BLER / BLER of BP(50); lower is better):")
    for r in rows:
        log(f"  {r['score']:6.3f}  {r['config']:10s} hard-set FER {r['best_hard_fer']}  {r['overrides']}")
    with open(os.path.join(out, "ranking.json"), "w") as f:
        json.dump(dict(code=code.name, bp50={e: r["bler"] for e, r in bp.items()}, rows=rows), f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
