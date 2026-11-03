"""Hyper-parameter sweep to pick the *best* version of each learned decoder before
the final comparison.

part normal: MacKay (204,102)   part 5g: 5G NR BG2 (k=64, n=128)
  BP-GNN  base   d=16, hidden 48, 2 GNN layers, mining on Eb/N0 band A
  BP-GNN  large  d=32, hidden 96, 3 GNN layers, band A
  BP-GNN  mid    base model, mining on the lower band B
  GNN     several variants (incl. fixes for the 5G puncturing failure)
  NBP     per-edge, 10 and 50 unrolled iterations
All: plateau LR schedule (train to convergence within the time budget);
BP-GNN models are selected on a fixed set of BP failures (val_mode="hard").
Scored by deep Monte Carlo BLER at high Eb/N0 against BP(50).
round 2 (after round 1 showed BP-GNN = BP(50) on 5G): hard examples are frames
BP(50) fails on instead of BP(10) (on 5G most BP(10) failures only converge
slowly, so mining them teaches the GNN nothing that more BP iterations cannot do),
with 10 or 20 BP iterations per stage; low-SNR band as the round-1 winner.
Usage: python run_tuning.py --part normal|5g [--round 1|2] [--quick] [--minutes 40]
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "5g_ldpc"))
from common.code import Code  # noqa: E402
from common.experiment import Logger, run_experiment  # noqa: E402
from common.gf2 import parse_alist  # noqa: E402
from common.report import write_report  # noqa: E402
from nr_ldpc import BG_SHAPE, NUM_BG_EDGES, nr_code  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["normal", "5g"], required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"tuning{'' if args.round == 1 else args.round}_{args.part}"
    out = args.out or (f"/kaggle/working/results/{name}" if os.path.isdir("/kaggle/working")
                       else os.path.join(ROOT, "results", name))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    if args.part == "normal":
        H = parse_alist(os.path.join(ROOT, "normal_ldpc", "codes", "mackay_204_102.alist"))
        code = Code("mackay_204_102", H, device=device)
        types = dict(num_vn_types=8, num_cn_types=8, num_edge_types=1)
        band_hi, band_lo, val_hard, val_ebno = (2.5, 5.5), (1.0, 4.0), (3.5, 5.0), 3.5
        ebno, batch = [3.0, 3.5, 4.0, 4.5, 5.0], 384
    else:
        code = nr_code(64, 128, bg="bg2", device=device)
        rows, cols = BG_SHAPE["bg2"]
        types = dict(num_vn_types=cols, num_cn_types=rows, num_edge_types=NUM_BG_EDGES["bg2"])
        band_hi, band_lo, val_hard, val_ebno = (2.5, 6.0), (0.5, 4.5), (3.5, 5.5), 3.5
        ebno, batch = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5], 384

    minutes = 0.5 if q else args.minutes
    base = dict(steps=60 if q else 10 ** 7, batch=32 if q else batch, lr_min=1e-5, train_ebno=band_hi,
                lam_syn=0.1, eval_every=20 if q else 500, val_ebno=val_ebno,
                val_frames=256 if q else 16384, grad_clip=1.0, max_minutes=minutes,
                schedule="plateau", patience=6)
    mining = dict(hard_frac=0.75, mining_bp_iters=10, pool_bins=4, pool_new=64 if not q else 8,
                  pool_init=1024 if not q else 16, pool_size=20000, pool_refresh_every=10,
                  mining_chunk=4096 if q else 32768, mining_max_frames=8192 if q else 262144,
                  val_mode="hard", val_hard_ebno=val_hard, val_hard_bins=3,
                  val_hard_frames=64 if q else 1000, val_hard_max_frames=8192 if q else 8_000_000)
    gkw = dict(types, d=16, d_e=8, hidden=48)
    gkw_l = dict(types, d=32, d_e=8, hidden=96)
    nkw = dict(num_edges=code.graph.num_edges, num_edge_types=types["num_edge_types"])
    specs = [
        dict(kind="bpgnn", tag="bpgnn_base", label="BP-GNN base (d16,h48,L2)",
             kwargs=dict(gkw, stages=4, bp_iters=10, gnn_layers=2), train=dict(base, lr=1e-3, **mining)),
        dict(kind="bpgnn", tag="bpgnn_large", label="BP-GNN large (d32,h96,L3)",
             kwargs=dict(gkw_l, stages=4, bp_iters=10, gnn_layers=3), train=dict(base, lr=7e-4, **mining)),
        dict(kind="bpgnn", tag="bpgnn_mid", label="BP-GNN base, low-SNR mining",
             kwargs=dict(gkw, stages=4, bp_iters=10, gnn_layers=2),
             train=dict(base, lr=1e-3, **dict(mining, train_ebno=band_lo))),
    ]
    if args.round == 2:
        m50 = dict(base, lr=1e-3, **dict(mining, train_ebno=band_lo, mining_bp_iters=50,
                                         mining_max_frames=8192 if q else 1_048_576))
        specs = [
            dict(kind="bpgnn", tag="bpgnn_m50", label="BP-GNN 5x10, BP(50)-failure mining",
                 kwargs=dict(gkw, stages=4, bp_iters=10, gnn_layers=2), train=m50),
            dict(kind="bpgnn", tag="bpgnn_m50_s20", label="BP-GNN 5x20, BP(50)-failure mining",
                 kwargs=dict(gkw, stages=4, bp_iters=20, gnn_layers=2), train=m50),
        ]
    elif args.part == "normal":
        specs += [
            dict(kind="gnn", tag="gnn_base", label="GNN base (d16,h48,10 it)",
                 kwargs=dict(gkw, iters=10, use_soft_syndrome=True),
                 train=dict(base, lr=2e-3, train_ebno=band_lo)),
            dict(kind="gnn", tag="gnn_large", label="GNN large (d32,h96,10 it)",
                 kwargs=dict(gkw_l, iters=10, use_soft_syndrome=True),
                 train=dict(base, lr=1e-3, train_ebno=band_lo)),
        ]
    else:
        specs += [
            dict(kind="gnn", tag="gnn_lowlr", label="GNN lr 5e-4",
                 kwargs=dict(gkw, iters=10, use_soft_syndrome=True),
                 train=dict(base, lr=5e-4, train_ebno=band_lo)),
            dict(kind="gnn", tag="gnn_nosyn", label="GNN no soft-syndrome",
                 kwargs=dict(gkw, iters=10, use_soft_syndrome=False),
                 train=dict(base, lr=1e-3, lam_syn=0.0, train_ebno=band_lo)),
            dict(kind="gnn", tag="gnn_large20", label="GNN large, 20 it",
                 kwargs=dict(gkw_l, iters=20, use_soft_syndrome=True),
                 train=dict(base, lr=5e-4, batch=32 if q else 192, train_ebno=band_lo)),
        ]
    specs += [] if args.round == 2 else [
        dict(kind="nbp", tag="nbp10", label="NBP per-edge (10 it)", kwargs=dict(nkw, iters=10, per_edge=True),
             train=dict(base, lr=5e-3, batch=32 if q else 1024, train_ebno=band_lo,
                        max_minutes=min(minutes, 15))),
        dict(kind="nbp", tag="nbp50", label="NBP per-edge (50 it)", kwargs=dict(nkw, iters=50, per_edge=True),
             train=dict(base, lr=3e-3, batch=32 if q else 1024, train_ebno=band_lo,
                        max_minutes=min(minutes, 20))),
    ]
    eval_cfg = dict(bp_iters=(50,) if args.round == 1 else (50, 200), perturb=None, osd=False, batch=256 if q else 4096,
                    min_frame_errors=20 if q else 100, max_frames=1000,
                    max_seconds_per_point=10 if q else 150)
    spec = dict(code=code, ebno=ebno[::2] if q else ebno, prof_ebno=val_ebno,
                max_frames=2000 if q else 3_000_000, profile=False)
    res = run_experiment(name, [code], [spec], specs, eval_cfg, {}, out, device, log)
    md = write_report(res, out)
    log(md)
    # compact ranking: BLER relative to BP(50) at the highest SNR points with >= 20 errors
    rows = []
    bp = {r["ebno"]: r for r in res["codes"][code.name]["ber_bler"]["BP (50 it)"]}
    for name, rs in res["codes"][code.name]["ber_bler"].items():
        ratios = [r["bler"] / bp[r["ebno"]]["bler"] for r in rs
                  if r["ebno"] in bp and bp[r["ebno"]]["frame_errors"] >= 20 and r["ebno"] >= ebno[2]]
        rows.append((name, sum(ratios) / len(ratios) if ratios else float("nan")))
    rows.sort(key=lambda x: x[1])
    log("\nranking (mean BLER / BLER of BP(50) at high Eb/N0, lower is better):")
    for n, r in rows:
        log(f"  {r:6.3f}  {n}")
    with open(os.path.join(out, "ranking.json"), "w") as f:
        json.dump(rows, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
