"""Trade-off study: unrolled neural BP vs. chained BP->GNN vs. end-to-end BP-GNN.

Tests three claims:
  1. GPU memory  - peak memory of a training step, largest trainable batch,
                   inference memory, for every decoder;
  2. adaptability - train on one code, evaluate unchanged on other codes
                   (classical: MacKay (3,6)-regular family; 5G: other lifting sizes);
  3. error floor - deep Monte Carlo (up to millions of frames per point) at high SNR.

Decoders:
  BP-GNN (end-to-end, the paper)  BP-GNN detached-BP ("chained", BP without autograd)
  GNN (pure, predecessor)         NBP per-edge (Nachmani-style, code specific)
  NBP shared (per-iteration / per-base-graph-edge weights, transferable)
  + BP(10), BP(50), BP(50)+OSD-0, BP + random perturbation.
Usage: python run_tradeoffs.py --part normal|5g [--quick] [--out DIR]
"""

import argparse
import os
import sys

import numpy as np
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


def mackay(name, device):
    H = parse_alist(os.path.join(ROOT, "normal_ldpc", "codes", f"{name}.alist"))
    return Code(name, H, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["normal", "5g"], required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--minutes", type=float, default=15.0, help="training budget per model")
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or (f"/kaggle/working/results/tradeoffs_{args.part}" if os.path.isdir("/kaggle/working")
                       else os.path.join(ROOT, "results", f"tradeoffs_{args.part}"))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    if args.part == "normal":
        exp = "tradeoffs_mackay"
        train = mackay("mackay_204_102", device)
        evals = [  # (code, Eb/N0 sweep, profiling Eb/N0, max frames, profile?)
            (train, np.arange(1.0, 5.51, 0.5), 3.0, 3_000_000, True),
            (mackay("mackay_96_48", device), np.arange(1.0, 7.01, 0.5), 4.0, 3_000_000, False),
            (mackay("mackay_408_204", device), np.arange(1.0, 4.51, 0.5), 2.75, 2_000_000, False),
        ]
        types = dict(num_vn_types=8, num_cn_types=8, num_edge_types=1)  # shared by the (3,6) family
        train_ebno, val_ebno, batch = (1.0, 5.0), 3.0, 384
    else:
        exp = "tradeoffs_5G_BG2"
        train = nr_code(64, 128, bg="bg2", device=device)
        evals = [
            (train, np.arange(0.5, 6.01, 0.5), 3.0, 3_000_000, True),
            (nr_code(128, 256, bg="bg2", device=device), np.arange(0.5, 5.01, 0.5), 2.5, 2_000_000, False),
            (nr_code(256, 512, bg="bg2", device=device), np.arange(0.5, 4.51, 0.5), 2.0, 1_000_000, False),
        ]
        rows, cols = BG_SHAPE["bg2"]
        types = dict(num_vn_types=cols, num_cn_types=rows, num_edge_types=NUM_BG_EDGES["bg2"])
        train_ebno, val_ebno, batch = (0.5, 5.0), 3.0, 384

    minutes = 0.4 if q else args.minutes
    base = dict(steps=40 if q else 60_000, batch=32 if q else batch, lr_min=2e-5, train_ebno=train_ebno,
                lam_syn=0.1, eval_every=20 if q else 1000, val_ebno=val_ebno,
                val_frames=256 if q else 8192, grad_clip=1.0, max_minutes=minutes)
    mine = dict(hard_frac=0.75, mining_factor=4, mining_bp_iters=10)
    gnn_kw = dict(types, d=16, d_e=8, hidden=48)
    nbp_kw = dict(num_edges=train.graph.num_edges, num_edge_types=types["num_edge_types"], iters=10)
    model_specs = [
        dict(kind="bpgnn", tag="bpgnn", kwargs=dict(gnn_kw, stages=4, bp_iters=10, gnn_layers=2),
             train=dict(base, lr=1e-3, **mine)),
        dict(kind="bpgnn", tag="bpgnn_detached", kwargs=dict(gnn_kw, stages=4, bp_iters=10, gnn_layers=2,
                                                             detach_bp=True),
             train=dict(base, lr=1e-3, **mine)),
        dict(kind="gnn", tag="gnn", kwargs=dict(gnn_kw, iters=10, use_soft_syndrome=True),
             train=dict(base, lr=2e-3)),
        dict(kind="nbp", tag="nbp_edge", kwargs=dict(nbp_kw, per_edge=True),
             train=dict(base, lr=5e-3, batch=32 if q else 1024)),
        dict(kind="nbp", tag="nbp_shared", kwargs=dict(nbp_kw, per_edge=False),
             train=dict(base, lr=5e-3, batch=32 if q else 1024)),
    ]
    specs = [dict(code=c, ebno=list(e[::3] if q else e), prof_ebno=pe, max_frames=2000 if q else mf,
                  profile=p) for c, e, pe, mf, p in evals]
    for s in specs:
        log(f"{s['code'].name}: {s['code'].summary()}")
    eval_cfg = dict(bp_iters=(10, 50), perturb=(4, 10, 0.5), osd=True, osd_learned=False,
                    batch=256 if q else 4096, min_frame_errors=20 if q else 100,
                    osd_max_frames=2000 if q else 500_000, max_frames=1000,
                    max_seconds_per_point=10 if q else 300)
    prof_cfg = dict(batch_sizes=(1, 64) if q else (1, 64, 1024, 4096), energy_batch=64 if q else 1024,
                    energy_seconds=1.0 if q else 4.0, cpu_batch_sizes=(1, 64),
                    memory=dict(ebno=val_ebno, train_batch=32 if q else 256,
                                infer_batch=64 if q else 4096, max_batch=128 if q else 1 << 16))
    res = run_experiment(exp, [train], specs, model_specs, eval_cfg, prof_cfg, out, device, log)
    md = write_report(res, out)
    log(md)
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write("# Trade-off study\n\n" + md)
    log("done")


if __name__ == "__main__":
    main()
