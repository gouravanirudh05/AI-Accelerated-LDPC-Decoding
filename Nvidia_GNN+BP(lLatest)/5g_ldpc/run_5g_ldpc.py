"""BP-GNN hybrid decoding (Gong, Cammerer & Renes 2023) of 5G NR LDPC codes vs.
pure GNN (Cammerer et al. 2022), BP, BP with random perturbation and OSD-0.

Protograph-aware models, one BP-GNN and one GNN per base graph. Each is trained on a single
short configuration and then evaluated, without retraining, on longer codes
with other lifting sizes Z (the GNN only knows base-graph positions, so it is
Z-agnostic):
  BG2, R=1/2: train (k=64, n=128) -> eval (64,128), (128,256), (256,512)
  BG1, R=3/4: train (k=528, n=704) -> eval (528,704), (1056,1408)
Usage: python run_5g_ldpc.py [--quick] [--exps bg2,bg1] [--models bpgnn,gnn] [--out DIR]
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from common.experiment import Logger, run_experiment  # noqa: E402
from common.report import write_report  # noqa: E402
from nr_ldpc import BG_SHAPE, NUM_BG_EDGES, nr_code  # noqa: E402

EXPERIMENTS = {
    "bg2": dict(
        train=[(64, 128)], train_ebno=(0.5, 5.0), val_ebno=3.0, batch=384,
        eval=[  # (k, n, Eb/N0 sweep, profiling Eb/N0, max frames)
            (64, 128, np.arange(0.5, 5.51, 0.5), 3.0, 300_000),
            (128, 256, np.arange(0.5, 5.01, 0.5), 2.5, 200_000),
            (256, 512, np.arange(0.5, 4.01, 0.5), 2.0, 150_000),
        ]),
    "bg1": dict(
        train=[(528, 704)], train_ebno=(2.0, 5.0), val_ebno=3.5, batch=192,
        eval=[
            (528, 704, np.arange(2.0, 5.01, 0.25), 3.5, 150_000),
            (1056, 1408, np.arange(2.0, 4.51, 0.25), 3.25, 60_000),
        ]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exps", default="bg2,bg1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default="bpgnn,gnn")
    ap.add_argument("--minutes", type=float, default=30.0, help="BP-GNN training budget per base graph")
    ap.add_argument("--gnn-minutes", type=float, default=30.0, help="GNN training budget per base graph")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or ("/kaggle/working/results/5g_ldpc" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "..", "results", "5g_ldpc"))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    q = args.quick
    eval_cfg = dict(bp_iters=(10, 50), perturb=(4, 10, 0.5), osd=True, batch=256 if q else 2048,
                    min_frame_errors=20 if q else 150, osd_max_frames=2000 if q else 60_000,
                    max_frames=1000, max_seconds_per_point=10 if q else 150)
    prof_cfg = dict(batch_sizes=(1, 64) if q else (1, 64, 1024, 4096), energy_batch=64 if q else 1024,
                    energy_seconds=1.0 if q else 5.0, cpu_batch_sizes=(1, 64))
    summary = []
    for exp in args.exps.split(","):
        E = EXPERIMENTS[exp]
        rows, cols = BG_SHAPE[exp]
        train_codes = [nr_code(k, n, bg=exp, device=device) for k, n in E["train"]]
        specs = []
        for k, n, ebno, prof_ebno, max_frames in (E["eval"][:2] if q else E["eval"]):
            c = nr_code(k, n, bg=exp, device=device)
            specs.append(dict(code=c, ebno=list(ebno[::3] if q else ebno), prof_ebno=prof_ebno,
                              max_frames=2000 if q else max_frames))
        for c in train_codes + [s["code"] for s in specs]:
            log(f"{c.name}: {c.summary()}")
        types = dict(num_vn_types=cols, num_cn_types=rows, num_edge_types=NUM_BG_EDGES[exp])
        base_train = dict(steps=40 if q else 60_000, batch=32 if q else E["batch"], lr_min=2e-5,
                          train_ebno=E["train_ebno"], lam_syn=0.1, eval_every=20 if q else 1000,
                          val_ebno=E["val_ebno"], val_frames=256 if q else 8192, grad_clip=1.0)
        model_specs = {
            "bpgnn": dict(kind="bpgnn", kwargs=dict(types, d=16, d_e=8, hidden=48, stages=4,
                                                    bp_iters=10, gnn_layers=2),
                          train=dict(base_train, lr=1e-3, max_minutes=0.5 if q else args.minutes,
                                     hard_frac=0.75, mining_factor=4, mining_bp_iters=10)),
            "gnn": dict(kind="gnn", kwargs=dict(types, d=16, d_e=8, hidden=48, iters=10,
                                                use_soft_syndrome=True),
                        train=dict(base_train, lr=2e-3, max_minutes=0.5 if q else args.gnn_minutes)),
        }
        res = run_experiment(f"5G_{exp.upper()}", train_codes, specs,
                             [model_specs[m] for m in args.models.split(",")], eval_cfg, prof_cfg,
                             out, device, log)
        md = write_report(res, out)
        log(md)
        summary.append(md)
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write("# BP-GNN / GNN decoding of 5G NR LDPC codes\n\n" + "\n\n".join(summary))
    log("done")


if __name__ == "__main__":
    main()
