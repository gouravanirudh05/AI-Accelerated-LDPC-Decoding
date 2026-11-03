"""BP-GNN hybrid decoding (Gong, Cammerer & Renes 2023) of standard classical
LDPC benchmark codes vs. pure GNN (Cammerer et al. 2022), BP, BP with random
perturbation restarts and OSD-0 post-processing.

Codes (the usual benchmarks of neural-decoder papers, e.g. Nachmani et al.,
ECCT, Cammerer et al.):
  MacKay (96,48) 96.33.964, CCSDS (128,64), MacKay (204,102) 204.33.484,
  MacKay (408,204) 408.33.844, IEEE 802.16e WiMAX (576,288) rate 1/2.
One BP-GNN and one GNN are trained per code. Usage:
  python run_normal_ldpc.py [--quick] [--codes mackay_96_48,...] [--models bpgnn,gnn] [--out DIR]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.code import Code  # noqa: E402
from common.experiment import Logger, run_experiment  # noqa: E402
from common.gf2 import parse_alist  # noqa: E402
from common.report import write_report  # noqa: E402

CODES = {
    # name: (Eb/N0 sweep, training Eb/N0 range, validation Eb/N0, profiling Eb/N0, train batch, max frames)
    "mackay_96_48": (np.arange(1.0, 6.51, 0.5), (1.0, 6.0), 4.0, 4.0, 512, 300_000),
    "ccsds_128_64": (np.arange(1.0, 6.01, 0.5), (1.0, 6.0), 4.0, 4.0, 512, 300_000),
    "mackay_204_102": (np.arange(1.0, 5.01, 0.5), (1.0, 5.0), 3.0, 3.0, 384, 200_000),
    "mackay_408_204": (np.arange(1.0, 4.51, 0.5), (1.0, 4.5), 2.75, 2.75, 256, 150_000),
    "wimax_576_288": (np.arange(1.0, 4.01, 0.25), (1.0, 4.0), 2.5, 2.5, 192, 150_000),
}


def load_code(name, device):
    H = parse_alist(os.path.join(HERE, "codes", f"{name}.alist"))
    return Code(name, H, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--codes", default=",".join(CODES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default="bpgnn,gnn")
    ap.add_argument("--minutes", type=float, default=20.0, help="BP-GNN training budget per code")
    ap.add_argument("--gnn-minutes", type=float, default=20.0, help="GNN training budget per code")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or ("/kaggle/working/results/normal_ldpc" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "..", "results", "normal_ldpc"))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    q = args.quick
    eval_cfg = dict(bp_iters=(10, 50), perturb=(4, 10, 0.5), osd=True, batch=256 if q else 2048,
                    min_frame_errors=20 if q else 150, osd_max_frames=2000 if q else 100_000,
                    max_frames=1000, max_seconds_per_point=10 if q else 150)
    prof_cfg = dict(batch_sizes=(1, 64) if q else (1, 64, 1024, 4096), energy_batch=64 if q else 1024,
                    energy_seconds=1.0 if q else 5.0, cpu_batch_sizes=(1, 64))
    summary = {}
    for name in args.codes.split(","):
        ebno, tr_rng, val_ebno, prof_ebno, batch, max_frames = CODES[name]
        code = load_code(name, device)
        log(f"\n##### {name}: {code.summary()}")
        H = code.H
        types = dict(num_vn_types=int(H.sum(0).max()) + 1, num_cn_types=int(H.sum(1).max()) + 1,
                     num_edge_types=1)
        base_train = dict(steps=40 if q else 40_000, batch=32 if q else batch, lr_min=2e-5,
                          train_ebno=tr_rng, lam_syn=0.1, eval_every=20 if q else 1000,
                          val_ebno=val_ebno, val_frames=256 if q else 8192, grad_clip=1.0)
        specs = {
            # the paper's method: BP(10) -> GNN -> BP(10) ... 4 GNN stages = 50 BP iterations
            "bpgnn": dict(kind="bpgnn", kwargs=dict(types, d=16, d_e=8, hidden=48, stages=4,
                                                    bp_iters=10, gnn_layers=2),
                          train=dict(base_train, lr=1e-3, max_minutes=0.5 if q else args.minutes,
                                     hard_frac=0.75, mining_factor=4, mining_bp_iters=10)),
            # predecessor: pure GNN decoder, 10 message-passing iterations
            "gnn": dict(kind="gnn", kwargs=dict(types, d=16, d_e=8, hidden=48, iters=10,
                                                use_soft_syndrome=True),
                        train=dict(base_train, lr=2e-3, max_minutes=0.5 if q else args.gnn_minutes)),
        }
        spec = dict(code=code, ebno=list(ebno[::3] if q else ebno), prof_ebno=prof_ebno,
                    max_frames=2000 if q else max_frames)
        res = run_experiment(name, [code], [spec], [specs[m] for m in args.models.split(",")],
                             eval_cfg, prof_cfg, out, device, log)
        summary[name] = write_report(res, out)
        log(summary[name])
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write("# BP-GNN / GNN decoding of standard LDPC codes\n\n" + "\n\n".join(summary.values()))
    log("done")


if __name__ == "__main__":
    main()
