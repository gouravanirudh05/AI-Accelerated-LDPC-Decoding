"""Final run of BP-DAT on every code of the BP-GNN / BP-SGT comparison.

Same protocol as SparseGT_BP/final/run_sgt_final.py and the learned stage of
Nvidia_GNN+BP(lLatest)/final/run_final.py: one model per code, BP-GNN training
recipe (80 min, 160 min on codes with n >= 512, as BP-PGNN), deep Monte
Carlo (>= 100 frame errors or up to 0.5-3 M frames per point) of DAT 5x10 and
the 5x40 inference variant (+ OSD-2), BP(50) as a same-session reference, GPU
memory, latency, throughput and NVML energy. final/merge_dat.py puts the
results next to BP, BP-GNN, BP-SGT and BP-PGNN.

Usage: python run_dat_final.py --group normal|5g [--codes a,b] [--config dat_config.json] [--quick]
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import dat  # noqa: E402,F401
from common.experiment import Logger, run_experiment  # noqa: E402
from common.report import write_report  # noqa: E402
from run_final import code_specs  # noqa: E402  (BP-GNN project: codes, sweeps, frame budgets)
from dat.specs import BANDS, dat_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["normal", "5g"], required=True)
    ap.add_argument("--codes", default=None)
    ap.add_argument("--config", default="dat_config.json", help="file in final/ (or a path)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-profile", action="store_true", help="BER/BLER only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config if os.path.isabs(args.config) else os.path.join(HERE, args.config)) as f:
        cfg = json.load(f)
    name = cfg.get("name", "DAT")  # decoder label prefix, e.g. "DAT" or "DAT-NB"
    tag = (f"final_dat_{name.lower().replace('-', '_')}_{args.group}"
           + (f"_{args.codes.replace(',', '+')}" if args.codes else ""))
    out = args.out or (f"/kaggle/working/results/{tag}" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "..", "results", tag))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    log(f"config: {cfg}")
    specs = code_specs(args.group, device)
    if args.codes:
        specs = [s for s in specs if s[0].name in args.codes.split(",")]
    if q:
        specs = specs[:1]
    prof_cfg = dict(batch_sizes=(1, 64) if q else (1, 64, 1024, 4096), energy_batch=64 if q else 1024,
                    energy_seconds=1.0 if q else 5.0, cpu_batch_sizes=(1, 64))
    eval_cfg = dict(bp_iters=(50,), bp_osd_iters=50, perturb=None, osd=True, osd_orders=(),
                    osd_learned_orders=(2,), batch=256 if q else 4096,
                    min_frame_errors=20 if q else 100, osd_max_frames=2000 if q else 3_000_000,
                    max_frames=1000, max_seconds_per_point=10 if q else 240)
    summary = []
    for code, ebno, prof_ebno, mf in specs:
        ms = dat_spec(code, cfg["overrides"], q, tag=name.lower(), name=f"{name} (5x10 BP it)")
        ms["inference_variants"] = [dict(label=f"{name} (5x40 BP it)", bp_iters=40)]
        log(f"{code.name}: model {ms['kwargs']} | train {ms['train']['max_minutes']} min, "
            f"batch {ms['train']['batch']}")
        spec = dict(code=code, ebno=list(ebno[::3] if q else ebno), prof_ebno=prof_ebno,
                    max_frames=2000 if q else mf, profile=not args.no_profile)
        e = code.graph.num_edges
        pow2 = lambda x: 1 << (max(1, int(x)).bit_length() - 1)
        mem = dict(ebno=BANDS[code.name]["val_ebno"],
                   train_batch=32 if q else min(256, pow2(6e5 / e), pow2(ms["train"]["batch"])),
                   infer_batch=64 if q else min(4096, pow2(1e7 / e)), max_batch=128 if q else 1 << 16)
        res = run_experiment(f"{tag}_{code.name}", [code], [spec], [ms], eval_cfg,
                             dict(prof_cfg, memory=mem) if not args.no_profile else {}, out, device, log)
        summary.append(write_report(res, out))
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write(f"# {tag}\n\n" + "\n\n".join(summary))
    log("done")


if __name__ == "__main__":
    main()
