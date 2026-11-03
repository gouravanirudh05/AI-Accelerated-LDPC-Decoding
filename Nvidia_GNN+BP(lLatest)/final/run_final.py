"""Final comparison, best version of every decoder on every code.

stage baselines: classical decoders only (independent of any training)
    BP(10), BP(50), BP(200), BP(50)+OSD-0, BP(50)+OSD-2, BP + random perturbation
stage learned:   tuned learned decoders (configs chosen by tuning/run_tuning.py)
Both stages: deep Monte Carlo (>= 100 frame errors or up to 1-3 M frames per point),
GPU / CPU latency, throughput, NVML power and energy per codeword.

Usage: python run_final.py --stage baselines|learned --group normal|5g [--quick]
"""

import argparse
import copy
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "5g_ldpc"))
sys.path.insert(0, os.path.join(ROOT, "normal_ldpc"))
from common.experiment import Logger, run_experiment  # noqa: E402
from common.report import write_report  # noqa: E402
from nr_ldpc import BG_SHAPE, NUM_BG_EDGES, nr_code  # noqa: E402
from run_5g_ldpc import EXPERIMENTS  # noqa: E402
from run_normal_ldpc import CODES, load_code  # noqa: E402


def code_specs(group, device):
    """[(code, Eb/N0 sweep, profiling Eb/N0, max frames)] for the group."""
    out = []
    if group == "normal":
        for name, (ebno, _, _, prof_ebno, _, _) in CODES.items():
            c = load_code(name, device)
            mf = 3_000_000 if c.n <= 204 else 2_000_000 if c.n <= 408 else 1_000_000
            out.append((c, ebno, prof_ebno, mf))
    else:
        for bg in ("bg2", "bg1"):
            for k, n, ebno, prof_ebno, _ in EXPERIMENTS[bg]["eval"]:
                c = nr_code(k, n, bg=bg, device=device)
                mf = 3_000_000 if n <= 128 else 2_000_000 if n <= 256 else 1_000_000 if n <= 704 else 500_000
                out.append((c, ebno, prof_ebno, mf))
    return out


def model_spec(t, code, band, quick):
    """Instantiate a learned-decoder template from learned_configs.json for one code."""
    t = copy.deepcopy(t)
    g = code.graph
    if "bg" in code.meta:
        rows, cols = BG_SHAPE[code.meta["bg"]]
        types = dict(num_vn_types=cols, num_cn_types=rows, num_edge_types=NUM_BG_EDGES[code.meta["bg"]])
    else:
        types = dict(num_vn_types=int(code.H.sum(0).max()) + 1, num_cn_types=int(code.H.sum(1).max()) + 1,
                     num_edge_types=1)
    if t["kind"] == "nbp":
        t["kwargs"].update(num_edges=g.num_edges, num_edge_types=types["num_edge_types"])
    else:
        t["kwargs"].update(types)
    tr = t["train"]
    # T4 (15 GB): cap batch x edges (BP-GNN / GNN ~12 GB per 1e6 at train time, NBP ~2.5 GB),
    # so the large BG1 graphs train with a smaller batch; codes up to WiMAX keep the tuned batch
    budget = 3e6 if t["kind"] == "nbp" else 6e5
    tr["batch"] = min(tr["batch"], max(32, int(budget // g.num_edges) // 32 * 32))
    lo, hi = band["train_ebno"]
    # "low": band shifted below the BP(10) waterfall (won the tuning sweep: full hard pools)
    shift = tr.pop("band_shift", None)
    train_ebno = [lo + shift[0], hi + shift[1]] if shift else [lo, hi]
    tr.update(train_ebno=train_ebno, val_ebno=band["val_ebno"])
    if tr.get("val_mode") == "hard":
        tr["val_hard_ebno"] = band["val_hard_ebno"]
    if quick:
        tr.update(max_minutes=0.5, steps=60, batch=32, eval_every=20, val_frames=256)
        for k, v in dict(pool_new=8, pool_init=16, mining_chunk=4096, mining_max_frames=8192,
                         val_hard_frames=64, val_hard_max_frames=8192).items():
            if k in tr:
                tr[k] = v
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baselines", "learned"], required=True)
    ap.add_argument("--group", choices=["normal", "5g"], required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--codes", default=None, help="comma-separated subset of code names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"final_{args.stage}_{args.group}" + (f"_{args.codes.replace(',', '+')}" if args.codes else "")
    out = args.out or (f"/kaggle/working/results/{tag}" if os.path.isdir("/kaggle/working")
                       else os.path.join(ROOT, "results", tag))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"device: {device} | torch {torch.__version__}"
        + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    prof_cfg = dict(batch_sizes=(1, 64) if q else (1, 64, 1024, 4096), energy_batch=64 if q else 1024,
                    energy_seconds=1.0 if q else 5.0, cpu_batch_sizes=(1, 64))
    specs = code_specs(args.group, device)
    if args.codes:
        keep = args.codes.split(",")
        specs = [s for s in specs if s[0].name in keep]
    if q:
        specs = specs[:2]
    summary = []
    if args.stage == "baselines":
        eval_cfg = dict(bp_iters=(10, 50, 200), bp_osd_iters=50, perturb=(4, 10, 0.5), osd=True,
                        osd_orders=(0, 2), osd_learned_orders=(), batch=256 if q else 4096,
                        min_frame_errors=20 if q else 100, osd_max_frames=2000 if q else 3_000_000,
                        max_frames=1000, max_seconds_per_point=10 if q else 300)
        for code, ebno, prof_ebno, mf in specs:
            spec = dict(code=code, ebno=list(ebno[::3] if q else ebno), prof_ebno=prof_ebno,
                        max_frames=2000 if q else mf, profile=True)
            res = run_experiment(f"{tag}_{code.name}", [code], [spec], [], eval_cfg, prof_cfg,
                                 out, device, log)
            summary.append(write_report(res, out))
    else:
        with open(os.path.join(HERE, "learned_configs.json")) as f:
            templates = json.load(f)[args.group]
        with open(os.path.join(HERE, "code_bands.json")) as f:
            bands = json.load(f)
        eval_cfg = dict(bp_iters=(50,), bp_osd_iters=50, perturb=None, osd=True, osd_orders=(),
                        osd_learned_orders=(2,), batch=256 if q else 4096,
                        min_frame_errors=20 if q else 100, osd_max_frames=2000 if q else 3_000_000,
                        max_frames=1000, max_seconds_per_point=10 if q else 240)
        for code, ebno, prof_ebno, mf in specs:
            model_specs = [model_spec(t, code, bands[code.name], q) for t in templates]
            spec = dict(code=code, ebno=list(ebno[::3] if q else ebno), prof_ebno=prof_ebno,
                        max_frames=2000 if q else mf, profile=True)
            # same batch for every decoder (comparable), power of two within the T4 memory
            e = code.graph.num_edges
            pow2 = lambda x: 1 << (max(1, int(x)).bit_length() - 1)
            res = run_experiment(f"{tag}_{code.name}", [code], [spec], model_specs, eval_cfg,
                                 dict(prof_cfg, memory=dict(ebno=bands[code.name]["val_ebno"],
                                                            train_batch=32 if q else min(256, pow2(6e5 / e)),
                                                            infer_batch=64 if q else min(4096, pow2(1e7 / e)),
                                                            max_batch=128 if q else 1 << 16)),
                                 out, device, log)
            summary.append(write_report(res, out))
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write(f"# {tag}\n\n" + "\n\n".join(summary))
    log("done")


if __name__ == "__main__":
    main()
