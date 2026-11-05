"""Clean latency / throughput / energy profile of the trained final BP-SGT models.

The final runs used two processes per Kaggle session (one per T4). NVML indexes
GPUs physically and ignores CUDA_VISIBLE_DEVICES, so the second process read
the power of the first GPU (idle baselines of 10-66 W). This script reloads the
saved checkpoints and profiles them alone on GPU 0, exactly as run_experiment
does (same decoders, batch sizes, idle baseline, NVML energy, CPU latency),
with BP(50) re-profiled in the same session as a reference.

Usage: python profile_sgt.py --codes a,b [--ckpt-root /kaggle/input] [--quick]
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import sgt  # noqa: E402,F401
from common.evaluate import base_decoders  # noqa: E402
from common.experiment import Logger, batch_cap, profiling_decoders  # noqa: E402
from common.profiling import GPUPowerMeter, profile_decoders  # noqa: E402
from run_final import code_specs  # noqa: E402
from sgt.sgt_decoder import SparseGTBPDecoder  # noqa: E402
from sgt.specs import sgt_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--ckpt-root", default="/kaggle/input")
    ap.add_argument("--config", default="sgt_config.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or ("/kaggle/working/results/profile_sgt" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "..", "results", "profile_sgt"))
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    with open(os.path.join(HERE, args.config)) as f:
        cfg = json.load(f)
    name = cfg.get("name", "SGT")
    specs = {s[0].name: s for g in ("normal", "5g") for s in code_specs(g, device)}
    meter = GPUPowerMeter(0) if device.type == "cuda" else None
    log(f"device: {device} | GPU {meter and meter.name} | NVML {meter and meter.available} | "
        f"visible devices {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for cname in args.codes.split(","):
        code, _, prof_ebno, _ = specs[cname]
        ckpt = glob.glob(os.path.join(args.ckpt_root, "**", f"*_{cname}_{name.lower()}.pt"), recursive=True)
        assert ckpt, f"no checkpoint for {cname} under {args.ckpt_root}"
        state = torch.load(ckpt[0], map_location=device)
        kw = sgt_spec(code, cfg["overrides"], q)["kwargs"]
        assert kw == state["kwargs"], (kw, state["kwargs"])
        model = SparseGTBPDecoder(**kw).to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.label = f"{name} (5x10 BP it)"
        m40 = copy.deepcopy(model)
        m40.bp_iters, m40.iters, m40.label = 40, 5 * 40, f"{name} (5x40 BP it)"
        models = [model, m40]
        log(f"\n--- {cname}: {ckpt[0]} | profiling at Eb/N0 = {prof_ebno} dB")
        idle_w = None
        if meter is not None and meter.available:
            time.sleep(10)
            idle_w = meter.idle_w(5.0)
        log(f"  idle GPU {idle_w and round(idle_w, 1)} W")
        osd_kw = dict(osd_orders=(), osd_learned_orders=(2,), bp_osd_iters=50)
        pdecs = profiling_decoders(code, models, (50,), None, **osd_kw)
        cpu = None
        if device.type == "cuda":
            cpu_code = copy.deepcopy(code).to(torch.device("cpu"))
            cpu_models = [copy.deepcopy(m).cpu() for m in models]
            cpu = {f"{n} [fixed]": ((lambda fn: lambda llr: fn(llr, False)[0])(fn), cpu_code)
                   for n, (fn, _) in base_decoders(cpu_code.graph, cpu_models, (50,)).items()}
        cap = batch_cap(code, kw["hidden"])
        bss = sorted({min(b, cap) for b in ((1, 64) if q else (1, 64, 1024, 4096))})
        prof = profile_decoders(pdecs, code, prof_ebno, bss, min(64 if q else 1024, cap),
                                meter if (meter is not None and meter.available) else None, idle_w,
                                1.0 if q else 5.0, (1, 64), cpu, log)
        with open(os.path.join(out, f"profile_{cname}.json"), "w") as f:
            json.dump(dict(code=cname, prof_ebno=prof_ebno, idle_gpu_w=idle_w, gpu=meter and meter.name,
                           checkpoint=os.path.basename(ckpt[0]), profile=prof), f, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
