"""Step 4: learned message-passing GNN decoders vs sum-product BP on the GPU.

For one code and every T in ``--T``:

1. **Train** the learned variants (``noms``: 2T shared scalars; ``wms``: a
   weight per edge per iteration) on the all-zero codeword with the
   multi-iteration BCE loss; checkpoint chosen on a fixed validation set.
2. **Evaluate** every decoder on the *same* frames (paired): sum-product
   (``spa``, the accuracy reference), min-sum (``ms``), normalised min-sum
   (``nms``), the learned variants, and fp16 versions of the min-sum family.
   Outputs use syndrome early-stopping semantics (first valid iteration).
3. **Measure on the GPU**, per decoder and execution mode -- eager fp32,
   CUDA graph fp32, CUDA graph fp16 (min-sum family only):
   latency at batch 1, throughput at a large batch, and energy per frame
   from the NVML energy counter (total and above idle).
4. **Equal FER.** For each sum-product reference SPA(T_ref), the cheapest
   decoder/mode whose FER at the operating SNR is no worse, with its latency
   and energy saving. Savings are split into *engineering* (SPA eager ->
   SPA graph) and *algorithm* (SPA graph -> learned decoder graph).

Full runs are GPU-only; ``--smoke`` is a seconds-long CPU pipeline check.

    python -m sparse_gt.experiments.step4 --code WIMAX_N576_K288 --snr 1.5 3.0 --T 5 10 20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..channel import awgn_llr
from ..codes import load_code
from ..energy import cuda_graph, idle_power_w, measure_energy, nvml_available
from ..mp_decoder import MessagePassingDecoder, multiloss
from ..simulate import fer_interval
from ..timing import gpu_time

LEARNED = ("noms", "wms")
LR = {"noms": 1e-2, "wms": 3e-3}


def log(msg: str) -> None:
    print(msg, flush=True)


def name_of(variant: str, T: int, fp16: bool = False) -> str:
    return f"{variant.upper()}({T})" + (" fp16" if fp16 else "")


# ------------------------------------------------------------------- train

@torch.no_grad()
def val_fer(dec, val_llr) -> float:
    errs = frames = 0
    for llr in val_llr:
        errs += int((dec.decode(llr) < 0).any(1).sum())
        frames += llr.shape[0]
    return errs / frames


def train(code, variant: str, T: int, args, device: str):
    torch.manual_seed(args.seed)
    dec = MessagePassingDecoder(code, variant, T).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    vgen = torch.Generator(device=device).manual_seed(args.seed + 1)
    lo, hi = args.snr
    val_llr = [awgn_llr(args.val_batch, code.n, float(s), code.rate, device, vgen)
               for s in np.linspace(lo, hi, 4)]
    opt = torch.optim.Adam(dec.parameters(), lr=LR[variant])
    best = (val_fer(dec, val_llr), copy.deepcopy(dec.state_dict()))
    log(f"[train {name_of(variant, T)}] {dec.num_parameters} params, start val FER {best[0]:.4e} (= NMS)")
    start, step, history = time.time(), 0, []
    budget = args.train_minutes * 60
    batch = max(64, min(args.batch, int(args.max_trace_elems // (T * code.n))))
    while time.time() - start < budget:
        frac = (time.time() - start) / budget
        for g in opt.param_groups:
            g["lr"] = LR[variant] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        snr = lo + (hi - lo) * torch.rand(batch, device=device, generator=gen)
        loss = multiloss(dec(awgn_llr(batch, code.n, snr, code.rate, device, gen)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % args.eval_every == 0:
            fer = val_fer(dec, val_llr)
            history.append({"step": step, "minutes": (time.time() - start) / 60,
                            "loss": float(loss), "val_fer": fer})
            if fer < best[0]:
                best = (fer, copy.deepcopy(dec.state_dict()))
            log(f"[train {name_of(variant, T)}] step {step:5d} {history[-1]['minutes']:4.1f} min "
                f"loss {float(loss):.4f} val FER {fer:.4e} best {best[0]:.4e}")
    dec.load_state_dict(best[1])
    dec.eval()
    summary = {}
    if variant == "noms":
        summary = {"beta": dec.beta.tolist(), "alpha": dec.alpha.tolist()}
    elif variant == "wms":
        w = dec.edge_weight.detach()
        summary = {"weight_mean_per_iter": w.mean(1).tolist(), "weight_std_per_iter": w.std(1).tolist(),
                   "alpha": dec.alpha.tolist()}
    log(f"[train {name_of(variant, T)}] done {step} steps, best val FER {best[0]:.4e}")
    return dec, {"history": history, "best_val_fer": best[0], "steps": step, "params": summary}


# -------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(code, decoders: dict, args, device: str, stop_on: list[str]) -> list[dict]:
    T_max = max(d.T for d, _ in decoders.values())
    batch = max(64, min(args.eval_batch, int(args.max_trace_elems // (T_max * code.n))))
    rows = []
    for snr in np.round(np.arange(args.snr[0], args.snr[1] + 1e-9, args.eval_step), 3).tolist():
        gen = torch.Generator(device=device).manual_seed(20_000 + int(round(snr * 100)))
        fe = dict.fromkeys(decoders, 0)
        be = dict.fromkeys(decoders, 0)
        frames, t0 = 0, time.time()
        while frames < args.eval_max_frames and min(fe[n] for n in stop_on) < args.eval_min_fe:
            llr = awgn_llr(batch, code.n, snr, code.rate, device, gen)
            for name, (dec, fp16) in decoders.items():
                out = dec.decode(llr.half() if fp16 else llr)
                errors = (out < 0).sum(1)
                fe[name] += int((errors > 0).sum())
                be[name] += int(errors.sum())
            frames += batch
        rows.append({"ebn0": snr, "frames": frames, "seconds": time.time() - t0,
                     "frame_errors": fe, "fer": {n: fe[n] / frames for n in decoders},
                     "ber": {n: be[n] / (frames * code.n) for n in decoders}})
        log(f"[eval] {snr:.2f} dB, {frames} frames ({rows[-1]['seconds']:.0f}s): " +
            "  ".join(f"{n} {fe[n] / frames:.2e}" for n in decoders))
    return rows


# --------------------------------------------------------- timing & energy

@torch.no_grad()
def measure(code, decoders: dict, args, device: str) -> dict:
    cuda = device.startswith("cuda")
    big = max(64, min(args.timing_batch, int(args.max_trace_elems // code.n)))
    gen = torch.Generator(device=device).manual_seed(5)
    x1 = awgn_llr(1, code.n, float(np.mean(args.snr)), code.rate, device, gen)
    xb = awgn_llr(big, code.n, float(np.mean(args.snr)), code.rate, device, gen)
    idle = idle_power_w() if cuda and nvml_available() else None
    out = {"batch": big, "idle_power_w": idle, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
           "nvml": nvml_available(), "rows": []}
    reps = 50 if not args.smoke else 2
    for name, (dec, fp16) in decoders.items():
        if fp16:  # fp16 copies are timed as the "graph fp16" mode of their fp32 decoder
            continue
        modes = [("eager", False)]
        if cuda:
            modes.append(("graph", False))
            if dec.variant != "spa":  # sum-product's log/exp stays in fp32
                modes.append(("graph", True))
        for mode, half in modes:
            d = copy.deepcopy(dec).half() if half else dec
            fwd = lambda x, d=d: d(x, return_trace=False)
            row = {"decoder": name_of(dec.variant, dec.T, half), "variant": dec.variant, "T": dec.T,
                   "mode": mode, "fp16": half}
            for label, x in (("b1", x1), ("big", xb)):
                xin = x.half() if half else x
                fn = cuda_graph(fwd, xin) if mode == "graph" else (lambda xin=xin, fwd=fwd: fwd(xin))
                t = gpu_time(fn, device, repeats=reps if label == "b1" else max(5, reps // 5))
                row[f"{label}_ms"] = t["ms"]
                row[f"{label}_p90_ms"] = t["p90"]
                if label == "big":
                    row["frames_per_s"] = big / t["ms"] * 1e3
                    e = measure_energy(fn, min_seconds=args.energy_seconds, idle_w=idle) if cuda else None
                    if e:
                        row["uj_per_frame"] = e["joules_per_call"] / big * 1e6
                        row["dynamic_uj_per_frame"] = e.get("dynamic_joules_per_call", float("nan")) / big * 1e6
                        row["mean_power_w"] = e["mean_power_w"]
                        row["energy_source"] = e["source"]
            out["rows"].append(row)
            log(f"[gpu] {row['decoder']:16s} {mode:5s}  b1 {row['b1_ms']:7.3f} ms   "
                f"b{big} {row['big_ms']:8.2f} ms ({row['frames_per_s']:10.0f} fr/s)"
                + (f"  {row['uj_per_frame']:8.2f} uJ/frame ({row['mean_power_w']:.0f} W)"
                   if "uj_per_frame" in row else ""))
    return out


# ----------------------------------------------------------------- summary

def equal_fer(results: dict, T_list: list[int], min_ref_errors: int = 50) -> list[dict]:
    """Cheapest decoder/mode matching each SPA reference FER, at every SNR point
    where the reference has at least ``min_ref_errors`` frame errors.

    References are SPA(T) for T in ``T_list`` and every extra SPA (e.g. SPA(50)).
    "Matching" = FER not above the reference's 95% upper bound.
    """
    rows = {(r["decoder"], r["mode"]): r for r in results["gpu"]["rows"]}
    spa_refs = sorted({r["T"] for r in results["gpu"]["rows"] if r["variant"] == "spa"})
    out = []
    for op in results["eval"]:
        for T_ref in spa_refs:
            ref = name_of("spa", T_ref)
            if op["frame_errors"].get(ref, 0) < min_ref_errors:
                continue
            _, hi_ci = fer_interval(op["frame_errors"][ref], op["frames"])
            cands = [(r.get("uj_per_frame", r["big_ms"]), r["b1_ms"], dname, mode, op["fer"][dname], r)
                     for (dname, mode), r in rows.items()
                     if dname in op["fer"] and op["fer"][dname] <= hi_ci]
            best_e = min(cands, key=lambda c: c[0])
            best_l = min(cands, key=lambda c: c[1])
            out.append({"reference": ref, "ebn0": op["ebn0"], "reference_fer": op["fer"][ref],
                        "reference_eager": rows.get((ref, "eager")),
                        "reference_graph": rows.get((ref, "graph")),
                        "lowest_energy": {"decoder": best_e[2], "mode": best_e[3], "fer": best_e[4], **best_e[5]},
                        "lowest_latency": {"decoder": best_l[2], "mode": best_l[3], "fer": best_l[4], **best_l[5]}})
    return out


def summary_md(code, results, T_list) -> str:
    g = results["gpu"]
    L = [f"# Step 4: {code.name} (n={code.n}, k={code.k}, {code.num_edges} edges)\n",
         f"GPU {g['device']}, NVML energy: {g['nvml']}, idle power {g['idle_power_w']} W. "
         f"Throughput/energy batch {g['batch']}.\n",
         "## Equal FER, per SNR point (reference with >= 50 frame errors)\n",
         "Cheapest decoder whose FER is within the 95% upper bound of the SPA reference. "
         "Engineering = SPA eager -> SPA CUDA graph; algorithm = SPA graph -> chosen decoder.\n",
         "| reference (FER) | SPA eager: b1 ms / uJ/frame | SPA graph: b1 ms / uJ/frame | lowest-energy decoder (FER) | b1 ms / uJ/frame | lowest-latency decoder | b1 ms |",
         "|---|---|---|---|---|---|---|"]
    for e in results["equal_fer"]:
        re_, rg, be, bl = e["reference_eager"], e["reference_graph"], e["lowest_energy"], e["lowest_latency"]

        def cell(r):
            return f"{r['b1_ms']:.3f} / {r.get('uj_per_frame', float('nan')):.2f}" if r else "-"
        L.append(f"| {e['reference']} @ {e['ebn0']} dB ({e['reference_fer']:.2e}) | {cell(re_)} | {cell(rg)} | "
                 f"{be['decoder']} {be['mode']} ({be['fer']:.2e}) | {cell(be)} | "
                 f"{bl['decoder']} {bl['mode']} ({bl['fer']:.2e}) | {bl['b1_ms']:.3f} |")
    names = list(results["eval"][0]["fer"])
    L += ["\n## FER (paired, same frames; early-stop semantics)\n",
          "| decoder | " + " | ".join(f"{r['ebn0']} dB" for r in results["eval"]) + " |",
          "|---|" + "---:|" * len(results["eval"])]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{r['fer'][n]:.2e}" for r in results["eval"]) + " |")
    L.append("\nframes per SNR: " + ", ".join(str(r["frames"]) for r in results["eval"]))
    L += ["\n## GPU cost (fixed T, no early stop)\n",
          f"| decoder | mode | latency b1 (ms) | ms per batch of {g['batch']} | frames/s | uJ/frame | dynamic uJ/frame |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in g["rows"]:
        L.append(f"| {r['decoder']} | {r['mode']} | {r['b1_ms']:.3f} | {r['big_ms']:.2f} | {r['frames_per_s']:,.0f} | "
                 f"{r.get('uj_per_frame', float('nan')):.2f} | {r.get('dynamic_uj_per_frame', float('nan')):.2f} |")
    L += ["\n## Learned parameters\n"]
    for n, t in results["training"].items():
        L.append(f"- **{n}**: best val FER {t['best_val_fer']:.3e} after {t['steps']} steps; "
                 + ", ".join(f"{k}={np.round(v, 3).tolist()}" for k, v in t["params"].items()))
    return "\n".join(L) + "\n"


def remerge(results_path: Path, gpu_path: Path) -> None:
    results = json.loads(results_path.read_text())
    results["gpu_previous"] = results.get("gpu")
    results["gpu"] = json.loads(gpu_path.read_text())
    T_list = results["args"]["T"]
    results["equal_fer"] = equal_fer(results, T_list)
    code = load_code(results["code"])
    out = results_path.with_name("results_remerged.json")
    out.write_text(json.dumps(results, indent=1, default=float))
    results_path.with_name("summary_remerged.md").write_text(summary_md(code, results, T_list))
    log(f"wrote {out} and {results_path.with_name('summary_remerged.md')}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--code", default="WIMAX_N576_K288")
    p.add_argument("--snr", type=float, nargs=2, default=[1.5, 3.0])
    p.add_argument("--T", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--spa-extra", type=int, nargs="*", default=[50], help="extra SPA references")
    p.add_argument("--learned", nargs="*", default=list(LEARNED))
    p.add_argument("--train-minutes", type=float, default=6)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--val-batch", type=int, default=8192)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-step", type=float, default=0.25)
    p.add_argument("--eval-batch", type=int, default=8192)
    p.add_argument("--eval-min-fe", type=int, default=200)
    p.add_argument("--eval-max-frames", type=int, default=2_000_000)
    p.add_argument("--timing-batch", type=int, default=8192)
    p.add_argument("--energy-seconds", type=float, default=3.0)
    p.add_argument("--max-trace-elems", type=float, default=2.5e8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/step4")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--timing-only", action="store_true",
                   help="skip training/eval; re-measure GPU cost only (cost does not depend on "
                        "learned values) and write gpu.json")
    p.add_argument("--remerge", nargs=2, metavar=("RESULTS_JSON", "GPU_JSON"),
                   help="local: replace the GPU section of an earlier results.json with a "
                        "--timing-only gpu.json and rewrite the summary next to it")
    args = p.parse_args(argv)
    if args.remerge:
        remerge(Path(args.remerge[0]), Path(args.remerge[1]))
        return
    if not args.smoke and not args.device.startswith("cuda"):
        raise SystemExit("Full step-4 runs are GPU-only; use --smoke for a CPU pipeline check.")
    if args.smoke:
        args.train_minutes, args.batch, args.val_batch, args.eval_every = 0.05, 128, 256, 5
        args.eval_batch, args.eval_min_fe, args.eval_max_frames = 256, 5, 512
        args.timing_batch, args.energy_seconds, args.spa_extra = 64, 0.2, []
        args.eval_step = args.snr[1] - args.snr[0]

    code = load_code(args.code)
    out = Path(args.out) / code.name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{code}  device={args.device}")

    decoders, training = {}, {}
    for T in args.T:
        for v in ("spa", "ms", "nms"):
            decoders[name_of(v, T)] = (MessagePassingDecoder(code, v, T).to(args.device), False)
        for v in args.learned:
            if args.timing_only:
                decoders[name_of(v, T)] = (MessagePassingDecoder(code, v, T).to(args.device).eval(), False)
                continue
            dec, info = train(code, v, T, args, args.device)
            decoders[name_of(v, T)] = (dec, False)
            training[name_of(v, T)] = info
            torch.save(dec.state_dict(), out / f"{v}_T{T}.pt")
        for v in ("nms", *args.learned):
            if args.device.startswith("cuda"):
                decoders[name_of(v, T, True)] = (copy.deepcopy(decoders[name_of(v, T)][0]).half(), True)
    for T in args.spa_extra:
        decoders[name_of("spa", T)] = (MessagePassingDecoder(code, "spa", T).to(args.device), False)

    if args.timing_only:
        gpu = measure(code, decoders, args, args.device)
        (out / "gpu.json").write_text(json.dumps(gpu, indent=1, default=float))
        log(f"wrote {out / 'gpu.json'}")
        return

    stop_on = [name_of("spa", max(args.T))]
    results = {"code": code.name, "n": code.n, "k": code.k, "args": vars(args), "training": training}
    results["eval"] = evaluate(code, decoders, args, args.device, stop_on)
    results["gpu"] = measure(code, decoders, args, args.device)
    results["equal_fer"] = equal_fer(results, args.T)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "summary.md").write_text(summary_md(code, results, args.T))
    log((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
