"""Step 3: train BP + sparse GT end to end and compare at equal GPU latency.

For one code:

1. **Train** one :class:`SparseGTDecoder` per resume length R (``--R 0 5``).
   Loss is BCE on the corrected output of frames BP(T) did not decode
   (converged frames skip the transformer, at training and inference alike).
   Gradients flow through the resumed BP into the transformer. Checkpoint
   selection uses a fixed validation set, never the evaluation frames.
2. **Time** every decoder on the GPU, in this process: worst-case latency at
   batch 1 (a frame BP(T) fails, so the full pipeline runs) and throughput at
   the operating SNR.
3. **Equal latency.** Delta* = how many extra BP iterations fit in the
   pipeline's worst-case latency; BP(T + Delta*) is the baseline it must beat.
4. **Evaluate** all decoders on the *same* channel frames (paired), against
   BP(T), BP(T+R), BP(T+Delta*), BP(100), and Chase-BP on the same
   least-reliable bits.

Full runs require a GPU; ``--smoke`` runs a seconds-long CPU check.

    python -m sparse_gt.experiments.step3 --code WIMAX_N576_K288 --snr 1.5 2.75 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..baselines import bp_compact, chase_bp
from ..bp import SumProductBP
from ..channel import awgn_llr
from ..codes import load_code
from ..model import GTConfig, SparseGTDecoder
from ..timing import gpu_time

BP_LONG = 100


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------- train

@torch.no_grad()
def validation_fer(model: SparseGTDecoder, val_llr: list[torch.Tensor]) -> float:
    model.eval()
    errors = frames = 0
    for llr in val_llr:
        out, _ = model(llr)
        errors += int((out < 0).any(1).sum())
        frames += llr.shape[0]
    model.train()
    return errors / frames


def train(code, cfg: GTConfig, args, device: str) -> tuple[SparseGTDecoder, list[dict]]:
    torch.manual_seed(args.seed)
    model = SparseGTDecoder(code, cfg).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    val_gen = torch.Generator(device=device).manual_seed(args.seed + 1000)
    lo, hi = args.snr
    val_snrs = np.linspace(lo, hi, 4)
    val_llr = [awgn_llr(args.val_batch, code.n, float(s), code.rate, device, val_gen)
               for s in val_snrs for _ in range(args.val_batches)]
    opt = torch.optim.AdamW(model.gt.parameters(), lr=args.lr, weight_decay=1e-4)

    budget = args.train_minutes * 60
    start = time.time()
    history, step = [], 0
    best_fer = validation_fer(model, val_llr)  # untrained = BP(T+R)
    best = (best_fer, {k: v.clone() for k, v in model.gt.state_dict().items()})
    log(f"[train R={cfg.R}] untrained val FER {best_fer:.4e} (= BP(T+R))")
    while time.time() - start < budget:
        frac = (time.time() - start) / budget
        for group in opt.param_groups:  # cosine decay over the time budget
            group["lr"] = args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        snr = lo + (hi - lo) * torch.rand(args.batch, device=device, generator=gen)
        llr = awgn_llr(args.batch, code.n, snr, code.rate, device, gen)
        _, aux = model(llr)
        z = aux["active_llr"]
        if z is None:
            continue
        loss = F.binary_cross_entropy_with_logits(-z, torch.zeros_like(z))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gt.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.eval_every == 0:
            fer = validation_fer(model, val_llr)
            row = {"step": step, "minutes": (time.time() - start) / 60, "loss": float(loss),
                   "active_frames": int(z.shape[0]), "val_fer": fer, "lr": opt.param_groups[0]["lr"]}
            history.append(row)
            if fer < best[0]:
                best = (fer, {k: v.clone() for k, v in model.gt.state_dict().items()})
            log(f"[train R={cfg.R}] step {step:6d}  {row['minutes']:5.1f} min  loss {row['loss']:.4f}  "
                f"active {row['active_frames']:5d}  val FER {fer:.4e}  best {best[0]:.4e}")
    model.gt.load_state_dict(best[1])
    model.eval()
    log(f"[train R={cfg.R}] done: {step} steps, best val FER {best[0]:.4e}")
    return model, history


# ------------------------------------------------------------------ timing

def find_failing_frame(bp: SumProductBP, code, T: int, snr: float, device: str) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(7)
    for _ in range(100):
        llr = awgn_llr(256, code.n, snr, code.rate, device, gen)
        res = bp(llr, T)
        bad = torch.nonzero(~res.converged).squeeze(1)
        if bad.numel():
            return llr[bad[:1]]
    raise RuntimeError("no frame fails BP(T) at this SNR")


@torch.no_grad()
def timing(code, models: dict, args, device: str, T: int, snr_op: float) -> dict:
    bp = SumProductBP(code).to(device)
    frame = find_failing_frame(bp, code, T, args.snr[0], device)
    reps = 30 if not args.smoke else 2
    lat = {}
    for N in sorted({T, *range(T + 1, T + 41)}):
        lat[f"BP({N})"] = gpu_time(lambda: bp(frame, N, early_stop=False), device, repeats=reps)["ms"]
    per_iter = (lat[f"BP({T + 40})"] - lat[f"BP({T})"]) / 40
    for name, model in models.items():
        lat[name] = gpu_time(lambda: model(frame), device, repeats=reps)["ms"]
    lat[f"Chase(p={args.chase_p},R={args.chase_R})"] = gpu_time(
        lambda: chase_bp(bp, frame, T, args.chase_p, args.chase_R), device, repeats=reps)["ms"]

    # Throughput at the operating point: realistic mix of easy and hard frames.
    gen = torch.Generator(device=device).manual_seed(11)
    batch = awgn_llr(args.timing_batch, code.n, snr_op, code.rate, device, gen)
    thr = {f"BP({BP_LONG}) compact early-stop":
           gpu_time(lambda: bp_compact(bp, batch, BP_LONG), device, repeats=max(3, reps // 3))["ms"]}
    for N in (T, T + 5, T + 10, T + 20):
        thr[f"BP({N}) compact early-stop"] = gpu_time(
            lambda: bp_compact(bp, batch, N), device, repeats=max(3, reps // 3))["ms"]
    for name, model in models.items():
        thr[name] = gpu_time(lambda: model(batch), device, repeats=max(3, reps // 3))["ms"]
    thr[f"Chase(p={args.chase_p},R={args.chase_R})"] = gpu_time(
        lambda: chase_bp(bp, batch, T, args.chase_p, args.chase_R), device, repeats=max(3, reps // 3))["ms"]

    delta_star = {name: max(0, math.ceil((lat[name] - lat[f"BP({T})"]) / per_iter)) for name in models}
    for name, d in delta_star.items():
        log(f"[timing] {name}: worst-case latency {lat[name]:.3f} ms = BP({T}) + {d} iterations "
            f"(BP iteration {per_iter:.3f} ms)")
    return {"latency_ms_batch1": lat, "ms_per_bp_iteration": per_iter,
            "throughput_ms": thr, "throughput_batch": args.timing_batch, "operating_ebn0": snr_op,
            "equal_latency_extra_iterations": delta_star,
            "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"}


# -------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(code, models: dict, args, device: str, T: int, extra_bp: list[int]) -> list[dict]:
    bp = SumProductBP(code).to(device)
    decoders = {f"BP({N})": (lambda llr, N=N: bp(llr, N).llr) for N in sorted({T, *[T + d for d in extra_bp], BP_LONG})}
    decoders[f"Chase(p={args.chase_p},R={args.chase_R})"] = \
        lambda llr: chase_bp(bp, llr, T, args.chase_p, args.chase_R)
    for name, model in models.items():
        decoders[name] = lambda llr, m=model: m(llr)[0]
    rows = []
    lo, hi = args.snr
    grid = np.round(np.arange(lo, hi + 1e-9, args.eval_step), 3).tolist()
    for snr in grid:
        gen = torch.Generator(device=device).manual_seed(10_000 + int(snr * 100))
        fe = {name: 0 for name in decoders}
        be = {name: 0 for name in decoders}
        frames = 0
        start = time.time()
        while frames < args.eval_max_frames and min(fe[n] for n in models) < args.eval_min_fe:
            llr = awgn_llr(args.eval_batch, code.n, snr, code.rate, device, gen)
            for name, dec in decoders.items():
                errors = (dec(llr) < 0).sum(1)
                fe[name] += int((errors > 0).sum())
                be[name] += int(errors.sum())
            frames += args.eval_batch
        row = {"ebn0": snr, "frames": frames, "seconds": time.time() - start,
               "fer": {n: fe[n] / frames for n in decoders},
               "ber": {n: be[n] / (frames * code.n) for n in decoders},
               "frame_errors": fe}
        rows.append(row)
        log(f"[eval] Eb/N0 {snr:.2f}  frames {frames}  " +
            "  ".join(f"{n}:{row['fer'][n]:.2e}" for n in decoders))
    return rows


# -------------------------------------------------------------------- main

def summary_md(code, cfgs, results) -> str:
    T = cfgs[0].T
    lat = results["timing"]["latency_ms_batch1"]
    lines = [f"# Step 3: {code.name} (n={code.n}, k={code.k})\n",
             f"T={T}, k={cfgs[0].k} selected bits, d_model={cfgs[0].d_model}, layers={cfgs[0].layers}, "
             f"heads={cfgs[0].heads}. GPU: {results['timing']['device']}.\n",
             "## Worst-case latency, batch 1 (frame that BP(T) fails)\n",
             "| decoder | ms | equal-latency BP |", "|---|---:|---|"]
    for name, d in results["timing"]["equal_latency_extra_iterations"].items():
        lines.append(f"| {name} | {lat[name]:.3f} | BP({T + d}) = {lat.get(f'BP({T + d})', float('nan')):.3f} ms |")
    lines.append(f"| BP({T}) | {lat[f'BP({T})']:.3f} | |")
    lines += ["\n## Throughput at the operating point\n",
              f"batch {results['timing']['throughput_batch']} at Eb/N0 {results['timing']['operating_ebn0']} dB\n",
              "| decoder | ms per batch |", "|---|---:|"]
    for name, ms in results["timing"]["throughput_ms"].items():
        lines.append(f"| {name} | {ms:.2f} |")
    names = list(results["eval"][0]["fer"])
    lines += ["\n## FER (paired, same frames)\n", "| Eb/N0 | frames | " + " | ".join(names) + " |",
              "|---:|---:|" + "---:|" * len(names)]
    for r in results["eval"]:
        cells = [f"{r['fer'][n]:.2e}" for n in names]
        lines.append(f"| {r['ebn0']} | {r['frames']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--code", default="WIMAX_N576_K288")
    p.add_argument("--snr", type=float, nargs=2, default=[1.5, 2.75], help="train/eval Eb/N0 range")
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k", type=float, default=0.05, help="selected bits: int, or fraction of n if < 1")
    p.add_argument("--R", type=int, nargs="+", default=[0, 5], help="one model per resume length")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--no-structure-bias", action="store_true")
    p.add_argument("--train-minutes", type=float, default=40)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--val-batch", type=int, default=4096)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--eval-step", type=float, default=0.25)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument("--eval-min-fe", type=int, default=200)
    p.add_argument("--eval-max-frames", type=int, default=2_000_000)
    p.add_argument("--timing-batch", type=int, default=4096)
    p.add_argument("--chase-p", type=int, default=3)
    p.add_argument("--chase-R", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/step3")
    p.add_argument("--smoke", action="store_true", help="seconds-long CPU pipeline check")
    args = p.parse_args(argv)

    if not args.smoke and not args.device.startswith("cuda"):
        raise SystemExit("Full step-3 runs are GPU-only (the problem statement is GPU decoding). "
                         "Use --smoke for a CPU pipeline check.")
    if args.smoke:
        args.train_minutes, args.batch, args.eval_every = 0.1, 128, 5
        args.val_batch, args.val_batches, args.eval_batch = 128, 1, 256
        args.eval_min_fe, args.eval_max_frames, args.timing_batch = 5, 512, 64
        args.eval_step = (args.snr[1] - args.snr[0])

    code = load_code(args.code)
    k = int(args.k) if args.k >= 1 else max(8, round(args.k * code.n))
    out = Path(args.out) / code.name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{code}  device={args.device}  "
        f"{torch.cuda.get_device_name(0) if args.device.startswith('cuda') else ''}")

    cfgs, models, histories = [], {}, {}
    for R in args.R:
        cfg = GTConfig(T=args.T, k=k, R=R, d_model=args.d_model, heads=args.heads,
                       layers=args.layers, structure_bias=not args.no_structure_bias)
        model, hist = train(code, cfg, args, args.device)
        name = f"BP({args.T})+GT(k={k})+BP({R})"
        cfgs.append(cfg)
        models[name] = model
        histories[name] = hist
        torch.save({"cfg": cfg.__dict__, "state_dict": model.gt.state_dict()}, out / f"gt_R{R}.pt")
        log(f"{name}: {sum(p.numel() for p in model.gt.parameters())} parameters")

    snr_op = float(np.mean(args.snr))
    tim = timing(code, models, args, args.device, args.T, snr_op)
    extra = sorted({*tim["equal_latency_extra_iterations"].values(), *args.R, 5, 10, 20})
    ev = evaluate(code, models, args, args.device, args.T, extra)
    results = {"code": code.name, "n": code.n, "k": code.k, "args": vars(args),
               "configs": [c.__dict__ for c in cfgs], "train_history": histories,
               "timing": tim, "eval": ev}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text(summary_md(code, cfgs, results))
    log((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
