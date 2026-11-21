"""One-session T4 benchmark: BP, BP-GNN, BP-SGT and BP-DAT, original vs fast.

Every row runs in this one process on one GPU, interleaved in rounds, so the
rows are comparable (earlier profiles came from separate Kaggle sessions).

Decoders per code (5x10 schedule for the hybrids, BP(50) for BP):
  BP(50)            original common.bp.bp_decode (early stop with a host sync and
                    batch compaction every iteration)
  BP(50) fast       StagedEngine: masked early stop, padded layout, one CUDA graph
                    per block of --check-every iterations, one host check per block
  BP(50) fast fixed same with no host check and no compaction (all 50 iterations, no sync)
  BP-GNN / SGT      original decode, and "fast engine": the same trained module run
                    inside StagedEngine (the engine alone, module unchanged)
  DAT               original decode; "fast fp32": engine + fast_dat_refine (padded
                    softmax, no scatters); "fast fp16": the module under fp16 autocast
                    (BP stays fp32); "fast fp16 fixed": no host checks, no compaction
                    (zero device -> host reads per decode)
  (--compile adds "DAT fast fp32 compiled": torch.compile of every block before capture)

For every decoder:
  * error rates on the same frames (FER, BER, mean BP iterations) at the profiling
    Eb/N0 and 0.5 dB below it, and for every fast row the number of frames whose
    decision differs from its original and whether the same frames are in error;
  * latency at B = 1 over --b1-frames distinct frames (mean, median, p95), i.e.
    what one codeword waits, early stopping included;
  * throughput (codewords/s) at every --batches size on one fixed batch.

Models: trained BP-GNN and BP-SGT checkpoints of the final runs (committed in the
repo and bundled in the kernel); BP-DAT trained if a final_dat_*_<code>_dat.pt is
found (e.g. /kaggle/input after its Kaggle runs), otherwise untrained = BP with
restarts (zero head). Fixed-schedule timing does not depend on the weights;
early-stop timing does, so untrained DAT early-stop rows are only indicative.

Usage: python bench_t4.py [--codes 5G_BG1_k528_n704,5G_BG1_k1056_n1408] [--quick] [--out DIR]
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastdat  # noqa: E402,F401
from fastdat.dat_fast import fast_dat_refine  # noqa: E402
from fastdat.engine import StagedEngine  # noqa: E402
from fastdat.layout import Layout  # noqa: E402
from fastdat.models import build_bpgnn, build_dat, build_sgt  # noqa: E402
from common.bp import bp_decode  # noqa: E402
from dat.specs import load_any_code  # noqa: E402
from run_final import code_specs  # noqa: E402

ORIGINAL_OF = lambda name: name.split(" fast")[0]  # "DAT fast fp16" -> "DAT"


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def prof_ebno(name):
    for c, _, pe, _ in code_specs("5g", torch.device("cpu")):
        if c.name == name:
            return pe
    raise KeyError(name)


def decoders(code, dev, args, log):
    g = code.graph
    lay = Layout(g, "padded")
    ce = args.check_every
    out, info = {}, {}
    out["BP(50)"] = lambda x: bp_decode(x, g, 50)
    out["BP(50) fast"] = StagedEngine(g, 0, 50, layout=lay, check_every=ce)
    out["BP(50) fast fixed"] = StagedEngine(g, 0, 50, layout=lay, check_every=0, compact="none")
    for name, m in (("BP-GNN", build_bpgnn(code, dev)), ("SGT", build_sgt(code, dev))):
        if m is None:
            log(f"  {name}: no trained checkpoint for {code.name}, skipped")
            continue
        info[name] = dict(trained_from=m.trained_from, params=sum(p.numel() for p in m.parameters()))
        out[name] = (lambda mm: lambda x: mm.decode(x, g))(m)
        refine = m.refine_chunked if hasattr(m, "refine_chunked") else m.refine
        out[f"{name} fast engine"] = StagedEngine(g, m.stages, m.bp_iters, refine, layout=lay, check_every=ce,
                                                  warm=getattr(m, "warm", False))
    m = build_dat(code, dev)
    info["DAT"] = dict(trained_from=m.trained_from, params=sum(p.numel() for p in m.parameters()),
                       macs_per_stage=m.macs_per_refinement(g))
    if m.trained_from is None:
        log(f"  DAT: no trained checkpoint for {code.name}; untrained (zero head = BP with restarts)")
    out["DAT"] = lambda x: m.decode(x, g)
    out["DAT fast fp32"] = StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay), layout=lay,
                                        check_every=ce)
    if dev.type == "cuda":
        f16 = fast_dat_refine(m, lay, torch.float16)
        out["DAT fast fp16"] = StagedEngine(g, m.stages, m.bp_iters, f16, layout=lay, check_every=ce)
        out["DAT fast fp16 fixed"] = StagedEngine(g, m.stages, m.bp_iters, f16, layout=lay, check_every=0,
                                                  compact="none")
    if args.compile:
        out["DAT fast fp32 compiled"] = StagedEngine(g, m.stages, m.bp_iters, fast_dat_refine(m, lay), layout=lay,
                                                     check_every=ce, compile=True)
    return out, info


def error_rates(decs, code, ebnos, frames, batch, dev, log):
    res = {}
    torch.manual_seed(1234)
    sets = []
    for e in ebnos:
        cs, ls = [], []
        for i in range(0, frames, batch):
            c, llr = code.sample(min(batch, frames - i), e)
            cs.append(c.float().to(dev))
            ls.append(llr.to(dev))
        sets.append((e, cs, ls))
    for name, fn in decs.items():
        for e, cs, ls in sets:
            errs, bit_err, iters, dec = 0, 0.0, 0.0, []
            try:
                for c, llr in zip(cs, ls):
                    L, used = fn(llr)
                    hard = (L < 0).float()                       # as common.evaluate.simulate
                    err = hard != c
                    ferr = err.any(1)
                    errs += int(ferr.sum())
                    bit_err += float(err[:, code.info_pos].float().mean(1).sum())
                    iters += float(used.float().sum())
                    dec.append((hard.to(torch.uint8).cpu(), ferr.cpu()))
            except Exception as ex:  # noqa: BLE001
                log(f"    {name} @ {e} dB: {type(ex).__name__}: {str(ex)[:200]}")
                continue
            r = dict(fer=errs / frames, ber=bit_err / frames, mean_iters=iters / frames, frame_errors=errs)
            res.setdefault(name, {})[f"{e:.2f}"] = r
            res[name][f"_dec{e:.2f}"] = dec
    # fast rows against their original, on the same frames
    for name in list(res):
        orig = ORIGINAL_OF(name)
        if orig == name or orig not in res:
            continue
        for key in [k for k in res[name] if not k.startswith("_")]:
            a, b = res[orig].get(f"_dec{key}"), res[name].get(f"_dec{key}")
            if a is None or b is None:
                continue
            diff = sum(int((x[0] != y[0]).any(1).sum()) for x, y in zip(a, b))
            same_err = all(torch.equal(x[1], y[1]) for x, y in zip(a, b))
            res[name][key].update(frames_with_other_decision=diff, same_frames_in_error=same_err)
    for name in res:
        for k in [k for k in res[name] if k.startswith("_")]:
            del res[name][k]
    return res


def time_b1(fn, llrs, dev, warm=3):
    for x in llrs[:warm]:
        fn(x)
    ts = []
    for x in llrs:
        sync(dev)
        t0 = time.perf_counter()
        fn(x)
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(mean_ms=statistics.fmean(ts), median_ms=statistics.median(ts),
                p95_ms=ts[min(len(ts) - 1, int(0.95 * len(ts)))])


def time_batch(fn, x, dev, reps, max_s=4.0):
    fn(x)
    fn(x)
    ts = []
    t_end = time.perf_counter() + max_s
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        fn(x)
        sync(dev)
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() > t_end:
            break
    return statistics.median(ts)


def report(name, ebnos, res, batches):
    L = [f"## {name}", ""]
    info = res["models"]
    L.append("Models: " + "; ".join(f"{k}: {v['params']} params, "
                                     + ("trained" if v.get("trained_from") else "**untrained**")
                                     for k, v in info.items()))
    L.append("")
    cols = ["decoder", "B=1 mean ms", "B=1 median", "B=1 p95"] + [f"B={b} cw/s" for b in batches] + \
           [f"FER @{e:.2f}" for e in ebnos] + [f"iters @{ebnos[0]:.2f}", "host reads / decode", "vs original"]
    L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for d, r in res["rows"].items():
        b1 = r.get("b1", {})
        cells = [d, f"{b1.get('mean_ms', float('nan')):.2f}", f"{b1.get('median_ms', float('nan')):.2f}",
                 f"{b1.get('p95_ms', float('nan')):.2f}"]
        cells += [f"{r.get('tput', {}).get(str(b), float('nan')):,.0f}" for b in batches]
        er = res["errors"].get(d, {})
        cells += [f"{er[f'{e:.2f}']['fer']:.2e}" if f"{e:.2f}" in er else "-" for e in ebnos]
        k0 = f"{ebnos[0]:.2f}"
        cells.append(f"{er[k0]['mean_iters']:.2f}" if k0 in er else "-")
        cells.append(f"{r['host_syncs_per_decode_mean']:.1f}" if "host_syncs_per_decode_mean" in r
                     else "1 per iteration")
        eq = [er[k] for k in er if "frames_with_other_decision" in er[k]]
        cells.append("-" if not eq else
                     ("same frames in error" if all(x["same_frames_in_error"] for x in eq) else "**differs**")
                     + f", {sum(x['frames_with_other_decision'] for x in eq)} frames with other bits")
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="5G_BG1_k528_n704,5G_BG1_k1056_n1408")
    ap.add_argument("--batches", default="1,64,1024,4096")
    ap.add_argument("--b1-frames", type=int, default=200)
    ap.add_argument("--fer-frames", type=int, default=40960)
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.quick:
        args.batches, args.b1_frames, args.fer_frames, args.rounds = "1,16", 6, 64, 1
    batches = [int(b) for b in args.batches.split(",") if int(b) > 1]
    out = args.out or ("/kaggle/working/results/fastdat_t4" if os.path.isdir("/kaggle/working")
                       else os.path.join(HERE, "results", "fastdat_local"))
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"device {dev} | torch {torch.__version__}" + (f" | {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    log(f"args {vars(args)}")
    all_res, parts = {}, ["# Fast BP-DAT benchmark (one session)", "",
                         f"Device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}, torch "
                         f"{torch.__version__}. Fast rows: host check every {args.check_every} BP iterations "
                         "(\"fixed\": none). B=1 = mean over distinct frames at the profiling Eb/N0, "
                         "early stopping included. 'vs original' compares the fast row with the row it "
                         "accelerates on the same frames.", ""]
    for name in args.codes.split(","):
        code = load_any_code(name, dev)
        g = code.graph
        pe = prof_ebno(name)
        ebnos = [pe, pe - 0.5]
        log(f"\n=== {name}: n={g.n} m={g.m} E={g.num_edges} | profiling Eb/N0 {pe} dB ===")
        decs, info = decoders(code, dev, args, log)
        log("  decoders: " + ", ".join(decs))
        res = dict(models=info, rows={d: {} for d in decs})
        log("  error rates / equivalence ...")
        res["errors"] = error_rates(decs, code, ebnos, args.fer_frames, 1024 if dev.type == "cuda" else 32, dev, log)
        for d, r in res["errors"].items():
            log(f"    {d:26s} " + " | ".join(f"{k} dB FER {v['fer']:.2e} it {v['mean_iters']:.1f}"
                                            + (f" diff {v['frames_with_other_decision']} same-err "
                                               f"{v['same_frames_in_error']}" if "same_frames_in_error" in v else "")
                                            for k, v in r.items()))
        torch.manual_seed(99)
        b1 = [code.sample(1, pe)[1].to(dev) for _ in range(args.b1_frames)]
        xb = {b: code.sample(b, pe)[1].to(dev) for b in batches}
        for rnd in range(args.rounds):
            log(f"  timing round {rnd + 1}/{args.rounds}")
            for d, fn in decs.items():
                try:
                    t = time_b1(fn, b1, dev)
                    res["rows"][d].setdefault("b1_rounds", []).append(t)
                    for b in batches:
                        try:
                            s = time_batch(fn, xb[b], dev, reps=10 if b >= 1024 else 30)
                            res["rows"][d].setdefault("tput_rounds", {}).setdefault(str(b), []).append(b / s)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            log(f"    {d} B={b}: OOM")
                    log(f"    {d:26s} B=1 {t['mean_ms']:.2f} ms (median {t['median_ms']:.2f})  " +
                        "  ".join(f"B={b} {res['rows'][d].get('tput_rounds', {}).get(str(b), [float('nan')])[-1]:,.0f}"
                                  for b in batches))
                except Exception as ex:  # noqa: BLE001
                    log(f"    {d}: {type(ex).__name__}: {str(ex)[:200]}")
        for d, r in res["rows"].items():
            if "b1_rounds" in r:
                r["b1"] = {k: statistics.median(x[k] for x in r["b1_rounds"]) for k in r["b1_rounds"][0]}
            r["tput"] = {b: statistics.median(v) for b, v in r.get("tput_rounds", {}).items()}
            if isinstance(decs[d], StagedEngine):
                st = decs[d].stats
                r["host_syncs_per_decode_max"] = decs[d].host_syncs_per_decode()
                r["host_syncs_per_decode_mean"] = st["host_syncs"] / max(1, st["calls"])
                r["graph_count"] = len(decs[d]._graphs)
        all_res[name] = res
        parts.append(report(name, ebnos, res, batches))
        with open(os.path.join(out, "fastdat_bench.json"), "w") as f:
            json.dump(all_res, f, indent=1, default=str)
        with open(os.path.join(out, "REPORT.md"), "w") as f:
            f.write("\n\n".join(parts) + "\n")
        del decs
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("\n" + "\n\n".join(parts))
    log("done")


if __name__ == "__main__":
    main()
