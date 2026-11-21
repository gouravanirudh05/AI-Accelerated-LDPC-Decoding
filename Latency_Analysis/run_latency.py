"""Component-level latency analysis of BP and BP-SGT, all in ONE process on ONE GPU
so that every row of the report is comparable (earlier profile rows came from
separate Kaggle sessions).

  python run_latency.py --codes mackay_204_102,5G_BG1_k1056_n1408 [--ckpt-root /kaggle/input] [--quick]

Sections per code (see README.md for what each one answers):
  bp_iters      fixed-iteration BP, eager vs CUDA graph, T in {1,10,50,200}: fixed cost
                + cost per iteration, and how much of it is launch overhead
  bp_regions    where one BP decode spends its time, per component (sync wall time,
                GPU kernel time, kernel launches, host time, host syncs)
  bp_variants   one change at a time: no host sync, check every K, CUDA graphs,
                aggregation backend, fused sign, min-sum, fp16; latency, throughput,
                mean iterations and FER on the same frames
  agg_micro     the scatter-sum alone, per backend and batch size
  sgt_refine    one SGT stage per component (features, top-k, tokens, attention-bias
                build split into Hk gather / shared bmm / shared-unsat bmm / embed,
                attention, FFN, head), for the orig and two equivalent cheaper bias
                variants and struct_bias off
  sgt_k         one SGT stage versus k_frac
  sgt_decode    the full BP-SGT decode split into BP stages, SGT stages and
                compaction, fixed 5x10 and early-stop, random and hard frames
Outputs: <out>/<code>.json, <out>/REPORT.md (tables), <out>/traces/*.json (chrome traces).
"""

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lat  # noqa: E402,F401
from common.bp import bp_decode  # noqa: E402
from common.experiment import Logger  # noqa: E402
from lat import bp_inst, sgt_inst  # noqa: E402
from lat.bp_fast import AGGS, Aggregator, FastBP  # noqa: E402
from lat.codes import build_model, load, prof_ebno  # noqa: E402
from lat.regions import count_host_syncs  # noqa: E402
from lat.report import write_report  # noqa: E402
from lat.timing import graphed_or_none, interleaved, profiler_breakdown, sync, sync_breakdown, timed  # noqa: E402

DEFAULT_CODES = "mackay_204_102,wimax_576_288,5G_BG1_k1056_n1408"


def frames(code, B, ebno, seed):
    torch.manual_seed(seed)
    return code.sample(B, ebno)


def hard_frames(code, B, ebno, seed, iters=10, chunk=4096, max_chunks=64):
    """Frames that BP(iters) does not solve: the only frames an SGT stage ever sees."""
    torch.manual_seed(seed)
    cs, ls, have = [], [], 0
    for _ in range(max_chunks):
        c, llr = code.sample(chunk, ebno)
        L, _ = bp_decode(llr, code.graph, iters)
        bad = ~(code.graph.syndrome((L < 0).float()) == 0).all(1)
        cs.append(c[bad])
        ls.append(llr[bad])
        have += int(bad.sum())
        if have >= B:
            break
    c, llr = torch.cat(cs)[:B], torch.cat(ls)[:B]
    if c.shape[0] < B:  # repeat if the SNR is too high to find B failures
        r = math.ceil(B / max(1, c.shape[0]))
        c, llr = c.repeat(r, 1)[:B], llr.repeat(r, 1)[:B]
    return c, llr


def fer(dec, code, c, llr):
    L, used = dec(llr)
    err = ((L < 0).to(c.dtype) != c).any(1)
    return float(err.float().mean()), float(used.float().mean())


# --------------------------------------------------------------------------- BP
def sec_bp_iters(code, ebno, bss, q, log):
    g, out = code.graph, {}
    Ts = (1, 10, 50) if q else (1, 10, 50, 200)
    for B in bss:
        _, llr = frames(code, B, ebno, 1)
        cases = {}
        for T in Ts:
            fn = (lambda T: lambda x: bp_decode(x, g, T, early_stop=False)[0])(T)
            cases[f"eager T={T}"] = fn
            gf = graphed_or_none(fn, llr, log)
            if gf is not None:
                cases[f"graph T={T}"] = gf
        r = interleaved(cases, llr, rounds=2, reps=10 if q else 20)
        fit = {}
        for kind in ("eager", "graph"):
            pts = [(T, r[f"{kind} T={T}"]["median_ms"]) for T in Ts if f"{kind} T={T}" in r]
            if len(pts) >= 2:
                n = len(pts)
                mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
                b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
                fit[kind] = dict(fixed_ms=my - b * mx, per_iter_ms=b)
        out[B] = dict(times=r, fit=fit)
        log(f"  bp_iters B={B}: " + " | ".join(f"{k} {v['median_ms']:.3f}" for k, v in r.items()))
        log(f"    fit: " + " | ".join(f"{k}: {v['fixed_ms']:.3f} ms + {v['per_iter_ms'] * 1e3:.1f} us/it"
                                     for k, v in fit.items()))
    return out


def sec_bp_regions(code, ebno, bss, q, out_dir, log):
    g, dev, out = code.graph, code.device, {}
    for B in bss:
        _, llr = frames(code, B, ebno, 2)
        for name, T, es in (("early-stop T=50", 50, True), ("fixed T=50", 50, False)):
            run = (lambda T, es: lambda R: bp_inst.bp_decode(llr, g, T, early_stop=es, R=R))(T, es)
            plain = timed(lambda x: bp_decode(x, g, T, early_stop=es), llr, reps=10 if q else 20)
            sb = sync_breakdown(run, dev, 3 if q else 10)
            trace = os.path.join(out_dir, "traces", f"{code.name}_bp_{'es' if es else 'fixed'}_B{B}.json")
            pb = profiler_breakdown(run, dev, 2 if q else 5, trace if B == bss[0] else None)
            syncs = count_host_syncs(lambda: bp_decode(llr, g, T, early_stop=es), dev)
            _, used = bp_decode(llr, g, T, early_stop=es)
            out[f"{name} B={B}"] = dict(plain_ms=plain, sync=sb, prof=pb, host_syncs=syncs,
                                        mean_iters=float(used.float().mean()), max_iters=int(used.max()),
                                        B=B, T=T)
            log(f"  bp_regions {name} B={B}: plain {plain:.3f} ms, sync-mode {sb['wall_ms']:.3f} ms, "
                f"host syncs {syncs}, iters mean {used.float().mean():.1f} max {int(used.max())}")
    return out


def bp_variant_cases(g, T, dev):
    v = {"orig bp_decode early-stop": lambda x: bp_decode(x, g, T)}
    v["orig bp_decode fixed"] = lambda x: bp_decode(x, g, T, early_stop=False)
    v["masked, check every 1 (eager)"] = FastBP(g, T, check_every=1)
    v["masked, check every 10 (eager)"] = FastBP(g, T, check_every=10)
    v["masked, never check (eager)"] = FastBP(g, T, check_every=T)
    if dev.type == "cuda":
        v["masked, check every 10, CUDA graph"] = FastBP(g, T, check_every=10, graph=True)
        v["masked, check every 5, CUDA graph"] = FastBP(g, T, check_every=5, graph=True)
        v["masked, never check, CUDA graph"] = FastBP(g, T, check_every=T, graph=True)
        gk = dict(check_every=10, graph=True)
        v["  + fused sign"] = FastBP(g, T, fuse_sign=True, **gk)
        for a in AGGS[1:]:
            v[f"  + fused sign, agg={a}"] = FastBP(g, T, fuse_sign=True, agg=a, **gk)
        v["  + fused sign, torch.compile"] = FastBP(g, T, fuse_sign=True, compile=True, **gk)
        v["  min-sum fp32"] = FastBP(g, T, cn="nms", **gk)
        v["  min-sum fp16"] = FastBP(g, T, cn="nms", dtype=torch.float16, **gk)
        v["  min-sum fp16, torch.compile"] = FastBP(g, T, cn="nms", dtype=torch.float16, compile=True, **gk)
    else:
        v["  + fused sign (eager)"] = FastBP(g, T, fuse_sign=True, check_every=10)
        v["  + fused sign, agg=padded (eager)"] = FastBP(g, T, fuse_sign=True, agg="padded", check_every=10)
        v["  + fused sign, torch.compile (eager)"] = FastBP(g, T, fuse_sign=True, check_every=10, compile=True)
    return v


def sec_bp_variants(code, ebno, bss, q, log):
    g, dev, out = code.graph, code.device, {}
    for T in ((50,) if q else (50, 200)):
        cases = bp_variant_cases(g, T, dev)
        # same frames for every variant: FER / mean iterations sanity check
        c, llr = frames(code, 2048 if q else 16384, ebno, 3)
        qual = {}
        for n, dec in cases.items():
            try:
                qual[n] = fer(dec, code, c, llr)
            except Exception as e:  # noqa: BLE001
                qual[n] = (float("nan"), float("nan"))
                log(f"      {n}: {type(e).__name__}: {str(e)[:160]}")
        times = {}
        for B in bss:
            _, x = frames(code, B, ebno, 4)
            fns = {n: (lambda d: lambda y: d(y)[0])(d) for n, d in cases.items()}
            times[B] = interleaved(fns, x, rounds=2, reps=10 if q else 20)
        out[T] = dict(fer_iters=qual, times=times)
        for n in cases:
            log(f"  bp_variants T={T} {n:42s} FER {qual[n][0]:.2e} it {qual[n][1]:5.2f} | " +
                " | ".join(f"B={B} {times[B][n]['median_ms']:.3f} ms" for B in bss))
    return out


def sec_agg_micro(code, bss, q, log):
    g, dev, out = code.graph, code.device, {}
    for side, index, size in (("check", g.e_c, g.m), ("variable", g.e_v, g.n)):
        for a in AGGS:
            agg = Aggregator(index, size, a, torch.float32, dev)
            for B in bss:
                x = torch.randn(B, g.num_edges, device=dev)
                n_loop = 20

                def loop(y, agg=agg):
                    for _ in range(n_loop):
                        r = agg(y)
                    return r
                try:
                    eager = timed(loop, x, reps=10) / n_loop
                    gf = graphed_or_none(loop, x, log)
                    gpu = timed(gf, x, reps=10) / n_loop if gf is not None else float("nan")
                except Exception as e:  # noqa: BLE001
                    eager = gpu = float("nan")
                    log(f"      agg {a} {side} B={B}: {type(e).__name__}: {str(e)[:120]}")
                out[f"{side} {a} B={B}"] = dict(side=side, agg=a, B=B, eager_us=eager * 1e3, graph_us=gpu * 1e3)
    for k, v in out.items():
        log(f"  agg_micro {k:28s} eager {v['eager_us']:8.1f} us  graph {v['graph_us']:8.1f} us")
    return out


# -------------------------------------------------------------------------- SGT
def sec_sgt_refine(code, model, ebno, bss, q, out_dir, log):
    g, dev, out = code.graph, code.device, {}
    for B in bss:
        _, llr = hard_frames(code, B, ebno, 5, chunk=1024 if q else 4096, max_chunks=8 if q else 64)
        L, v2c, c2v, _, _ = bp_inst.bp_stage(llr, g, model.bp_iters, early_stop=False)
        for variant in sgt_inst.VARIANTS:
            ref = sgt_inst.make_refine(model, variant)
            run = (lambda ref: lambda R: ref(llr, llr, L, v2c, c2v, g, R))(ref)
            try:
                plain = timed(lambda x: ref(x, x, L, v2c, c2v, g), llr, reps=10 if q else 20)
                sb = sync_breakdown(run, dev, 3 if q else 10)
                trace = os.path.join(out_dir, "traces", f"{code.name}_sgt_refine_{variant}_B{B}.json")
                pb = profiler_breakdown(run, dev, 2 if q else 5,
                                        trace if (B == bss[-1] and variant == "orig") else None)
            except Exception as e:  # noqa: BLE001
                log(f"      sgt_refine {variant} B={B}: {type(e).__name__}: {str(e)[:160]}")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            out[f"{variant} B={B}"] = dict(variant=variant, B=B, plain_ms=plain, sync=sb, prof=pb,
                                           k=model.num_selected(g.n))
            log(f"  sgt_refine {variant:8s} B={B}: plain {plain:.3f} ms, sync-mode {sb['wall_ms']:.3f} ms")
    return out


def sec_sgt_k(code, ebno, B, q, log):
    g, dev, out = code.graph, code.device, {}
    _, llr = hard_frames(code, B, ebno, 6, chunk=1024 if q else 4096, max_chunks=8 if q else 64)
    L, v2c, c2v, _, _ = bp_inst.bp_stage(llr, g, 10, early_stop=False)
    for kf in ((0.1, 0.5) if q else (0.05, 0.1, 0.2, 0.3, 0.5)):
        model = build_model(code, dev, k_frac=kf)
        for variant in ("orig", "gather16"):
            ref = sgt_inst.make_refine(model, variant)
            try:
                sb = sync_breakdown(lambda R: ref(llr, llr, L, v2c, c2v, g, R), dev, 3 if q else 10)
            except Exception as e:  # noqa: BLE001
                log(f"      sgt_k {kf} {variant}: {type(e).__name__}: {str(e)[:120]}")
                continue
            out[f"k_frac={kf} {variant}"] = dict(k_frac=kf, k=model.num_selected(g.n), variant=variant,
                                                 sync=sb, B=B)
            log(f"  sgt_k k_frac={kf} (k={model.num_selected(g.n)}) {variant}: {sb['wall_ms']:.3f} ms")
    return out


def sec_sgt_decode(code, model, ebno, bss, q, log):
    g, dev, out = code.graph, code.device, {}
    for B in bss:
        for fname, (c, llr) in (("random", frames(code, B, ebno, 7)),
                                ("hard", hard_frames(code, B, ebno, 8, chunk=1024 if q else 4096,
                                                     max_chunks=8 if q else 64))):
            for es in (True, False):
                for variant in ("orig", "gather16"):
                    ref = sgt_inst.make_refine(model, variant)
                    stats = []

                    def run(R, ref=ref, es=es, llr=llr):
                        stats.clear()
                        return bp_inst.staged_decode(llr, g, model.stages, model.bp_iters, ref, es, R, stats)
                    try:
                        plain = timed(lambda x: model.decode(x, g, early_stop=es), llr, reps=5 if q else 10) \
                            if variant == "orig" else None
                        sb = sync_breakdown(run, dev, 2 if q else 5)
                        syncs = count_host_syncs(lambda: run(lat.regions.OFF), dev)
                        _, used = run(lat.regions.OFF)
                    except Exception as e:  # noqa: BLE001
                        log(f"      sgt_decode {fname} es={es} {variant} B={B}: {type(e).__name__}: {str(e)[:160]}")
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    key = f"{fname} {'early-stop' if es else 'fixed'} {variant} B={B}"
                    out[key] = dict(frames=fname, early_stop=es, variant=variant, B=B, plain_ms=plain,
                                    sync=sb, host_syncs=syncs, active_per_stage=list(stats),
                                    mean_bp_iters=float(used.float().mean()))
                    log(f"  sgt_decode {key}: plain {plain if plain is None else round(plain, 3)} ms, "
                        f"sync-mode {sb['wall_ms']:.3f} ms, active/stage {stats}, host syncs {syncs}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=DEFAULT_CODES)
    ap.add_argument("--sections", default="bp_iters,bp_regions,bp_variants,agg_micro,sgt_refine,sgt_k,sgt_decode")
    ap.add_argument("--ckpt-root", default="/kaggle/input")
    ap.add_argument("--quick", action="store_true", help="seconds-long CPU smoke test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cost-model", action="store_true", help="also write cost_model.md (analytic MACs)")
    args = ap.parse_args()
    q = args.quick
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or ("/kaggle/working/results/latency" if os.path.isdir("/kaggle/working")
                           else os.path.join(HERE, "results", "quick" if q else "latency"))
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    log = Logger(os.path.join(out_dir, "log.txt"))
    env = dict(device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
               cpu=platform.processor() or platform.machine(), visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               started=time.strftime("%Y-%m-%d %H:%M:%S"), quick=q)
    log(f"env: {env}")
    torch.backends.cuda.matmul.allow_tf32 = False  # T4 has no TF32; keep other GPUs comparable
    secs = args.sections.split(",")
    for cname in args.codes.split(","):
        code = load(cname, dev)
        ebno = prof_ebno(cname)
        g = code.graph
        big = g.num_edges > 3000
        bss = (1, 64) if q else ((1, 64, 1024) if big else (1, 64, 1024, 4096))
        sgt_bss = (1, 16) if q else ((1, 64, 512) if big else (1, 64, 1024))
        model = build_model(code, dev, args.ckpt_root)
        log(f"\n=== {cname}: n {g.n} m {g.m} E {g.num_edges} | Eb/N0 {ebno} dB | SGT k={model.num_selected(g.n)} "
            f"| checkpoint {model.trained_from} | config vs assumed final {model.config_diff}")
        res = dict(code=cname, n=g.n, m=g.m, E=g.num_edges, ebno=ebno, env=env,
                   sgt=dict(k=model.num_selected(g.n), heads=model.heads, d=model.inp.out_features,
                            layers=len(model.blocks), stages=model.stages, bp_iters=model.bp_iters,
                            trained_from=model.trained_from))
        t0 = time.time()
        if "bp_iters" in secs:
            res["bp_iters"] = sec_bp_iters(code, ebno, bss, q, log)
        if "bp_regions" in secs:
            res["bp_regions"] = sec_bp_regions(code, ebno, (1, 1024) if not q else (1, 64), q, out_dir, log)
        if "bp_variants" in secs:
            res["bp_variants"] = sec_bp_variants(code, ebno, bss, q, log)
        if "agg_micro" in secs:
            res["agg_micro"] = sec_agg_micro(code, bss, q, log)
        if "sgt_refine" in secs:
            res["sgt_refine"] = sec_sgt_refine(code, model, ebno, sgt_bss, q, out_dir, log)
        if "sgt_k" in secs:
            res["sgt_k"] = sec_sgt_k(code, ebno, 16 if q else 64, q, log)
        if "sgt_decode" in secs:
            res["sgt_decode"] = sec_sgt_decode(code, model, ebno, (1, 16) if q else (1, sgt_bss[-1]), q, log)
        res["seconds"] = time.time() - t0
        with open(os.path.join(out_dir, f"{cname}.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        log(f"  {cname} done in {res['seconds'] / 60:.1f} min")
        sync(dev)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if args.cost_model:
        import contextlib
        import cost_model
        with open(os.path.join(out_dir, "cost_model.md"), "w") as f, contextlib.redirect_stdout(f):
            sys.argv = [sys.argv[0]]
            cost_model.main()
    write_report(out_dir)
    log(f"report: {os.path.join(out_dir, 'REPORT.md')}")


if __name__ == "__main__":
    main()
