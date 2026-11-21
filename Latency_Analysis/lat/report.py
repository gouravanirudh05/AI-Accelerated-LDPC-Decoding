"""REPORT.md from the per-code JSON files of run_latency.py."""

import glob
import json
import os
import re


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.{nd}f}"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _key(d, B):
    return d.get(str(B), d.get(B))


def region_table(entry, top=None):
    """Merge sync-mode wall time and profiler attribution per region path."""
    sync = entry["sync"]["regions"]
    prof = entry["prof"]["regions"] if "prof" in entry else {}
    paths = sorted(set(sync) | {p for p in prof if p != "(outside regions)"},
                   key=lambda p: -sync.get(p, {}).get("self_ms", 0))
    tot_self = sum(v["self_ms"] for v in sync.values()) or 1.0
    rows = []
    for p in paths[:top]:
        s, r = sync.get(p, {}), prof.get(p, {})
        rows.append([f"`{p}`", _f(s.get("self_ms")), f"{100 * s.get('self_ms', 0) / tot_self:.1f}%",
                     _f(r.get("gpu_ms")), _f(r.get("ops"), 1), _f(r.get("kernels"), 1), _f(r.get("syncs"), 1),
                     _f(s.get("calls"), 0)])
    gpu = sum(v.get("gpu_ms", 0) for v in prof.values())
    kern = sum(v.get("kernels", 0) for v in prof.values())
    ops = sum(v.get("ops", 0) for v in prof.values())
    return rows, dict(sync_total=tot_self, gpu_total=gpu, kernels=kern, ops=ops)


REG_HEAD = ["region", "sync-mode self ms", "share", "GPU kernel ms", "torch ops", "kernels", "host syncs", "calls"]


def group_share(sync_regions, pattern):
    rx = re.compile(pattern)
    return sum(v["self_ms"] for p, v in sync_regions.items() if rx.search(p))


def code_section(r):
    L = [f"## {r['code']} (n {r['n']}, m {r['m']}, E {r['E']}, Eb/N0 {r['ebno']} dB)", ""]
    s = r.get("sgt", {})
    L.append(f"SGT: k = {s.get('k')}, d {s.get('d')}, {s.get('heads')} heads, {s.get('layers')} layers, "
             f"{s.get('stages')} stages x BP({s.get('bp_iters')}); checkpoint: {s.get('trained_from') or 'untrained'}.")
    L.append("")
    if "bp_iters" in r:
        L += ["### BP, fixed iterations: fixed cost + cost per iteration", ""]
        rows = []
        for B, v in r["bp_iters"].items():
            e, g = v["fit"].get("eager", {}), v["fit"].get("graph", {})
            t50e = v["times"].get("eager T=50", {}).get("median_ms")
            t50g = v["times"].get("graph T=50", {}).get("median_ms")
            launch = (1 - t50g / t50e) * 100 if (t50e and t50g) else None
            rows.append([B, _f(e.get("fixed_ms")), _f(e.get("per_iter_ms", 0) * 1e3, 1),
                         _f(g.get("fixed_ms")), _f(g.get("per_iter_ms", 0) * 1e3 if g else None, 1),
                         _f(t50e), _f(t50g), f"{launch:.0f}%" if launch is not None else "-"])
        L.append(_table(["B", "eager fixed ms", "eager us/iter", "graph fixed ms", "graph us/iter",
                         "eager T=50 ms", "graph T=50 ms", "T=50 time that is launch overhead"], rows))
        L.append("")
    if "bp_regions" in r:
        L += ["### BP components", ""]
        for name, e in r["bp_regions"].items():
            rows, tot = region_table(e)
            busy = 100 * tot["gpu_total"] / e["plain_ms"] if e["plain_ms"] else 0
            L.append(f"**{name}**: plain run {_f(e['plain_ms'])} ms, GPU kernels busy {_f(tot['gpu_total'])} ms "
                     f"({busy:.0f}% of the plain run), {_f(tot['ops'], 0)} torch ops, {_f(tot['kernels'], 0)} kernel launches, "
                     f"{e['host_syncs']} host syncs, iterations mean {_f(e['mean_iters'], 1)} / max {e['max_iters']}.")
            L += ["", _table(REG_HEAD, rows), ""]
    if "bp_variants" in r:
        for T, v in r["bp_variants"].items():
            Bs = list(v["times"])
            L += [f"### BP variants, T = {T} (one change at a time; same frames for FER)", ""]
            base = {B: v["times"][B].get("orig bp_decode early-stop", {}).get("median_ms") for B in Bs}
            rows = []
            for n, (fe, it) in v["fer_iters"].items():
                row = [n.strip(), f"{fe:.2e}" if fe == fe else "-", _f(it, 2)]
                for B in Bs:
                    t = v["times"][B].get(n, {}).get("median_ms")
                    row.append(f"{_f(t)} ({base[B] / t:.2f}x)" if (t and base[B] and t == t) else _f(t))
                Bl = Bs[-1]
                t = v["times"][Bl].get(n, {}).get("median_ms")
                row.append(f"{int(Bl) / (t / 1e3):,.0f}" if t and t == t else "-")
                rows.append(row)
            L.append(_table(["variant", "FER", "mean it"] + [f"B={B} ms (speedup)" for B in Bs]
                            + [f"cw/s at B={Bs[-1]}"], rows))
            L.append("")
    if "agg_micro" in r:
        L += ["### Scatter-sum alone (per call; eager = with launch overhead, graph = GPU time)", ""]
        rows = [[v["side"], v["agg"], v["B"], _f(v["eager_us"], 1), _f(v["graph_us"], 1)]
                for v in r["agg_micro"].values()]
        L += [_table(["side", "backend", "B", "eager us", "graph us"], rows), ""]
    if "sgt_refine" in r:
        L += ["### One SGT stage (hard frames)", ""]
        rows = []
        for key, e in r["sgt_refine"].items():
            reg = e["sync"]["regions"]
            bias = group_share(reg, r"(^|/)bias(/|$)")
            att = group_share(reg, r"/attn$")
            ffn = group_share(reg, r"/ffn$")
            feat = group_share(reg, r"(^|/)(features|select|tokens|head)$")
            tot = sum(v["self_ms"] for v in reg.values()) or 1
            rows.append([e["variant"], e["B"], _f(e["plain_ms"]), _f(e["sync"]["wall_ms"]),
                         f"{100 * bias / tot:.0f}%", f"{100 * att / tot:.0f}%", f"{100 * ffn / tot:.0f}%",
                         f"{100 * feat / tot:.0f}%"])
        L += [_table(["variant", "B", "plain ms", "sync-mode ms", "bias build", "attention", "FFN",
                      "features+top-k+tokens+head"], rows), ""]
        for key, e in r["sgt_refine"].items():
            if e["variant"] == "orig":
                rows, tot = region_table(e)
                L.append(f"**orig B={e['B']}**: plain {_f(e['plain_ms'])} ms, GPU kernels {_f(tot['gpu_total'])} ms, "
                         f"{_f(tot['ops'], 0)} torch ops, {_f(tot['kernels'], 0)} kernel launches")
                L += ["", _table(REG_HEAD, rows), ""]
    if "sgt_k" in r:
        L += ["### One SGT stage versus k", ""]
        rows = []
        for e in r["sgt_k"].values():
            reg = e["sync"]["regions"]
            tot = sum(v["self_ms"] for v in reg.values()) or 1
            rows.append([e["k_frac"], e["k"], e["variant"], e["B"], _f(e["sync"]["wall_ms"]),
                         f"{100 * group_share(reg, r'(^|/)bias(/|$)') / tot:.0f}%"])
        L += [_table(["k_frac", "k", "variant", "B", "sync-mode ms", "bias build share"], rows), ""]
    if "sgt_decode" in r:
        L += ["### Full BP-SGT decode (5 BP stages, 4 SGT stages)", ""]
        rows = []
        for key, e in r["sgt_decode"].items():
            reg = e["sync"]["regions"]
            tot = sum(v["self_ms"] for v in reg.values()) or 1
            bp = group_share(reg, r"^stage\d+/bp(/|$)")
            ref = group_share(reg, r"^stage\d+/refine(/|$)")
            bias = group_share(reg, r"/refine/.*bias")
            rest = tot - bp - ref
            rows.append([e["frames"], "early-stop" if e["early_stop"] else "fixed", e["variant"], e["B"],
                         _f(e["plain_ms"]), _f(e["sync"]["wall_ms"]), f"{100 * bp / tot:.0f}%",
                         f"{100 * ref / tot:.0f}% ({100 * bias / tot:.0f}%)", f"{100 * rest / tot:.0f}%",
                         _f(e["mean_bp_iters"], 1), e["host_syncs"], e["active_per_stage"]])
        L += [_table(["frames", "mode", "variant", "B", "plain ms", "sync-mode ms", "BP stages",
                      "SGT stages (bias)", "compaction/sync/writeback", "mean BP it", "host syncs",
                      "active frames per stage"], rows), ""]
    return L


def write_report(out_dir):
    res = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*.json")))]
    if not res:
        return
    env = res[0]["env"]
    L = ["# Latency analysis of BP and BP-SGT (generated by run_latency.py)", "",
         f"Device {env['device']} ({env['gpu']}), torch {env['torch']}, CUDA {env['cuda']}, started {env['started']}"
         f"{', QUICK smoke run: numbers are meaningless' if env['quick'] else ''}. All rows come from one process.",
         "", "How to read the tables: *plain* is the uninstrumented decoder (median wall time with device sync). "
         "*sync-mode* synchronises at every region boundary, so each region's self time includes its launch "
         "overhead but the regions no longer overlap with the CPU; their sum is larger than plain. *GPU kernel "
         "ms* is the kernel time from torch.profiler; when it is much smaller than the sync-mode time the region "
         "is bound by launches and host syncs, not by arithmetic.", ""]
    for r in res:
        L += code_section(r)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
