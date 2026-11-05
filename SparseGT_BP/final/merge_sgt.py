"""Put the final BP-SGT runs next to the BP-GNN final comparison.

Reads the merged per-code results of the BP-GNN project
(Nvidia_GNN+BP(lLatest)/results/final_comparison/final_<code>_results.json:
classical baselines, NBP, GNN, BP-GNN) and every final_sgt_*_results.json under
results/, adds the SGT decoders (the BP(50) copy of the SGT session is kept
only as "BP (50 it) [SGT session]" in the JSON), and writes per-code plots /
tables plus COMPARISON.md to results/final_comparison/.

Usage: python final/merge_sgt.py
"""

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sgt import NV_ROOT  # noqa: E402
from common.report import write_report  # noqa: E402

RES = os.path.join(HERE, "..", "results")
BASE = [("BP(50)", "BP (50 it)"), ("BP(200)", "BP (200 it)"), ("BP(50)+OSD-2", "BP (50 it) + OSD-2"),
        ("NBP(50)", "NBP per-edge (50 it)"), ("BP-GNN 5x10", "BP-GNN (5x10 BP it)"),
        ("BP-GNN 5x40", "BP-GNN (5x40 BP it)"), ("BP-GNN 5x40+OSD-2", "BP-GNN (5x40 BP it) + OSD-2")]
BUDGETS = [("5x10", "(5x10 BP it)"), ("5x40", "(5x40 BP it)"), ("5x10+OSD-2", "(5x10 BP it) + OSD-2"),
           ("5x40+OSD-2", "(5x40 BP it) + OSD-2")]


def sgt_variants(merged):
    """SGT model names present, e.g. ["SGT", "SGT-warm"] (decoders "<name> (5x10 BP it)")."""
    names = {n.split(" (")[0] for m in merged.values() for n in m["models"] if n.startswith("SGT")}
    return sorted(names, key=lambda n: (len(n), n))


def columns(merged):
    return BASE + [(f"{v} {b}", f"{v} {suffix}") for v in sgt_variants(merged) for b, suffix in BUDGETS]


def merge():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, "**", "final_sgt_*_results.json"), recursive=True)):
        if "final_comparison" in p:
            continue
        with open(p) as f:
            r = json.load(f)
        for cname, rec in r["codes"].items():
            if cname not in out:
                with open(os.path.join(NV_ROOT, "results", "final_comparison", f"final_{cname}_results.json")) as f:
                    out[cname] = json.load(f)
            m = out[cname]
            mrec = m["codes"][cname]
            m["models"].update(r["models"])
            m.setdefault("memory", {}).update(r.get("memory", {}))
            mrec["macs"].update(rec.get("macs", {}))
            for n, v in rec["ber_bler"].items():
                mrec["ber_bler"][n if n.startswith("SGT") else f"{n} [SGT session]"] = v
            for n, v in (rec.get("profile") or {}).items():
                if n.startswith("SGT"):
                    mrec["profile"][n] = v
            mrec.setdefault("learned_runs", []).append(r["experiment"])
    # clean single-process profiles (final/profile_sgt.py) replace the final runs' SGT profiles,
    # whose NVML readings came from the wrong GPU; their BP(50) is kept as a same-session reference
    for p in sorted(glob.glob(os.path.join(RES, "kaggle_profile_*", "**", "profile_*.json"), recursive=True)):
        with open(p) as f:
            r = json.load(f)
        if r["code"] not in out:
            continue
        mrec = out[r["code"]]["codes"][r["code"]]
        for n, v in r["profile"].items():
            mrec["profile"][n if n.startswith("SGT") else n.replace("BP (50 it)", "BP (50 it) [SGT session]")] = v
        mrec["sgt_profile"] = dict(source=os.path.relpath(p, RES), idle_gpu_w=r["idle_gpu_w"], gpu=r["gpu"])
    return out


def headline(merged):
    L = ["# Final comparison: BP-SGT (sparse graph transformer) vs BP-GNN (NVIDIA) and baselines", "",
         "BLER at the two highest Eb/N0 points where BP(200) still has >= 20 frame errors "
         "(deep Monte Carlo, >= 100 frame errors or up to 0.5-3 M frames per point; '-' = not run). "
         "Baselines, NBP and BP-GNN from Nvidia_GNN+BP(lLatest)/results/final_comparison.", ""]
    HEADLINE = columns(merged)
    PAIRS = [(f"{v} vs BP-GNN, {b}", f"BP-GNN {suffix}", f"{v} {suffix}")
             for v in sgt_variants(merged) for b, suffix in BUDGETS]
    PAIRS += [(f"{v} 5x10 vs BP(50)", "BP (50 it)", f"{v} (5x10 BP it)") for v in sgt_variants(merged)]
    PAIRS += [(f"{v} 5x40 vs BP(200)", "BP (200 it)", f"{v} (5x40 BP it)") for v in sgt_variants(merged)]
    L.append("code | Eb/N0 | " + " | ".join(t for t, _ in HEADLINE))
    L.append("---|---|" + "---|" * len(HEADLINE))
    ratios = {p[0]: [] for p in PAIRS}
    for cname in sorted(merged):
        bb = merged[cname]["codes"][cname]["ber_bler"]
        ref = [r for r in bb.get("BP (200 it)", []) if r["frame_errors"] >= 20]
        for e in sorted(r["ebno"] for r in ref)[-2:]:
            cells = []
            for _, n in HEADLINE:
                d = {r["ebno"]: r for r in bb.get(n, [])}
                cells.append(f"{d[e]['bler']:.2e}" if e in d else "-")
            L.append(f"{cname} | {e:.2f} | " + " | ".join(cells))
            for tag, ref_name, sgt in PAIRS:
                g = {r["ebno"]: r for r in bb.get(ref_name, [])}
                s = {r["ebno"]: r for r in bb.get(sgt, [])}
                if e in g and e in s and g[e]["frame_errors"] >= 5:
                    # None: SGT made no frame error at this point (ratio not defined, counted as better)
                    ratios[tag].append((cname, e, s[e]["bler"] / g[e]["bler"] if s[e]["frame_errors"] else None))
    L += ["", "## Head to head at equal BP budget (BLER ratio SGT / reference at the headline points; "
          "< 1 = SGT better; points where SGT made no frame error count as better and are left out of the mean)",
          "", "comparison | points | SGT better | geometric-mean ratio", "---|---|---|---"]
    for tag, rs in ratios.items():
        if rs:
            fin = [r for _, _, r in rs if r is not None]
            gm = math.exp(sum(map(math.log, fin)) / len(fin)) if fin else float("nan")
            L.append(f"{tag} | {len(rs)} | {sum(r is None or r < 1 for _, _, r in rs)} | {gm:.3f}")
    models = ["BP-GNN"] + sgt_variants(merged)
    L += ["", "## Complexity of one refinement stage (per codeword) and model size", "",
          "code | " + " | ".join(f"{v} MACs" for v in models) + " | " + " | ".join(f"{v} params" for v in models),
          "---|" + "---|" * (2 * len(models))]
    for cname in sorted(merged):
        m = merged[cname]
        macs = m["codes"][cname]["macs"]
        L.append(f"{cname} | " + " | ".join(f"{macs.get(f'{v} (5x10 BP it)', 0):,}" for v in models) + " | "
                 + " | ".join(str(m["models"].get(f"{v} (5x10 BP it)", {}).get("params", "-")) for v in models))
    L += ["", "## Energy and latency (early-stop decoding, profiling Eb/N0 of each code, Tesla T4)", "",
          "Baselines and BP-GNN: profiled in the BP-GNN project's sessions. SGT: profiled alone on one GPU "
          "(final/profile_sgt.py); 'BP (50 it) [SGT session]' is BP(50) re-profiled in that same session, "
          "to show the session-to-session spread.", "",
          "code | decoder | GPU lat. B=1 [ms] | throughput large B [cw/s] | energy/cw [uJ] | dyn. energy/cw [uJ]",
          "---|---|---|---|---|---"]
    nan = float("nan")
    for cname in sorted(merged):
        prof = merged[cname]["codes"][cname]["profile"]
        for _, n in HEADLINE[:1] + [("", "BP (50 it) [SGT session]")] + HEADLINE[1:]:
            p = prof.get(f"{n} [early-stop]")
            if not p or not p.get("gpu"):
                continue
            g = list(p["gpu"].items())
            e = p.get("gpu_energy", {})
            L.append(f"{cname} | {n} | {g[0][1]['median_ms']:.3f} | {g[-1][1]['throughput_cw_s']:.0f} | "
                     f"{e.get('energy_per_cw_uj', nan):.1f} | {e.get('dynamic_energy_per_cw_uj', nan):.1f}")
    L += ["", "## Cost relative to BP(50) of the same profiling session (removes the machine-to-machine spread)", "",
          "Large-batch throughput and energy per codeword divided by those of BP(50) profiled on the same "
          "Kaggle machine (BP-GNN: BP-GNN project sessions; SGT: final/profile_sgt.py sessions). "
          "Throughput < 1 = slower than BP(50); energy > 1 = more energy than BP(50).", "",
          "code | BP-GNN 5x10 thr. | SGT 5x10 thr. | BP-GNN 5x40 thr. | SGT 5x40 thr. | "
          "BP-GNN 5x10 energy | SGT 5x10 energy | BP-GNN 5x40 energy | SGT 5x40 energy", "---|" + "---|" * 8]
    for cname in sorted(merged):
        prof = merged[cname]["codes"][cname]["profile"]

        def metric(n, ref, f):
            a, b = prof.get(f"{n} [early-stop]"), prof.get(f"{ref} [early-stop]")
            if not a or not b or not a.get("gpu") or not b.get("gpu"):
                return "-"
            return f"{f(a) / f(b):.2f}"
        thr = lambda p: list(p["gpu"].values())[-1]["throughput_cw_s"]
        en = lambda p: p.get("gpu_energy", {}).get("energy_per_cw_uj", nan)
        cells = []
        for f in (thr, en):
            for b in ("5x10", "5x40"):
                cells += [metric(f"BP-GNN ({b} BP it)", "BP (50 it)", f),
                          metric(f"SGT ({b} BP it)", "BP (50 it) [SGT session]", f)]
        L.append(f"{cname} | " + " | ".join(cells))
    return "\n".join(L)


def main():
    merged = merge()
    out = os.path.join(RES, "final_comparison")
    os.makedirs(out, exist_ok=True)
    parts = [headline(merged)]
    for cname in sorted(merged):
        merged[cname]["experiment"] = f"final_{cname}"
        with open(os.path.join(out, f"final_{cname}_results.json"), "w") as f:
            json.dump(merged[cname], f, indent=1)
        parts.append(write_report(merged[cname], out))
    with open(os.path.join(out, "COMPARISON.md"), "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print(parts[0])


if __name__ == "__main__":
    main()
