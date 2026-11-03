"""Merge the final baseline and learned runs into one comparison per code.

Reads every results/final_*/**/*_results.json, combines the decoders of all runs
that evaluated a code (classical entries from the baseline run win on name
clashes), and writes per-code plots / tables plus COMPARISON.md with the
headline best-vs-best table.

Usage: python final/merge_report.py [--out results/final_comparison]
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from common.report import write_report  # noqa: E402

# headline columns: (column title, decoder name in the results)
HEADLINE = [("BP(50)", "BP (50 it)"), ("BP(200)", "BP (200 it)"), ("BP(50)+OSD-2", "BP (50 it) + OSD-2"),
            ("NBP(50)", "NBP per-edge (50 it)"), ("NBP+OSD-2", "NBP per-edge (50 it) + OSD-2"),
            ("GNN", "GNN only (10 it)"), ("BP-GNN 5x10", "BP-GNN (5x10 BP it)"),
            ("BP-GNN 5x40", "BP-GNN (5x40 BP it)"), ("BP-GNN 5x10+OSD-2", "BP-GNN (5x10 BP it) + OSD-2"),
            ("BP-GNN 5x40+OSD-2", "BP-GNN (5x40 BP it) + OSD-2")]


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "results", "final_*", "**", "*_results.json"), recursive=True)):
        if "final_comparison" in p:
            continue
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def merge(runs):
    """{code name: merged results dict in the run_experiment format}."""
    out = {}
    # baseline runs last so their classical entries overwrite the learned runs' BP(50) copy
    for r in sorted(runs, key=lambda r: "baselines" in r["experiment"]):
        for cname, rec in r["codes"].items():
            m = out.setdefault(cname, dict(experiment=f"final_{cname}", device=r["device"], gpu=r.get("gpu"),
                                           models={}, memory={}, codes={cname: dict(rec, ber_bler={}, macs={},
                                                                                    profile={})}))
            mrec = m["codes"][cname]
            learned = set(r["models"])
            m["models"].update(r["models"])
            m["memory"].update(r.get("memory", {}))
            mrec["macs"].update(rec.get("macs", {}))
            for n, v in rec["ber_bler"].items():
                mrec["ber_bler"][n] = v
            for n, v in (rec.get("profile") or {}).items():
                mrec["profile"][n] = v
            if "baselines" in r["experiment"]:
                mrec["idle_gpu_w"] = rec.get("idle_gpu_w")
                mrec["prof_ebno"] = rec.get("prof_ebno")
            if learned:
                mrec.setdefault("learned_runs", []).append(r["experiment"])
    # order: classical first, then learned
    for m in out.values():
        rec = next(iter(m["codes"].values()))
        is_learned = lambda n: n.split(" + OSD")[0] in m["models"]
        rec["ber_bler"] = dict(sorted(rec["ber_bler"].items(), key=lambda kv: (is_learned(kv[0]), kv[0])))
        if not m["memory"]:
            del m["memory"]
    return out


def headline(merged):
    L = ["# Final comparison: best tuned version of every decoder", "",
         "BLER at the two highest Eb/N0 points where BP(200) still has >= 20 frame errors "
         "(deep Monte Carlo, >= 100 frame errors or up to 0.5-3 M frames per point; '-' = not run).", ""]
    L.append("code | Eb/N0 | " + " | ".join(t for t, _ in HEADLINE))
    L.append("---|---|" + "---|" * len(HEADLINE))
    for cname in sorted(merged):
        rec = next(iter(merged[cname]["codes"].values()))
        bb = rec["ber_bler"]
        ref = [r for r in bb.get("BP (200 it)", []) if r["frame_errors"] >= 20]
        for e in sorted(r["ebno"] for r in ref)[-2:]:
            cells = []
            for _, n in HEADLINE:
                d = {r["ebno"]: r for r in bb.get(n, [])}
                cells.append(f"{d[e]['bler']:.2e}" if e in d else "-")
            L.append(f"{cname} | {e:.2f} | " + " | ".join(cells))
    L += ["", "## Energy and latency (early-stop decoding, profiling Eb/N0 of each code)", "",
          "code | decoder | GPU lat. B=1 [ms] | throughput large B [cw/s] | energy/cw [uJ] | dyn. energy/cw [uJ]",
          "---|---|---|---|---|---"]
    for cname in sorted(merged):
        rec = next(iter(merged[cname]["codes"].values()))
        for _, n in HEADLINE:
            p = rec["profile"].get(f"{n} [early-stop]")
            if not p or not p.get("gpu"):
                continue
            g = list(p["gpu"].items())
            e = p.get("gpu_energy", {})
            nan = float("nan")
            L.append(f"{cname} | {n} | {g[0][1]['median_ms']:.3f} | {g[-1][1]['throughput_cw_s']:.0f} | "
                     f"{e.get('energy_per_cw_uj', nan):.1f} | {e.get('dynamic_energy_per_cw_uj', nan):.1f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "final_comparison"))
    args = ap.parse_args()
    merged = merge(load_runs(ROOT))
    os.makedirs(args.out, exist_ok=True)
    parts = [headline(merged)]
    for cname in sorted(merged):
        with open(os.path.join(args.out, f"final_{cname}_results.json"), "w") as f:
            json.dump(merged[cname], f, indent=1)
        parts.append(write_report(merged[cname], args.out))
    with open(os.path.join(args.out, "COMPARISON.md"), "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print(parts[0])


if __name__ == "__main__":
    main()
