"""Put the final BP-PGNN runs next to BP, BP-GNN and BP-SGT.

Reads the merged per-code results of the BP-SGT project
(SparseGT_BP/results/final_comparison/final_<code>_results.json: classical
baselines, NBP, GNN, BP-GNN and SGT) and every final_pgnn_*_results.json under
PooledGNN_BP/results/ (Kaggle outputs copied there), adds the PGNN decoders
(the BP(50) copy of the PGNN session is kept as "BP (50 it) [PGNN session]"),
and writes per-code plots / tables plus COMPARISON.md to
results/final_comparison/.

Comparison is at equal BP iterations (5x10 vs BP(50), 5x40 vs BP(200)), as
chosen by Gourav. Training time differs on codes with n >= 512 (PGNN 160 min,
BP-GNN / BP-SGT 80 min); the table marks those rows.

Usage: python final/merge_pgnn.py
"""

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgnn import SGT_ROOT  # noqa: E402
from common.report import write_report  # noqa: E402

RES = os.path.join(HERE, "..", "results")
BASE = [("BP(50)", "BP (50 it)"), ("BP(200)", "BP (200 it)"),
        ("BP-GNN 5x10", "BP-GNN (5x10 BP it)"), ("BP-GNN 5x40", "BP-GNN (5x40 BP it)"),
        ("SGT 5x10", "SGT (5x10 BP it)"), ("SGT 5x40", "SGT (5x40 BP it)")]
BUDGETS = [("5x10", "(5x10 BP it)"), ("5x40", "(5x40 BP it)")]
REFS = {"5x10": [("BP(50)", "BP (50 it)"), ("BP-GNN", "BP-GNN (5x10 BP it)"), ("SGT", "SGT (5x10 BP it)")],
        "5x40": [("BP(200)", "BP (200 it)"), ("BP-GNN", "BP-GNN (5x40 BP it)"), ("SGT", "SGT (5x40 BP it)")]}


def variants(merged):
    names = {n.split(" (")[0] for m in merged.values() for n in m["models"] if n.startswith("PGNN")}
    return sorted(names, key=lambda n: (len(n), n))


def merge():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, "**", "final_pgnn_*_results.json"), recursive=True)):
        if "final_comparison" in p:
            continue
        with open(p) as f:
            r = json.load(f)
        for cname, rec in r["codes"].items():
            if cname not in out:
                src = os.path.join(SGT_ROOT, "results", "final_comparison", f"final_{cname}_results.json")
                with open(src) as f:
                    out[cname] = json.load(f)
            m = out[cname]
            mrec = m["codes"][cname]
            m["models"].update(r["models"])
            m.setdefault("memory", {}).update(r.get("memory", {}))
            mrec["macs"].update(rec.get("macs", {}))
            for n, v in rec["ber_bler"].items():
                mrec["ber_bler"][n if n.startswith("PGNN") else f"{n} [PGNN session]"] = v
            for n, v in (rec.get("profile") or {}).items():
                mrec.setdefault("profile", {})[n if n.startswith("PGNN") else
                                               n.replace("BP (50 it)", "BP (50 it) [PGNN session]")] = v
            mrec.setdefault("learned_runs", []).append(r["experiment"])
    return out


def train_minutes(m, prefix):
    for n, info in m["models"].items():
        if n.startswith(prefix) and "train_minutes" in info and "variant_of" not in info:
            return info["train_minutes"]
    return None


def headline(merged):
    vs = variants(merged)
    cols = BASE + [(f"{v} {b}", f"{v} {s}") for v in vs for b, s in BUDGETS]
    L = ["# Final comparison: BP-PGNN (pooled bipartite GNN) vs BP, BP-GNN and BP-SGT", "",
         "BLER at the two highest Eb/N0 points where BP(200) still has >= 20 frame errors "
         "(>= 100 frame errors or up to 0.5-3 M frames per point; '-' = not run). Equal BP iterations: "
         "5x10 against BP(50), 5x40 against BP(200). Train min = minutes of training of PGNN / BP-GNN / "
         "BP-SGT on that code; rows marked * gave PGNN twice the training time (not like-for-like).", ""]
    L.append("code | Eb/N0 | train min | " + " | ".join(t for t, _ in cols))
    L.append("---|---|---|" + "---|" * len(cols))
    pairs = [(f"{v} {b} vs {rn}", rd, f"{v} {s}") for v in vs for b, s in BUDGETS for rn, rd in REFS[b]]
    ratios = {p[0]: [] for p in pairs}
    for cname in sorted(merged):
        m = merged[cname]
        bb = m["codes"][cname]["ber_bler"]
        tm = [train_minutes(m, p) for p in (vs[0] if vs else "PGNN", "BP-GNN", "SGT")]
        star = "*" if tm[0] and tm[1] and tm[0] > 1.2 * tm[1] else ""
        tms = "/".join("-" if t is None else f"{t:.0f}" for t in tm) + star
        ref = [r for r in bb.get("BP (200 it)", []) if r["frame_errors"] >= 20]
        for e in sorted(r["ebno"] for r in ref)[-2:]:
            cells = []
            for _, n in cols:
                d = {r["ebno"]: r for r in bb.get(n, [])}
                cells.append(f"{d[e]['bler']:.2e}" if e in d else "-")
            L.append(f"{cname}{star} | {e:.2f} | {tms} | " + " | ".join(cells))
            for tag, rname, pname in pairs:
                g = {r["ebno"]: r for r in bb.get(rname, [])}
                s = {r["ebno"]: r for r in bb.get(pname, [])}
                if e in g and e in s and g[e]["frame_errors"] >= 5:
                    ratios[tag].append((cname, e, s[e]["bler"] / g[e]["bler"] if s[e]["frame_errors"] else None))
    L += ["", "## Head to head at equal BP iterations (BLER ratio PGNN / reference at the headline points; "
          "< 1 = PGNN better; noise: about 25 % on one code, 12 % on the mean over codes)", "",
          "comparison | points | PGNN better | geometric-mean ratio | geo-mean, codes n >= 512 only", "---|---|---|---|---"]
    big = {c for c in merged if merged[c]["codes"][c]["code"]["n"] >= 512}
    gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
    for tag, rs in ratios.items():
        if rs:
            fin = [r for _, _, r in rs if r is not None]
            finb = [r for c, _, r in rs if r is not None and c in big]
            L.append(f"{tag} | {len(rs)} | {sum(r is None or r < 1 for _, _, r in rs)} | {gm(fin):.3f} | {gm(finb):.3f}")
    models = ["BP-GNN", "SGT"] + vs
    L += ["", "## Complexity of one refinement stage (MACs per codeword) and model size", "",
          "code | " + " | ".join(f"{v} MACs" for v in models) + " | " + " | ".join(f"{v} params" for v in models),
          "---|" + "---|" * (2 * len(models))]
    for cname in sorted(merged):
        m = merged[cname]
        macs = m["codes"][cname]["macs"]
        L.append(f"{cname} | " + " | ".join(f"{macs.get(f'{v} (5x10 BP it)', 0):,}" for v in models) + " | "
                 + " | ".join(str(m["models"].get(f"{v} (5x10 BP it)", {}).get("params", "-")) for v in models))
    return "\n".join(L)


def main():
    merged = merge()
    if not merged:
        print(f"no final_pgnn_*_results.json under {RES}")
        return
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
