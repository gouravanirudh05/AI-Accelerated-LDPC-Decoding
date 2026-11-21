"""Task 1 report: BLER and BER of every BP schedule against BP with the same budget.

Reads traces_<code>.json (run_schedules.py) and writes into --out (default: next to
the traces):
  bler_<code>.png       BLER vs Eb/N0, every schedule and the BP baselines
  SCHEDULES.md          per code: BLER / BER / average BP iterations / average SGT
                        calls per Eb/N0, and the BLER ratio of every schedule against
                        (a) BP with the same total budget and (b) the trained 5x10

Usage: python plot_schedules.py --traces <dir or file> [--out dir] [--ref "SGT 5x10"]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plots  # noqa: E402
from plots import fmt, plt  # noqa: E402

MIN_ERR = 20  # ratios only where both decoders have at least this many frame errors


def ratio(a, b):
    if a["frame_errors"] < MIN_ERR or b["frame_errors"] < MIN_ERR or b["bler"] == 0:
        return None
    return a["bler"] / b["bler"]


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def code_report(name, t, out, ref):
    decs = t["decoders"]
    labels = list(decs)
    st = plots.styles(labels, decs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab in labels:
        pts = [p for p in decs[lab]["points"] if p["frame_errors"] > 0]
        for ax, key in zip(axes, ("bler", "ber")):
            ax.semilogy([p["ebno"] for p in pts], [p[key] for p in pts], label=lab, **st[lab])
    for ax, key in zip(axes, ("BLER", "BER")):
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel(key)
    axes[0].set_title(f"{name} (n={t['code']['n']}): BLER vs Eb/N0", loc="left", fontsize=9)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    png = plots.save(fig, out, f"bler_{name}.png")

    ebno = t["ebno"]
    md = [f"## {name}  (n={t['code']['n']}, k={t['code']['k']})", "",
          f"![BLER]({os.path.basename(png)})", "",
          "BLER (frame errors in brackets) | " + " | ".join(f"{e} dB" for e in ebno),
          "---|" + "---|" * len(ebno)]
    for lab in labels:
        by = {p["ebno"]: p for p in decs[lab]["points"]}
        md.append(f"{lab} | " + " | ".join(
            f"{fmt(by[e]['bler'])} ({by[e]['frame_errors']})" if e in by else "-" for e in ebno))
    md += ["", "Average BP iterations / average SGT calls per frame | " + " | ".join(f"{e} dB" for e in ebno),
           "---|" + "---|" * len(ebno)]
    for lab in labels:
        by = {p["ebno"]: p for p in decs[lab]["points"]}
        md.append(f"{lab} | " + " | ".join(
            f"{by[e]['avg_iters']:.2f} / {by[e]['avg_sgt_calls']:.3f}" if e in by else "-" for e in ebno))
    # ratios: vs BP of the same budget, vs the reference schedule
    bp = {d["budget"]: lab for lab, d in decs.items() if d["kind"] == "bp"}
    md += ["", f"BLER ratio (< 1 = schedule better; only points with >= {MIN_ERR} errors on both sides; "
               "geometric mean over those points)", "",
           f"schedule | total BP it | SGT stages | vs BP (same budget) | vs {ref}", "---|---|---|---|---"]
    summary = {}
    for lab in labels:
        d = decs[lab]
        if d["kind"] != "sgt":
            continue
        mine = {p["ebno"]: p for p in d["points"]}
        r_bp = r_ref = None
        if d["budget"] in bp:
            other = {p["ebno"]: p for p in decs[bp[d["budget"]]]["points"]}
            r_bp = geo([ratio(mine[e], other[e]) for e in mine if e in other])
        if ref in decs and lab != ref:
            other = {p["ebno"]: p for p in decs[ref]["points"]}
            r_ref = geo([ratio(mine[e], other[e]) for e in mine if e in other])
        summary[lab] = (r_bp, r_ref)
        md.append(f"{lab} | {d['budget']} | {len(d['schedule']) - 1} | "
                  f"{'-' if r_bp is None else f'{r_bp:.2f}'} | {'-' if r_ref is None else f'{r_ref:.2f}'}")
    return "\n".join(md), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="SGT 5x10", help="schedule the others are compared with")
    args = ap.parse_args()
    out = args.out or (args.traces if os.path.isdir(args.traces) else os.path.dirname(args.traces))
    traces = plots.load_traces(args.traces)
    parts, summ = [], {}
    for name, t in traces.items():
        md, s = code_report(name, t, out, args.ref)
        parts.append(md)
        summ[name] = s
    labels = list(dict.fromkeys(lab for s in summ.values() for lab in s))
    head = ["# Task 1: BP iteration schedules in the interleaved BP+SGT pipeline", "",
            "Inference-only: every schedule uses the final BP-SGT weights trained as 5x10. Notation RxT = R BP runs "
            "of T iterations with R-1 SGT stages between them; a-b-c = runs of a, b, c iterations.", "",
            "Geometric-mean BLER ratio per code (vs BP with the same total iterations / vs " + args.ref + ")", "",
            "code | " + " | ".join(labels), "---|" + "---|" * len(labels)]
    for name, s in summ.items():
        head.append(f"{name} | " + " | ".join(
            "-" if lab not in s else " / ".join("-" if v is None else f"{v:.2f}" for v in s[lab]) for lab in labels))
    with open(os.path.join(out, "SCHEDULES.md"), "w") as f:
        f.write("\n".join(head) + "\n\n" + "\n\n".join(parts) + "\n")
    print(f"wrote {os.path.join(out, 'SCHEDULES.md')}")


if __name__ == "__main__":
    main()
