"""Task 4: average total number of BP iterations performed.

Total BP iterations of a frame = sum over the BP runs it went through of the
iterations run in each (early stopping inside a run, and no further runs after the
syndrome is satisfied). Reported per decoder and Eb/N0:

  mean and standard deviation over all frames
  mean over the frames that BP run 1 did not solve (the ones that reached the SGT)
  mean number of SGT calls per frame
  budget = the maximum (a frame that never converges)

Reads traces_<code>.json (task 1 / task 5 runners) and writes
avg_iters_<code>.png and AVG_ITERATIONS.md (plus a cross-code table at each code's
middle Eb/N0 point).

Usage: python avg_iterations.py --traces <dir or file> [--out dir]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plots  # noqa: E402
from plots import plt  # noqa: E402


def stats(d, p):
    """dict of the task-4 numbers of one decoder at one Eb/N0 point."""
    nf = p["frames"]
    h = np.asarray(p["iter_hist"], float)
    first = d["schedule"][0]
    # frames that reached run 2: converged after more than `first` iterations, or never converged
    later = h[first + 1:].sum() + p["never"]
    later_iters = (np.arange(len(h))[first + 1:] * h[first + 1:]).sum() + p["never"] * d["budget"]
    return dict(mean=p["avg_iters"], std=p["std_iters"], calls=p["avg_sgt_calls"], frames=nf,
                frac_later=later / nf, mean_later=(later_iters / later) if later else None,
                budget=d["budget"])


def code_part(name, t, out):
    decs = t["decoders"]
    labs = list(decs)
    st = plots.styles(labs, decs)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for lab in labs:
        pts = decs[lab]["points"]
        ax.plot([p["ebno"] for p in pts], [p["avg_iters"] for p in pts], label=lab, **st[lab])
    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("average total BP iterations per frame")
    ax.set_title(f"{name}: average BP iterations (early stopping)", loc="left", fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    png = plots.save(fig, out, f"avg_iters_{name}.png")

    md = [f"## {name}", "", f"![avg iterations]({os.path.basename(png)})", "",
          "decoder | budget | Eb/N0 | mean it (std) | frames not solved by run 1 | mean it of those | mean SGT calls",
          "---|---|---|---|---|---|---"]
    for lab in labs:
        d = decs[lab]
        for p in d["points"]:
            s = stats(d, p)
            ml = "-" if s["mean_later"] is None else f"{s['mean_later']:.1f}"
            md.append(f"{lab} | {s['budget']} | {p['ebno']:g} | {s['mean']:.3f} ({s['std']:.2f}) | "
                      f"{plots.fmt(s['frac_later'])} | {ml} | {s['calls']:.4f}")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (args.traces if os.path.isdir(args.traces) else os.path.dirname(args.traces))
    traces = plots.load_traces(args.traces)
    labs = list(dict.fromkeys(lab for t in traces.values() for lab in t["decoders"]))
    head = ["# Task 4: average total number of BP iterations", "",
            "Cross-code summary at the middle Eb/N0 point of each code: mean total BP iterations per frame "
            "(mean SGT calls per frame in brackets).", "",
            "code | Eb/N0 | " + " | ".join(labs), "---|---|" + "---|" * len(labs)]
    parts = []
    for name, t in traces.items():
        e = t["ebno"][len(t["ebno"]) // 2]
        cells = []
        for lab in labs:
            d = t["decoders"].get(lab)
            p = d and next((q for q in d["points"] if abs(q["ebno"] - e) < 1e-9), None)
            cells.append("-" if not p else f"{p['avg_iters']:.2f} ({p['avg_sgt_calls']:.3f})")
        head.append(f"{name} | {e:g} | " + " | ".join(cells))
        parts.append(code_part(name, t, out))
    with open(os.path.join(out, "AVG_ITERATIONS.md"), "w") as f:
        f.write("\n".join(head) + "\n\n" + "\n\n".join(parts) + "\n")
    print(f"wrote {os.path.join(out, 'AVG_ITERATIONS.md')}")


if __name__ == "__main__":
    main()
