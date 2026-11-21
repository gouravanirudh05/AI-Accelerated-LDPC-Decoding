"""Task 2: distribution of the stopping points across the stages of the BP+SGT pipeline.

A frame "stops" in BP run r (r = 1 is the first BP run, before any SGT; run r+1
follows the r-th SGT stage) at the first iteration whose hard decision satisfies
every parity check. Frames that never satisfy the syndrome use the full budget
(detected failure); frames that satisfy it with a wrong codeword are undetected
errors. Reads traces_<code>.json (task1_bp_schedules/run_schedules.py or
task5_single_vs_interleaved/run_single_vs_interleaved.py) and writes per code:

  stop_runs_<code>.png   per SGT schedule, stacked bars per Eb/N0:
                         left  = all frames (stopped in run 1, 2, ..., wrong codeword, never)
                         right = only the frames BP run 1 did not solve, i.e. the frames
                                 the SGT acted on: in which later run did they stop?
  stop_iters_<code>.png  histogram of the total BP iterations at the stopping point
                         (log y), one panel per schedule, at --ebno (default: the
                         middle point), dashed lines = SGT stages
  STOPPING.md            the same numbers as tables

Usage: python plot_stopping_points.py --traces <dir or file> [--out dir] [--ebno 3.5]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plots  # noqa: E402
from plots import NEVER, RUN_RAMP, WRONG, plt  # noqa: E402

MAX_BINS = 6  # runs 1..5 get their own colour, later runs are pooled as "run 6+"


def buckets(runs):
    n = min(runs, MAX_BINS)
    names = [f"run {i + 1}" for i in range(n)]
    if runs > MAX_BINS:
        names[-1] = f"runs {MAX_BINS}-{runs}"
    return names


def pooled(vec):
    v = np.asarray(vec, float)
    if len(v) <= MAX_BINS:
        return v
    return np.concatenate([v[:MAX_BINS - 1], [v[MAX_BINS - 1:].sum()]])


def run_figure(name, t, out):
    sgt = [lab for lab, d in t["decoders"].items() if d["kind"] == "sgt"]
    if not sgt:
        return None
    fig, axes = plt.subplots(len(sgt), 2, figsize=(10, 1.9 * len(sgt) + 0.6), squeeze=False)
    for row, lab in enumerate(sgt):
        d = t["decoders"][lab]
        runs = len(d["schedule"])
        names = buckets(runs)
        cols = [RUN_RAMP(i / max(1, len(names) - 1)) for i in range(len(names))]
        pts = d["points"]
        x = np.arange(len(pts))
        for col, cond in enumerate((False, True)):
            ax = axes[row, col]
            bottom = np.zeros(len(pts))
            stacks = []
            for p in pts:
                nf = p["frames"]
                ok = pooled(p["run_ok"]) / nf
                wrong = pooled(p["run_wrong"]).sum() / nf
                never = p["never"] / nf
                vec = np.concatenate([ok, [wrong, never]])
                if cond:
                    vec[0] = 0.0
                    s = vec.sum()
                    vec = vec / s if s > 0 else vec
                stacks.append(vec)
            stacks = np.array(stacks)
            for k in range(stacks.shape[1]):
                c = cols[k] if k < len(names) else (WRONG if k == len(names) else NEVER)
                lbl = names[k] if k < len(names) else ("wrong codeword" if k == len(names) else "never converged")
                ax.bar(x, stacks[:, k], bottom=bottom, color=c, width=0.7, label=lbl,
                       edgecolor="white", linewidth=1, hatch="//" if k == len(names) else None)
                bottom += stacks[:, k]
            if cond:  # number of frames each conditional bar is based on
                ticks = [f"{p['ebno']:g}\nn={p['frames'] - p['run_ok'][0] - p['run_wrong'][0]}" for p in pts]
            else:
                ticks = [f"{p['ebno']:g}" for p in pts]
            ax.set_xticks(x, ticks)
            ax.set_ylim(0, 1)
            ax.grid(axis="x", visible=False)
            if row == len(sgt) - 1:
                ax.set_xlabel("Eb/N0 [dB]")
            ttl = (f"{lab}: frames BP run 1 did not solve" if cond else f"{lab}: all frames")
            ax.set_title(ttl, loc="left", fontsize=9)
            ax.set_ylabel("fraction")
        axes[row, 1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(f"{name}: BP run in which each frame stopped (right: n = frames that reached the SGT)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return plots.save(fig, out, f"stop_runs_{name}.png")


def iter_figure(name, t, out, ebno):
    decs = t["decoders"]
    labs = list(decs)
    e = ebno if ebno is not None else t["ebno"][len(t["ebno"]) // 2]
    ncol = 2
    nrow = (len(labs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.0 * nrow + 0.4), squeeze=False)
    st = plots.styles(labs, decs)
    for i, lab in enumerate(labs):
        ax = axes[i // ncol, i % ncol]
        d = decs[lab]
        p = next((q for q in d["points"] if abs(q["ebno"] - e) < 1e-9), None)
        if p is None:
            ax.set_visible(False)
            continue
        h = np.asarray(p["iter_hist"], float) / p["frames"]
        it = np.arange(len(h))
        nz = h > 0
        ax.bar(it[nz], h[nz], width=1.0, color=st[lab]["color"], edgecolor="none")
        if p["never"]:
            ax.bar([d["budget"] + 1.5], [p["never"] / p["frames"]], width=1.0, color=NEVER, hatch="//",
                   edgecolor="white")
        for b in np.cumsum(d["schedule"])[:-1]:
            ax.axvline(b + 0.5, color="#7a7a76", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlim(0, d["budget"] + 3)
        ax.set_title(f"{lab}  (mean {p['avg_iters']:.2f} it, never conv. {p['never'] / p['frames']:.1e})",
                     loc="left", fontsize=9)
        ax.set_ylabel("fraction of frames")
        ax.set_xlabel("total BP iterations at stop")
    for j in range(len(labs), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    fig.suptitle(f"{name} at {e:g} dB: total BP iterations when the syndrome was satisfied "
                 f"(dotted = SGT stage, hatched grey at the right = never converged)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    return plots.save(fig, out, f"stop_iters_{name}.png")


def tables(name, t):
    md = [f"## {name}", "",
          "Fraction of all frames that stopped in each BP run (run 1 = before any SGT), "
          "then wrong codeword / never converged", ""]
    for lab, d in t["decoders"].items():
        runs = len(d["schedule"])
        if runs == 1:
            continue
        names = buckets(runs)
        md += [f"**{lab}** (schedule {d['schedule']})", "",
               "Eb/N0 | frames | " + " | ".join(names) + " | wrong codeword | never | mean SGT calls",
               "---|" * (len(names) + 5)]
        for p in d["points"]:
            nf = p["frames"]
            ok = pooled(p["run_ok"]) / nf
            wr = sum(p["run_wrong"]) / nf
            md.append(f"{p['ebno']:g} | {nf} | " + " | ".join(plots.fmt(v) for v in ok)
                      + f" | {plots.fmt(wr)} | {plots.fmt(p['never'] / nf)} | {p['avg_sgt_calls']:.4f}")
        md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ebno", type=float, default=None, help="Eb/N0 of the iteration histograms")
    args = ap.parse_args()
    out = args.out or (args.traces if os.path.isdir(args.traces) else os.path.dirname(args.traces))
    parts = ["# Task 2: stopping points across the stages of the BP+SGT pipeline", ""]
    for name, t in plots.load_traces(args.traces).items():
        a = run_figure(name, t, out)
        b = iter_figure(name, t, out, args.ebno)
        parts += [tables(name, t)] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in (a, b) if p] + [""]
    with open(os.path.join(out, "STOPPING.md"), "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {os.path.join(out, 'STOPPING.md')}")


if __name__ == "__main__":
    main()
