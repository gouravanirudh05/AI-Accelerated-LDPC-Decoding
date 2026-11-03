"""Turn step-2 outputs into plots and a markdown summary.

    python -m sparse_gt.experiments.report_step2 results/kaggle_step2/results/step2

Writes into the same directory:

* ``validation.png``  our BP(100) vs AFF3CT for WiMAX-576 and 10G-Ethernet.
* ``bp_curves.png``   BP(100) FER for every code.
* ``coverage.png``    frame coverage vs k/n, recoverable and stuck frames.
* ``summary.md``      the tables behind the step-2 verdict, including the
                      oracle bound: FER of BP(T) followed by a *perfect*
                      corrector restricted to the top-k set, against simply
                      running more BP iterations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..codes import CODES_DIR  # noqa: E402
from ..simulate import fer_interval, read_aff3ct_reference  # noqa: E402

# Reference palette (light surface): categorical slots in fixed order, blue ramp
# for ordinal T, text tokens for ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
T_PLOT = (2, 5, 10, 20)
K_FRAC = 0.05  # k/n used for the oracle table

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "font.size": 9,
})


def k_for_coverage(hist: np.ndarray, level: float) -> int | None:
    """Smallest k whose frame coverage reaches ``level`` (hist = k_needed counts)."""
    total = hist.sum()
    if total == 0:
        return None
    return int(np.searchsorted(np.cumsum(hist), level * total))


def plot_validation(curves: dict, out: Path) -> list[str]:
    names = [n for n in curves if curves[n].get("reference")]
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 6.5), squeeze=False)
    lines = ["| Code | Eb/N0 | ours FER [95% CI] | AFF3CT FER [95% CI] | overlap | ours BER | AFF3CT BER |",
             "|---|---:|---|---|:-:|---:|---:|"]
    for col, name in enumerate(names):
        ours = curves[name]["rows"]
        ref = read_aff3ct_reference(CODES_DIR / "references" / curves[name]["reference"])
        ref_by_snr = {round(r["ebn0"], 2): r for r in ref}
        lo_snr, hi_snr = ours[0]["ebn0"] - 0.1, ours[-1]["ebn0"] + 0.1
        ref = [r for r in ref if lo_snr <= r["ebn0"] <= hi_snr]
        for row, metric in enumerate(("fer", "ber")):
            ax = axes[row][col]
            x = [r["ebn0"] for r in ours]
            y = [r[metric] for r in ours]
            if metric == "fer":
                ci = np.array([fer_interval(r["frame_errors"], r["frames"]) for r in ours])
                ax.fill_between(x, ci[:, 0], ci[:, 1], color=SERIES[0], alpha=0.15, linewidth=0)
            seen = [(a, b) for a, b in zip(x, y) if b > 0]  # zero counts have no log value
            ax.semilogy(*zip(*seen), "-o", color=SERIES[0], label="ours, BP(100) flooding SPA")
            ax.semilogy([r["ebn0"] for r in ref], [r[metric] for r in ref], "--s",
                        color=SERIES[1], label="AFF3CT reference")
            ax.set_title(f"{name}: {metric.upper()}", color=INK, loc="left")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel(metric.upper())
            if row == 0 and col == 0:
                ax.legend()
        for r in ours:
            ref_row = ref_by_snr.get(round(r["ebn0"], 2))
            if ref_row is None:
                continue
            a = fer_interval(r["frame_errors"], r["frames"])
            b = fer_interval(ref_row["frame_errors"], ref_row["frames"])
            overlap = "yes" if a[0] <= b[1] and b[0] <= a[1] else "**no**"
            lines.append(f"| {name} | {r['ebn0']:.2f} | {r['fer']:.2e} [{a[0]:.1e}, {a[1]:.1e}] | "
                         f"{ref_row['fer']:.2e} [{b[0]:.1e}, {b[1]:.1e}] | {overlap} | "
                         f"{r['ber']:.2e} | {ref_row['ber']:.2e} |")
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=150)
    plt.close(fig)
    return lines


def plot_curves(curves: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, c) in enumerate(curves.items()):
        rows = [r for r in c["rows"] if r["fer"] > 0]
        ax.semilogy([r["ebn0"] for r in rows], [r["fer"] for r in rows], "-o",
                    color=SERIES[i % len(SERIES)], label=f"{name} (rate {c['k'] / c['n']:.2f})")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("FER")
    ax.set_title("BP(100) frame error rate, all codes", color=INK, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bp_curves.png", dpi=150)
    plt.close(fig)


def middle(snrs: list[float]) -> float:
    return snrs[len(snrs) // 2]


def plot_coverage(coverage: dict, out: Path) -> None:
    names = list(coverage)
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 2.3 * len(names)), squeeze=False,
                             sharey=True)
    for row, name in enumerate(names):
        n, snr = coverage[name]["n"], middle(coverage[name]["snrs"])
        hists = np.load(out / f"coverage_hist_{name}.npz")
        for col, group in enumerate(("recoverable", "stuck")):
            ax = axes[row][col]
            for T, color in zip(T_PLOT, BLUE_RAMP):
                key = f"{snr}/{T}/k_needed_{group}"
                if key not in hists or hists[key].sum() == 0:
                    continue
                h = hists[key]
                cov = np.cumsum(h)[1:] / h.sum()
                k = np.arange(1, n + 1)
                ax.plot(100 * k / n, cov, color=color, label=f"T={T} ({h.sum()} frames)")
            ax.set_xlim(0, 25)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{name} @ {snr} dB: {group}", color=INK, loc="left", fontsize=8.5)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="lower right")
            if col == 0:
                ax.set_ylabel("frame coverage")
            if row == len(names) - 1:
                ax.set_xlabel("k / n (%)")
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=150)
    plt.close(fig)


def coverage_tables(coverage: dict, out: Path) -> list[str]:
    lines = []
    for name, c in coverage.items():
        n = c["n"]
        k = max(1, round(K_FRAC * n))
        hists = np.load(out / f"coverage_hist_{name}.npz")
        by = {(s["ebn0"], s["T"]): s for s in c["summaries"]}
        Ts = sorted({s["T"] for s in c["summaries"]})
        lines += [f"\n### {name} (n={n}, k={c['k']})\n",
                  f"Oracle = FER of BP(T) then a perfect corrector limited to the top-k set, "
                  f"k = {k} ({100 * K_FRAC:.0f}% of n).\n",
                  "| Eb/N0 | T | FER BP(T) | stuck share | k/n for 90% frames (all / stuck) | "
                  f"oracle FER | FER BP(2T) | FER BP(100) | P(err\\|flipped) vs base | errs / unsat (median) |",
                  "|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
        for snr in c["snrs"]:
            for T in Ts:
                s = by[(snr, T)]
                if s["failed_frames"] == 0:
                    continue
                h_all = sum(hists[f"{snr}/{T}/k_needed_{g}"] for g in ("recoverable", "stuck"))
                h_stuck = hists[f"{snr}/{T}/k_needed_stuck"]
                k90 = k_for_coverage(h_all, 0.9)
                k90s = k_for_coverage(h_stuck, 0.9)
                covered = h_all[: k + 1].sum() / h_all.sum()
                oracle = s["fer"] * (1 - covered)
                two_t = by.get((snr, 2 * T))
                stuck_share = s["failed_by_group"]["stuck"] / s["failed_frames"]
                f = s["flips"]
                k90_stuck = f"{100 * k90s / n:.1f}%" if k90s else "-"
                fer_2t = f"{two_t['fer']:.2e}" if two_t else "-"
                lines.append(
                    f"| {snr} | {T} | {s['fer']:.2e} | {stuck_share:.0%} | "
                    f"{100 * k90 / n:.1f}% / {k90_stuck} | {oracle:.2e} | {fer_2t} | "
                    f"{s['fer_reference']:.2e} | {f['p_error_given_flipped']:.2f} vs "
                    f"{f['p_error_baseline']:.3f} | {s['errors_per_failed_frame']['p50']} / "
                    f"{s['unsatisfied_checks_per_failed_frame']['p50']} |"
                )
    return lines


def timing_table(timing: dict) -> list[str]:
    """Latency (batch 1) and throughput (largest batch) per code."""
    lines = []
    for name, info in timing.items():
        rows = info["rows"]
        batches = sorted({r["batch"] for r in rows})
        big = batches[-1]
        lines += [f"\n**{name}** (n={info['n']}, {info['edges']} edges, {info['device']})\n",
                  f"| decoder | latency @ B=1 (ms, p10-p90) | B={big}: ms per batch | frames/s @ B={big} |",
                  "|---|---|---:|---:|"]
        fixed = {(r["batch"], r["T"]): r for r in rows if r["kind"] == "bp_fixed"}
        for T in sorted({T for _, T in fixed}):
            a, b = fixed.get((1, T)), fixed.get((big, T))
            lat = f"{a['ms']:.3f} ({a['p10']:.3f}-{a['p90']:.3f})" if a else "-"
            lines.append(f"| BP({T}), fixed | {lat} | {b['ms']:.2f} | {b['frames_per_s']:,.0f} |" if b else
                         f"| BP({T}), fixed | {lat} | - | - |")
        early = {r["batch"]: r for r in rows if r["kind"] == "bp_early_stop"}
        if early:
            a, b = early.get(1), early.get(big)
            lines.append(
                f"| BP(≤{b['max_iterations']}), early stop @ {b['ebn0']} dB "
                f"(mean {b['mean_iterations']:.1f} it, batch max {b['max_iterations_used']}) | "
                f"{a['ms']:.3f} | {b['ms']:.2f} | {b['frames_per_s']:,.0f} |")
        topk = {(r["batch"], r["k"]): r for r in rows if r["kind"] == "topk"}
        for k in sorted({k for _, k in topk}):
            a, b = topk.get((1, k)), topk.get((big, k))
            lines.append(f"| top-k selection, k={k} ({100 * k / info['n']:.0f}% of n) | "
                         f"{a['ms']:.3f} | {b['ms']:.3f} | - |" if a and b else "")
    return lines


def main(argv=None) -> None:
    out = Path((argv or sys.argv[1:])[0])
    curves = json.loads((out / "curves.json").read_text())
    coverage = json.loads((out / "coverage.json").read_text()) if (out / "coverage.json").exists() else {}
    timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}

    md = ["# Step 2 results\n", "## BP validation against AFF3CT\n", "![validation](validation.png)\n"]
    md += plot_validation(curves, out)
    plot_curves(curves, out)
    md += ["\n## BP(100) curves\n", "![curves](bp_curves.png)\n"]
    if coverage:
        plot_coverage(coverage, out)
        md += ["\n## Top-k coverage\n", "![coverage](coverage.png)\n"]
        md += coverage_tables(coverage, out)
    if timing and all(isinstance(v, dict) and "rows" in v for v in timing.values()):
        md += ["\n## BP timing\n"] + timing_table(timing)
    elif timing:
        md += ["\n## BP timing\n", "timing.json is in the pre-harness format; see the separate timing run."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
