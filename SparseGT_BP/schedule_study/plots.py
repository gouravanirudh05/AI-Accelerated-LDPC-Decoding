"""Reading traces_<code>.json files and the shared plot style (numpy + matplotlib only,
no torch, so the analysis runs on any laptop after downloading the Kaggle output)."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# categorical slots in fixed order (SGT schedules); BP baselines are neutral and dashed
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAYS = ["#4a4a48", "#7a7a76", "#9d9d98", "#bdbdb7", "#2f2f2d"]
BP_MARKERS = ["s", "D", "^", "v", "P"]
# ordinal ramp for "stopped in BP run r" (light = early run, dark = late run)
RUN_RAMP = LinearSegmentedColormap.from_list("runs", ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
NEVER = "#bdbdb7"
WRONG = "#e34948"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "axes.edgecolor": "#7a7a76",
                     "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
                     "lines.markersize": 5, "figure.dpi": 130})


def load_traces(path):
    """path: a traces_*.json file or a folder (searched recursively). Returns {code: trace}."""
    files = [path] if path.endswith(".json") else sorted(glob.glob(os.path.join(path, "**", "traces_*.json"),
                                                                    recursive=True))
    out = {}
    for f in files:
        with open(f) as fh:
            t = json.load(fh)
        name = t["code"]["name"]
        if name in out:  # several runs of one code: merge their decoders
            out[name]["decoders"].update(t["decoders"])
        else:
            out[name] = t
    return out


def styles(labels, decs):
    """{label: matplotlib kwargs}: SGT decoders take categorical slots in the order given,
    BP baselines take dashed neutrals."""
    st, i, j = {}, 0, 0
    for lab in labels:
        if decs[lab]["kind"] == "sgt":
            st[lab] = dict(color=SERIES[i % len(SERIES)], marker="o", linestyle="-")
            i += 1
        else:
            st[lab] = dict(color=GRAYS[j % len(GRAYS)], marker=BP_MARKERS[j % len(BP_MARKERS)], linestyle="--",
                           linewidth=1.5)
            j += 1
    return st


def stop_fractions(p, runs):
    """(fraction converged correctly in each run, fraction converged to a wrong codeword
    in each run, fraction never converged) at one Eb/N0 point."""
    nf = p["frames"]
    ok = np.array(p["run_ok"][:runs], float) / nf
    wrong = np.array(p["run_wrong"][:runs], float) / nf
    return ok, wrong, p["never"] / nf


def fmt(x):
    return "-" if x is None else ("0" if x == 0 else f"{x:.2e}")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p
