"""Plot BP, GAT+BP, ECCT, and final AECCT on one figure.

The GAT+BP and BP values are read from the saved-checkpoint evaluation at
the end of ``GAT+BP/gat_bp_training.log``.  ECCT and AECCT values are read
from ``comparison/data/test_results.csv``.

Run from the repository root with:

    python3 comparison/plot_all_four.py

The figure is written to ``comparison/plots/all_four_comparison.png``.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA = HERE / "data"
OUT = HERE / "plots"
GAT_LOG = REPO_ROOT / "GAT+BP" / "gat_bp_training.log"

MODELS = [
    "BP",
    "BP + bidirectional edge-GATv2",
    "ECCT (FP32)",
    "AECCT final (ternary W / int8 A)",
]
COLORS = {
    "BP": "#4d4d4d",
    "BP + bidirectional edge-GATv2": "#d95f02",
    "ECCT (FP32)": "#1f77b4",
    "AECCT final (ternary W / int8 A)": "#2ca25f",
}
MARKERS = {
    "BP": "x",
    "BP + bidirectional edge-GATv2": "o",
    "ECCT (FP32)": "s",
    "AECCT final (ternary W / int8 A)": "^",
}
LINESTYLES = {
    "BP": "--",
    "BP + bidirectional edge-GATv2": "-",
    "ECCT (FP32)": "-",
    "AECCT final (ternary W / int8 A)": "-",
}


def read_reference_results():
    with (DATA / "test_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = {}
    for row in rows:
        model = row["model"]
        if model in {"ECCT (FP32)", "AECCT final (ternary W / int8 A)"}:
            selected[(model, int(float(row["ebno_db"])))]=row
    return selected


def read_saved_gat_evaluation():
    checkpoint = None
    with GAT_LOG.open() as handle:
        for line in handle:
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "checkpoint_loss" in record:
                checkpoint = record["evaluation"]

    if checkpoint is None:
        raise RuntimeError(f"No saved-checkpoint evaluation found in {GAT_LOG}")
    return checkpoint


def make_rows():
    references = read_reference_results()
    gat_eval = read_saved_gat_evaluation()
    rows = []

    for ebno in (4, 5, 6):
        values = gat_eval[str(ebno)]
        rows.extend(
            [
                {
                    "model": "BP",
                    "ebno_db": ebno,
                    "ber": values["bp_ber"],
                    "fer": values["bp_fer"],
                },
                {
                    "model": "BP + bidirectional edge-GATv2",
                    "ebno_db": ebno,
                    "ber": values["gat_bp_ber"],
                    "fer": values["gat_bp_fer"],
                },
            ]
        )

        for model in ("ECCT (FP32)", "AECCT final (ternary W / int8 A)"):
            reference = references[(model, ebno)]
            rows.append(
                {
                    "model": model,
                    "ebno_db": ebno,
                    "ber": float(reference["ber"]),
                    "fer": float(reference["fer"]),
                }
            )

    return rows


def style_axis(axis):
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xticks([4, 5, 6])
    axis.set_xlabel(r"$E_b/N_0$ (dB)")


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)

    for axis, metric, ylabel in zip(
        axes,
        ("ber", "fer"),
        ("BER (lower is better)", "FER (lower is better)"),
    ):
        for model in MODELS:
            model_rows = [row for row in rows if row["model"] == model]
            model_rows.sort(key=lambda row: row["ebno_db"])
            axis.plot(
                [row["ebno_db"] for row in model_rows],
                [row[metric] for row in model_rows],
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                linewidth=2,
                markersize=7,
                label=model,
            )
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        style_axis(axis)

    axes[0].set_title("Bit error rate")
    axes[1].set_title("Frame error rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.suptitle("LDPC(49,24): BP, GAT+BP, ECCT, and AECCT", fontsize=15)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))

    output = OUT / "all_four_comparison.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    OUT.mkdir(exist_ok=True)
    output = plot(make_rows())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
