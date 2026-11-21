"""Task 5 report: single-stage vs interleaved BP+SGT at equal BP budget.

Reads traces_<code>.json of run_single_vs_interleaved.py and writes
SINGLE_VS_INTERLEAVED.md (+ bler_<code>.png): per code the BLER / BER / average BP
iterations / average SGT calls of every decoder, and the geometric-mean BLER ratio of
each decoder against BP with the same budget and against interleaved 5x10.

Usage: python compare.py --traces <dir or file> [--out dir]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "task1_bp_schedules"))
import plots  # noqa: E402
from plot_schedules import code_report  # noqa: E402

REF = "SGT 5x10 (interleaved)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (args.traces if os.path.isdir(args.traces) else os.path.dirname(args.traces))
    parts, rows, labels = [], [], []
    for name, t in plots.load_traces(args.traces).items():
        md, s = code_report(name, t, out, REF)
        parts.append(md)
        rows.append((name, s))
        labels += [lab for lab in s if lab not in labels]
    head = ["# Task 5: single-stage vs interleaved BP+SGT", "",
            "Interleaved = 5 BP runs with 4 SGT stages (5x10 / 5x40). Single-stage = 2 BP runs with 1 SGT stage at "
            "the same total BP iterations. '(interleaved weights)' reuses the final 5x10 model with one stage; "
            "'(trained)' is a model trained as single-stage. Ratios are geometric-mean BLER ratios "
            "(vs BP with the same budget / vs " + REF + "); < 1 = lower BLER.", "",
            "code | " + " | ".join(labels), "---|" + "---|" * len(labels)]
    for name, s in rows:
        head.append(f"{name} | " + " | ".join(
            "-" if lab not in s else " / ".join("-" if v is None else f"{v:.2f}" for v in s[lab]) for lab in labels))
    with open(os.path.join(out, "SINGLE_VS_INTERLEAVED.md"), "w") as f:
        f.write("\n".join(head) + "\n\n" + "\n\n".join(parts) + "\n")
    print(f"wrote {os.path.join(out, 'SINGLE_VS_INTERLEAVED.md')}")


if __name__ == "__main__":
    main()
