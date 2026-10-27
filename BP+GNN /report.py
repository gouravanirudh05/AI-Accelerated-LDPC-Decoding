"""Turn an evaluation CSV into a readable comparison table.

    python -m ldpc_corrector.report --results Results_Corrector/eval/results.csv

Reports BER/BLER per Eb/N0 for every variant, the relative change against a
reference decoder (BP(5) by default), and the compute-per-decode table -- the
cost/accuracy tradeoff is the point of putting a corrector on top of BP(5)
instead of just running BP(20).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def read_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for record in csv.DictReader(handle):
            if not record.get("model"):
                continue
            row: Dict[str, Any] = {"model": record["model"]}
            for key in ("ebno_db", "ber", "fer", "flops_per_frame", "seconds_per_frame"):
                value = record.get(key, "")
                row[key] = float(value) if value not in ("", None) else None
            for key in ("frames", "frame_errors", "params"):
                value = record.get(key, "")
                row[key] = int(float(value)) if value not in ("", None) else None
            rows.append(row)
    return rows


NAME_WIDTH = 34


def _fit(name: str, width: int = NAME_WIDTH) -> str:
    """Left-align within ``width``, eliding names too long to fit."""
    if len(name) > width - 1:
        return name[: width - 2] + "~ "
    return name.ljust(width)


def _relative(value: Optional[float], reference: Optional[float]) -> str:
    """Percent change vs the reference; negative is an improvement."""
    if value is None or reference in (None, 0):
        return "     -"
    return f"{100.0 * (value - reference) / reference:+6.1f}%"


# Below this many observed frame errors, a rate estimate is too noisy to
# support a comparison, so the report marks it rather than presenting it as a
# clean number.
LOW_ERROR_THRESHOLD = 30


def accuracy_table(
    rows: Sequence[Dict[str, Any]], reference: str, metric: str
) -> List[str]:
    by_model: Dict[str, Dict[float, float]] = defaultdict(dict)
    noisy: Dict[str, Dict[float, bool]] = defaultdict(dict)
    for row in rows:
        if row["ebno_db"] is not None and row[metric] is not None:
            by_model[row["model"]][row["ebno_db"]] = row[metric]
            errors = row.get("frame_errors")
            noisy[row["model"]][row["ebno_db"]] = (
                errors is not None and errors < LOW_ERROR_THRESHOLD
            )

    snrs = sorted({row["ebno_db"] for row in rows if row["ebno_db"] is not None})
    header = _fit("model") + "".join(f"{s:>12.1f} dB" for s in snrs)
    lines = [header, "-" * len(header)]

    # Reference first, then everything else alphabetically.
    order = ([reference] if reference in by_model else []) + sorted(
        m for m in by_model if m != reference
    )
    for model in order:
        cells = []
        for snr in snrs:
            value = by_model[model].get(snr)
            if value is None:
                cells.append(f"{'-':>15}")
                continue
            marker = "*" if noisy[model].get(snr) else " "
            cells.append(f"{value:>14.3e}{marker}")
        lines.append(_fit(model) + "".join(cells))

        if model != reference and reference in by_model:
            deltas = []
            for snr in snrs:
                deltas.append(
                    f"{_relative(by_model[model].get(snr), by_model[reference].get(snr)):>15}"
                )
            lines.append(_fit("  vs " + reference) + "".join(deltas))
    return lines


def precision_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Per-SNR resolution limit, so differences are read against it.

    A rate estimated from ``k`` observed frame errors has relative standard
    error ~``1/sqrt(k)``.  Without this, a 2% gap at an SNR point carrying a
    20% standard error reads like a result.
    """
    import math

    best: Dict[float, int] = {}
    frames: Dict[float, int] = {}
    for row in rows:
        snr, errors = row["ebno_db"], row.get("frame_errors")
        if snr is None or not errors:
            continue
        if errors > best.get(snr, -1):
            best[snr] = errors
            frames[snr] = row.get("frames") or 0

    header = f"{'Eb/N0':>8}{'frames':>12}{'frame errors':>15}{'rel. sigma':>13}{'resolves':>12}"
    lines = [header, "-" * len(header)]
    for snr in sorted(best):
        errors = best[snr]
        sigma = 100.0 / math.sqrt(errors) if errors else float("inf")
        # Two independent estimates differ by sqrt(2)*sigma at one sigma, so
        # call ~3 sigma the smallest difference worth believing.
        resolves = f">{3 * sigma * 1.414:.0f}%" if errors else "-"
        lines.append(
            f"{snr:>8.1f}{frames[snr]:>12,}{errors:>15,}{sigma:>12.1f}%{resolves:>12}"
        )
    return lines


def cost_table(rows: Sequence[Dict[str, Any]], reference: str) -> List[str]:
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["model"] not in seen and row["flops_per_frame"] is not None:
            seen[row["model"]] = row

    baseline = seen.get(reference)
    header = (
        _fit("model")
        + f"{'FLOPs/frame':>14}{'rel':>8}{'us/frame':>12}{'rel':>8}{'params':>10}"
    )
    lines = [header, "-" * len(header)]
    for model, row in sorted(seen.items()):
        flops = row["flops_per_frame"] or 0.0
        seconds = row["seconds_per_frame"] or 0.0
        flops_rel = (
            f"{flops / baseline['flops_per_frame']:6.2f}x"
            if baseline and baseline.get("flops_per_frame")
            else "     -"
        )
        time_rel = (
            f"{seconds / baseline['seconds_per_frame']:6.2f}x"
            if baseline and baseline.get("seconds_per_frame")
            else "     -"
        )
        lines.append(
            _fit(model)
            + f"{flops:>14,.0f}{flops_rel:>8}"
            + f"{seconds * 1e6:>12.1f}{time_rel:>8}{row.get('params') or 0:>10,}"
        )
    return lines


def build_report(
    results: Path, reference: str = "BP(5)", cost_reference: str = "BP(20)"
) -> str:
    rows = read_results(results)
    if not rows:
        raise ValueError(f"No usable rows in {results}.")

    sections = [
        f"# Results — {results}",
        "",
        "## BER vs Eb/N0",
        "",
        *accuracy_table(rows, reference, "ber"),
        "",
        "## BLER (frame error rate) vs Eb/N0",
        "",
        *accuracy_table(rows, reference, "fer"),
        "",
        f"## Compute per decode (relative to {cost_reference})",
        "",
        *cost_table(rows, cost_reference),
        "",
        "## Statistical resolution",
        "",
        *precision_table(rows),
        "",
        "Read every difference against the resolution at that SNR. Points with",
        "few frame errors cannot separate variants no matter how the numbers look.",
        "",
        "Negative percentages are improvements. Reference rows imported from a",
        "prior run (ECCT/AECCT) have no compute figures.",
        "",
        f"* fewer than {LOW_ERROR_THRESHOLD} frame errors observed: the estimate is",
        "  too noisy to compare. Raise eval.frames for those points.",
    ]
    return "\n".join(sections)


# ------------------------------------------------------- seed aggregation


def _group_key(name: str) -> str:
    """Strip a trailing ``_s<seed>`` so seed repeats group together."""
    import re

    return re.sub(r"_s\d+$", "", name)


def seed_summary(checkpoint_dir: Path, pattern: str = "*.best.pt") -> str:
    """Mean and spread of validation FER across seed repeats.

    A single run cannot support an architecture claim: on this setup two runs
    of an identical model landed 0.0130 FER apart, which is wider than most of
    the differences between architectures.  This groups ``name_s42`` /
    ``name_s43`` / ... and reports the spread alongside the mean.
    """
    import statistics

    import torch

    groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for path in sorted(checkpoint_dir.glob(pattern)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fer = payload.get("validation_fer")
        if fer is None:
            continue
        name = path.name.replace(".best.pt", "")
        groups[_group_key(name)].append((name, float(fer)))

    if not groups:
        raise ValueError(f"No checkpoints with a validation FER in {checkpoint_dir}.")

    header = (
        _fit("variant")
        + f"{'runs':>6}{'mean FER':>12}{'min':>11}{'max':>11}{'spread':>10}"
    )
    lines = [header, "-" * len(header)]
    for key in sorted(groups, key=lambda k: statistics.fmean(v for _, v in groups[k])):
        values = [v for _, v in groups[key]]
        spread = max(values) - min(values)
        lines.append(
            _fit(key)
            + f"{len(values):>6}{statistics.fmean(values):>12.6f}"
            + f"{min(values):>11.6f}{max(values):>11.6f}{spread:>10.6f}"
        )

    single = [k for k, v in groups.items() if len(v) < 2]
    if single:
        lines += [
            "",
            "Single-seed (no spread available, treat as one sample): "
            + ", ".join(sorted(single)),
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="Results_Corrector/eval/results.csv")
    parser.add_argument("--reference", default="BP(5)")
    parser.add_argument("--cost-reference", default="BP(20)")
    parser.add_argument("--out", default="", help="Also write the report here.")
    parser.add_argument(
        "--seed-dir",
        default="",
        help="Summarize validation FER across seed repeats in this directory "
        "instead of reading an evaluation CSV.",
    )
    args = parser.parse_args(argv)

    if args.seed_dir:
        report = seed_summary(Path(args.seed_dir))
        print(report)
        if args.out:
            Path(args.out).write_text(report + "\n")
        return

    report = build_report(
        Path(args.results), args.reference, args.cost_reference
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
