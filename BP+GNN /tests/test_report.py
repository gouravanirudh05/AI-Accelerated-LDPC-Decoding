from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ldpc_corrector.evaluate import CSV_FIELDS
from ldpc_corrector.report import build_report, read_results

ROWS = [
    # model, ebno, ber, fer, flops, seconds, params
    ("BP(5)", 3.0, 2.0e-2, 1.7e-1, 25480, 4.1e-4, 0),
    ("BP(5)", 5.0, 6.0e-4, 7.3e-3, 25480, 4.1e-4, 0),
    ("BP(20)", 3.0, 1.5e-2, 1.0e-1, 101920, 1.6e-3, 0),
    ("BP(20)", 5.0, 2.4e-4, 2.0e-3, 101920, 1.6e-3, 0),
    ("BP(5)+GAT", 3.0, 1.0e-2, 8.0e-2, 1714888, 9.0e-4, 18297),
    ("BP(5)+GAT", 5.0, 1.2e-4, 1.0e-3, 1714888, 9.0e-4, 18297),
]


@pytest.fixture
def results_csv(tmp_path) -> Path:
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model, ebno, ber, fer, flops, seconds, params in ROWS:
            writer.writerow(
                {
                    "model": model,
                    "ebno_db": ebno,
                    "ber": ber,
                    "fer": fer,
                    "frames": 20480,
                    "frame_errors": 100,
                    "flops_per_frame": flops,
                    "seconds_per_frame": seconds,
                    "params": params,
                }
            )
    return path


def test_reads_every_row(results_csv) -> None:
    rows = read_results(results_csv)
    assert len(rows) == len(ROWS)
    assert {row["model"] for row in rows} == {"BP(5)", "BP(20)", "BP(5)+GAT"}


def test_report_contains_both_metrics_and_the_cost_table(results_csv) -> None:
    report = build_report(results_csv)
    assert "## BER vs Eb/N0" in report
    assert "## BLER (frame error rate) vs Eb/N0" in report
    assert "Compute per decode" in report
    for model in ("BP(5)", "BP(20)", "BP(5)+GAT"):
        assert model in report


def test_relative_change_sign_and_magnitude(results_csv) -> None:
    """Improvements must read negative, regressions positive."""
    report = build_report(results_csv, reference="BP(5)")
    # GAT BER at 3 dB: 1.0e-2 vs 2.0e-2 = -50%.
    assert "-50.0%" in report
    # BP(20) BLER at 5 dB: 2.0e-3 vs 7.3e-3 = -72.6%.
    assert "-72.6%" in report


def test_regression_is_reported_as_positive(tmp_path) -> None:
    path = tmp_path / "worse.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model, fer in (("BP(5)", 0.114), ("BP(5)+GCN", 0.1164)):
            writer.writerow(
                {"model": model, "ebno_db": 3.0, "ber": 0.01, "fer": fer,
                 "flops_per_frame": 1000, "seconds_per_frame": 1e-4, "params": 0}
            )
    report = build_report(path, reference="BP(5)")
    assert "+2.1%" in report, "a FER regression must not be shown as a gain"


def test_cost_table_is_relative_to_the_cost_reference(results_csv) -> None:
    report = build_report(results_csv, cost_reference="BP(20)")
    # 1714888 / 101920 = 16.83x FLOPs, but 9.0e-4 / 1.6e-3 = 0.56x wall-clock.
    assert "16.83x" in report
    assert "0.56x" in report


def test_missing_cells_are_dashes_not_zeros(tmp_path) -> None:
    """A model measured at only some SNRs must not read as BER 0."""
    path = tmp_path / "sparse.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {"model": "BP(5)", "ebno_db": 3.0, "ber": 0.02, "fer": 0.17,
             "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
        )
        # An imported reference row: no compute figures, different SNR.
        writer.writerow(
            {"model": "ECCT (FP32)", "ebno_db": 5.0, "ber": 3.4e-4, "fer": 5.1e-3,
             "flops_per_frame": "", "seconds_per_frame": "", "params": ""}
        )
    report = build_report(path)
    assert "ECCT (FP32)" in report
    # The BER table must show a dash where the model was not measured.
    ber_row = next(
        line for line in report.splitlines() if line.startswith("ECCT (FP32)")
    )
    assert "-" in ber_row and "0.000e+00" not in ber_row

    # The reference row has no cost data, so it stays out of the cost table --
    # checked against the table rows only, since the trailing note names ECCT.
    cost_rows = [
        line
        for line in report.split("Compute per decode")[1].splitlines()
        if line.startswith(("BP(", "ECCT", "AECCT"))
    ]
    assert cost_rows and not any(line.startswith("ECCT") for line in cost_rows)


def test_long_model_names_keep_the_table_aligned(tmp_path) -> None:
    path = tmp_path / "long.csv"
    long_name = "AECCT final (ternary W / int8 A)"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model in ("BP(5)", long_name):
            writer.writerow(
                {"model": model, "ebno_db": 4.0, "ber": 0.01, "fer": 0.1,
                 "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
            )
    report = build_report(path)
    # Compare within one table: the BER section only.
    ber_section = report.split("## BER vs Eb/N0")[1].split("##")[0]
    table = [
        line
        for line in ber_section.splitlines()
        if line.startswith(("BP(5)", "AECCT", "model"))
    ]
    assert len(table) >= 3, "expected a header and both model rows"
    widths = {len(line) for line in table}
    assert len(widths) == 1, f"misaligned table rows: {widths}"
    # This name fits the field exactly, so it must appear in full.
    assert any(long_name in line for line in table)


def test_names_too_long_to_fit_are_elided(tmp_path) -> None:
    """A name wider than the field must be elided rather than overflow."""
    from ldpc_corrector.report import NAME_WIDTH

    path = tmp_path / "verylong.csv"
    long_name = "X" * (NAME_WIDTH + 20)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for model in ("BP(5)", long_name):
            writer.writerow(
                {"model": model, "ebno_db": 4.0, "ber": 0.01, "fer": 0.1,
                 "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
            )
    report = build_report(path)
    ber_section = report.split("## BER vs Eb/N0")[1].split("##")[0]
    table = [
        line
        for line in ber_section.splitlines()
        if line.startswith(("BP(5)", "X", "model"))
    ]
    assert len({len(line) for line in table}) == 1, "elision must keep alignment"
    assert any("~" in line for line in table), "expected an elision marker"


def test_empty_results_are_rejected(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
    with pytest.raises(ValueError, match="No usable rows"):
        build_report(path)


def test_noisy_estimates_are_marked(tmp_path) -> None:
    """A rate built on a handful of frame errors must be flagged, not hidden."""
    from ldpc_corrector.report import LOW_ERROR_THRESHOLD

    path = tmp_path / "noisy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        # 3 dB: plenty of errors. 6 dB: only a few.
        writer.writerow(
            {"model": "BP(5)", "ebno_db": 3.0, "ber": 2e-2, "fer": 1.7e-1,
             "frames": 2048, "frame_errors": 348,
             "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
        )
        writer.writerow(
            {"model": "BP(5)", "ebno_db": 6.0, "ber": 1e-5, "fer": 2e-4,
             "frames": 20480, "frame_errors": LOW_ERROR_THRESHOLD - 25,
             "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
        )
    report = build_report(path)
    row = next(line for line in report.splitlines() if line.startswith("BP(5)"))
    assert row.count("*") == 1, "exactly the low-error cell should be marked"
    assert "too noisy to compare" in report


def test_well_sampled_estimates_are_not_marked(results_csv) -> None:
    report = build_report(results_csv)
    body = report.split("## BER vs Eb/N0")[1].split("##")[0]
    model_rows = [l for l in body.splitlines() if l.startswith(("BP(", "BP(5)+GAT"))]
    assert model_rows and not any("*" in line for line in model_rows)


def test_precision_section_reports_the_resolution_limit(tmp_path) -> None:
    """The report must state how large a difference each SNR can resolve."""
    path = tmp_path / "prec.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        # 3 dB is well sampled; 6 dB is not.
        writer.writerow(
            {"model": "BP(5)", "ebno_db": 3.0, "ber": 2e-2, "fer": 1.1e-1,
             "frames": 8772, "frame_errors": 1000,
             "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
        )
        writer.writerow(
            {"model": "BP(5)", "ebno_db": 6.0, "ber": 1e-5, "fer": 5e-4,
             "frames": 51200, "frame_errors": 26,
             "flops_per_frame": 100, "seconds_per_frame": 1e-4, "params": 0}
        )
    report = build_report(path)
    assert "## Statistical resolution" in report

    section = report.split("## Statistical resolution")[1]
    well_sampled = next(l for l in section.splitlines() if l.strip().startswith("3.0"))
    under_sampled = next(l for l in section.splitlines() if l.strip().startswith("6.0"))
    # 1000 errors -> 3.2% sigma; 26 errors -> 19.6% sigma.
    assert "3.2%" in well_sampled
    assert "19.6%" in under_sampled
    # And the poorly sampled point must admit it resolves much less.
    assert "1,000" in well_sampled and "26" in under_sampled
    # 3 sigma * sqrt(2): ~13% at 3 dB, ~83% at 6 dB.
    assert ">13%" in well_sampled and ">83%" in under_sampled


def _write_checkpoint(path, fer: float) -> None:
    import torch

    torch.save({"step": 1, "config": {}, "model_state": {}, "validation_fer": fer}, path)


def test_seed_summary_groups_repeats_and_reports_spread(tmp_path) -> None:
    from ldpc_corrector.report import seed_summary

    for seed, fer in ((42, 0.1150), (43, 0.1283), (44, 0.1200)):
        _write_checkpoint(tmp_path / f"gat_s{seed}.best.pt", fer)
    _write_checkpoint(tmp_path / "gcn_s42.best.pt", 0.1160)

    report = seed_summary(tmp_path)
    gat_row = next(l for l in report.splitlines() if l.startswith("gat"))

    assert "3" in gat_row  # three runs grouped
    # mean of the three, and the spread max-min = 0.0133.
    assert "0.121100" in gat_row
    assert "0.013300" in gat_row
    # A one-run variant must be called out rather than shown as spread 0.
    assert "Single-seed" in report and "gcn" in report.split("Single-seed")[1]


def test_seed_summary_needs_checkpoints(tmp_path) -> None:
    from ldpc_corrector.report import seed_summary

    with pytest.raises(ValueError, match="No checkpoints"):
        seed_summary(tmp_path)


def test_seed_grouping_only_strips_a_trailing_seed_suffix() -> None:
    from ldpc_corrector.report import _group_key

    assert _group_key("pyg_gat_v1_s42") == "pyg_gat_v1"
    assert _group_key("pyg_gat_v1") == "pyg_gat_v1"
    # A name that merely contains digits must survive intact.
    assert _group_key("pyg_sparse_topk8") == "pyg_sparse_topk8"
    assert _group_key("local_gt_hop2") == "local_gt_hop2"
