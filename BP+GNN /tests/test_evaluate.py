from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from ldpc_corrector.config import load_config
from ldpc_corrector.decoder import BPBaseline
from ldpc_corrector.evaluate import (
    CSV_FIELDS,
    load_checkpoint_decoder,
    measure_seconds_per_frame,
    plot_curves,
    read_reference_csv,
    sweep_decoder,
    write_csv,
)


@pytest.fixture
def generator(ldpc49_code):
    gen, _ = ldpc49_code
    return torch.as_tensor(gen, dtype=torch.float32)


def test_sweep_reports_one_row_per_snr_point(generator, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    rows = sweep_decoder(
        BPBaseline(parity_check, 5),
        generator,
        [2.0, 5.0],
        frames=512,
        batch_size=256,
        device=torch.device("cpu"),
    )
    assert [row["ebno_db"] for row in rows] == [2.0, 5.0]
    for row in rows:
        assert 0.0 <= row["ber"] <= 1.0
        assert 0.0 <= row["fer"] <= 1.0
        # A block is wrong if any bit is, so FER >= BER always.
        assert row["fer"] >= row["ber"]
        assert row["frames"] == 512


def test_ber_falls_as_snr_rises(generator, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    rows = sweep_decoder(
        BPBaseline(parity_check, 5),
        generator,
        [1.0, 3.0, 5.0],
        frames=1024,
        batch_size=512,
        device=torch.device("cpu"),
    )
    bers = [row["ber"] for row in rows]
    assert bers == sorted(bers, reverse=True), f"BER did not fall with SNR: {bers}"


def test_variants_see_identical_channel_realizations(generator, ldpc49_code) -> None:
    """The sweep must be seeded per SNR point so comparisons are paired."""
    _, parity_check = ldpc49_code
    kwargs = dict(
        generator=generator,
        ebn0_points=[3.0],
        frames=512,
        batch_size=256,
        device=torch.device("cpu"),
        seed=99,
    )
    first = sweep_decoder(BPBaseline(parity_check, 5), **kwargs)
    second = sweep_decoder(BPBaseline(parity_check, 5), **kwargs)
    assert first[0]["bit_errors"] == second[0]["bit_errors"]


def test_min_frame_errors_stops_early(generator, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    rows = sweep_decoder(
        BPBaseline(parity_check, 5),
        generator,
        [1.0],
        frames=100_000,
        batch_size=256,
        device=torch.device("cpu"),
        min_frame_errors=10,
    )
    # Low SNR: 10 frame errors arrive in the first batch or two, so the sweep
    # must not have decoded all 100k frames.
    assert rows[0]["frames"] < 100_000
    assert rows[0]["frame_errors"] >= 10


def test_timing_is_positive(generator, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    seconds = measure_seconds_per_frame(
        BPBaseline(parity_check, 5),
        generator,
        torch.device("cpu"),
        batch_size=128,
        batches=2,
    )
    assert seconds > 0.0


def test_csv_has_a_stable_schema(tmp_path, generator, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    rows = sweep_decoder(
        BPBaseline(parity_check, 5), generator, [4.0], 256, 256, torch.device("cpu")
    )
    rows = [{"model": "BP(5)", **row} for row in rows]
    path = tmp_path / "results.csv"
    write_csv(rows, path)

    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_FIELDS
        loaded = list(reader)
    assert len(loaded) == 1
    assert loaded[0]["model"] == "BP(5)"


def test_reference_overlay_reads_the_repo_csv() -> None:
    """The already-reproduced ECCT/AECCT numbers must load as-is."""
    path = Path(__file__).resolve().parents[2] / "comparison" / "data" / "test_results.csv"
    if not path.is_file():
        pytest.skip("reference CSV not present")

    rows = read_reference_csv(path)
    assert rows, "no reference rows parsed"
    assert {row["model"] for row in rows}
    for row in rows:
        assert 0.0 <= row["ber"] <= 1.0
        assert set(CSV_FIELDS) >= set(row)


def test_reference_overlay_can_filter_models() -> None:
    path = Path(__file__).resolve().parents[2] / "comparison" / "data" / "test_results.csv"
    if not path.is_file():
        pytest.skip("reference CSV not present")
    rows = read_reference_csv(path, models=["ECCT (FP32)"])
    assert rows and {row["model"] for row in rows} == {"ECCT (FP32)"}


def test_missing_reference_csv_is_a_warning_not_a_crash(tmp_path) -> None:
    assert read_reference_csv(tmp_path / "nope.csv") == []


def test_plotting_writes_both_curves(tmp_path) -> None:
    rows = [
        {"model": "BP(5)", "ebno_db": 2.0, "ber": 1e-2, "fer": 1e-1},
        {"model": "BP(5)", "ebno_db": 4.0, "ber": 1e-3, "fer": 1e-2},
        {"model": "BP(5)+GCN", "ebno_db": 2.0, "ber": 9e-3, "fer": 9e-2},
        {"model": "BP(5)+GCN", "ebno_db": 4.0, "ber": 8e-4, "fer": 8e-3},
        # A zero rate cannot go on a log axis; it must be dropped, not crash.
        {"model": "ECCT (FP32)", "ebno_db": 6.0, "ber": 0.0, "fer": 0.0},
    ]
    written = plot_curves(rows, tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.is_file() and path.stat().st_size > 0


def test_checkpoint_round_trip_rebuilds_the_model(tmp_path, ldpc49_code) -> None:
    """A checkpoint must carry enough config to rebuild its own architecture."""
    from ldpc_corrector.train import build_model

    _, parity_check = ldpc49_code
    config = load_config(
        None,
        [
            "name=roundtrip",
            "model.corrector=gat",
            "model.hidden_dim=16",
            "model.heads=2",
            "model.head_dim=8",
        ],
    )
    model = build_model(config, parity_check)
    path = tmp_path / "roundtrip.pt"
    torch.save(
        {"step": 1, "config": config.to_dict(), "model_state": model.state_dict()}, path
    )

    label, restored = load_checkpoint_decoder(path, parity_check, torch.device("cpu"))
    assert "roundtrip" in label and "GAT" in label
    torch.manual_seed(0)
    channel_llr = torch.randn(4, 49) * 2.0
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(channel_llr), restored(channel_llr))


def test_checkpoint_without_config_is_rejected(tmp_path, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    path = tmp_path / "bare.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="no embedded config"):
        load_checkpoint_decoder(path, parity_check, torch.device("cpu"))


def test_mcnemar_counts_only_discordant_frames() -> None:
    from ldpc_corrector.evaluate import mcnemar

    #                    frame: 0      1      2      3      4
    reference = torch.tensor([True,  True,  False, False, True])
    variant = torch.tensor([False, True,  True,  False, False])
    # frame 0: ref wrong, variant right -> fixed
    # frame 1: both wrong               -> concordant, ignored
    # frame 2: ref right, variant wrong -> broke
    # frame 3: both right               -> concordant, ignored
    # frame 4: ref wrong, variant right -> fixed
    stats = mcnemar(reference, variant)
    assert stats["paired_fixed"] == 2
    assert stats["paired_broke"] == 1
    assert stats["paired_net"] == 1
    assert stats["paired_z"] == pytest.approx(1 / np.sqrt(3))


def test_mcnemar_is_zero_when_decoders_agree() -> None:
    from ldpc_corrector.evaluate import mcnemar

    mask = torch.tensor([True, False, True, False])
    stats = mcnemar(mask, mask.clone())
    assert stats["paired_fixed"] == 0 and stats["paired_broke"] == 0
    assert stats["paired_z"] == 0.0


def test_mcnemar_sign_favours_the_variant() -> None:
    from ldpc_corrector.evaluate import mcnemar

    reference = torch.tensor([True] * 10)
    improved = torch.tensor([False] * 10)  # fixes everything
    assert mcnemar(reference, improved)["paired_z"] > 0
    # And the reverse is negative.
    assert mcnemar(improved, reference)["paired_z"] < 0


def test_mcnemar_requires_paired_frames() -> None:
    from ldpc_corrector.evaluate import mcnemar

    with pytest.raises(ValueError, match="equal frame counts"):
        mcnemar(torch.tensor([True, False]), torch.tensor([True]))


def test_collecting_failures_disables_early_stopping(generator, ldpc49_code) -> None:
    """Pairing needs identical frame counts, so min_frame_errors must be ignored."""
    _, parity_check = ldpc49_code
    rows, failures = sweep_decoder(
        BPBaseline(parity_check, 5),
        generator,
        [1.0],
        frames=2048,
        batch_size=1024,
        device=torch.device("cpu"),
        min_frame_errors=5,  # would stop almost immediately if honoured
        collect_failures=True,
    )
    assert rows[0]["frames"] == 2048
    assert failures[1.0].shape == (2048,)
    assert int(failures[1.0].sum()) == rows[0]["frame_errors"]


def test_more_bp_iterations_never_break_a_frame(generator, ldpc49_code) -> None:
    """BP(20) is a strict refinement of BP(5) here: it fixes, never breaks.

    Worth pinning: it is what separates more BP iterations from a learned
    corrector, which trades fixes against breaks.
    """
    from ldpc_corrector.evaluate import mcnemar

    _, parity_check = ldpc49_code
    kwargs = dict(
        generator=generator,
        ebn0_points=[3.0],
        frames=2048,
        batch_size=1024,
        device=torch.device("cpu"),
        collect_failures=True,
        seed=99,
    )
    _, bp5 = sweep_decoder(BPBaseline(parity_check, 5), **kwargs)
    _, bp20 = sweep_decoder(BPBaseline(parity_check, 20), **kwargs)

    stats = mcnemar(bp5[3.0], bp20[3.0])
    assert stats["paired_fixed"] > 0
    assert stats["paired_broke"] == 0, "BP(20) unexpectedly broke a BP(5) success"
    assert stats["paired_z"] > 1.96
