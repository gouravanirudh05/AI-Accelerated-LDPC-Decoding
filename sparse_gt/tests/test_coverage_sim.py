import numpy as np
import torch

from sparse_gt.codes import load_code
from sparse_gt.coverage import _Accumulator, coverage_sweep, k_grid
from sparse_gt.simulate import fer_interval, read_aff3ct_reference, simulate_bp
from sparse_gt.codes import CODES_DIR


def test_accumulator_hand_example():
    # One frame, n=5, T=1. Output LLRs: bits 1 and 3 wrong (negative).
    out = torch.tensor([[3.0, -0.5, 1.0, -2.0, 0.2]])
    trace = torch.stack([torch.tensor([[3.0, 0.4, 1.0, -2.0, 0.2]]), out])
    acc = _Accumulator(n=5, m=2, T=1)
    acc.update(trace, stuck=torch.tensor([True]), unsat=torch.tensor([1]))
    # |LLR| ranks: 0.2->0, 0.5->1, 1.0->2, 2.0->3, 3.0->4. Errors at ranks 1 and 3.
    assert acc.err_rank["stuck"].tolist() == [0, 1, 0, 1, 0]
    assert acc.k_needed["stuck"][4] == 1  # need k=4 to cover both errors
    assert acc.failed == {"recoverable": 0, "stuck": 1}
    # Bit 1 flipped (+0.4 -> -0.5) and is an error; bit 3 never flipped.
    assert acc.flipped_err == 1 and acc.err_bits == 2
    s = acc.summary()
    cov = dict(zip(s["k"], s["frame_coverage"]["stuck"]))
    assert cov[4] == 1.0


def test_k_grid_bounds():
    for n in (204, 2048):
        ks = k_grid(n)
        assert ks == sorted(set(ks)) and ks[0] >= 1 and ks[-1] <= n


def test_coverage_sweep_runs_and_is_monotone():
    code = load_code("MACKAY_N204_K102")
    summaries, hists = coverage_sweep(code, [1.5], T_values=(2, 5), reference_iterations=20,
                                      batch=256, target_failed=50, max_frames=2048, log=None)
    assert len(summaries) == 2
    for s in summaries:
        cov = [c for c in s["frame_coverage"]["all"] if c is not None]
        assert all(b >= a for a, b in zip(cov, cov[1:]))
        assert s["failed_frames"] == sum(s["failed_by_group"].values())
    # Fewer iterations -> at least as many failures (same frames).
    by_T = {s["T"]: s for s in summaries}
    assert by_T[2]["failed_frames"] >= by_T[5]["failed_frames"]
    assert hists["1.5/2/err_rank_stuck"].shape == (code.n,)


def test_simulate_bp_smoke():
    code = load_code("MACKAY_N204_K102")
    rows = simulate_bp(code, [1.0], iterations=10, batch=256, min_frame_errors=10,
                       max_frames=1024, log=None)
    assert 0 < rows[0]["fer"] <= 1 and rows[0]["ber"] < 0.2


def test_reference_parser():
    rows = read_aff3ct_reference(CODES_DIR / "references" / "LDPC_N576_K288_flooding_SPA_i100_WiMAX.txt")
    assert rows[0]["ebn0"] == 0.0 and rows[0]["frames"] == 148
    point = next(r for r in rows if r["ebn0"] == 2.0)
    assert point["fer"] == 1.72e-02 and point["frame_errors"] == 108


def test_fer_interval_contains_estimate():
    lo, hi = fer_interval(100, 10_000)
    assert lo < 0.01 < hi and np.isclose(lo, 0.0082, atol=5e-4)


def test_inference_timing_rows():
    from sparse_gt.timing import inference_timing, select_least_confident

    code = load_code("MACKAY_N204_K102")
    rows = inference_timing(code, "cpu", batches=(1, 8), T_values=(2,), k_values=(10,),
                            early_stop_snr=3.0, early_stop_iterations=10, repeats=2)
    kinds = {(r["kind"], r["batch"]) for r in rows}
    assert kinds == {(k, b) for k in ("bp_fixed", "topk", "bp_early_stop") for b in (1, 8)}
    assert all(r["ms"] > 0 for r in rows)
    llr = torch.tensor([[3.0, -0.1, 2.0, 0.5, -4.0]])
    assert sorted(select_least_confident(llr, 2)[0].tolist()) == [1, 3]
