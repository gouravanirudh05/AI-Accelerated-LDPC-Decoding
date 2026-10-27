from __future__ import annotations

import numpy as np
import torch

from ldpc_corrector._compat import SumProductBP, LegacyTannerGraph
from ldpc_corrector.bp_trace import (
    CN_FEATURE_DIM,
    VN_FEATURE_DIM,
    build_cn_features,
    build_vn_features,
    derive_features,
    run_bp_with_features,
)
from ldpc_corrector.tanner_graph import build_tanner_graph

ITERATIONS = 5


def _bp(parity_check, iterations: int = ITERATIONS) -> SumProductBP:
    return SumProductBP(
        LegacyTannerGraph.from_parity_check(parity_check), iterations=iterations
    )


def test_trace_is_opt_in_and_does_not_change_the_decision(small_h) -> None:
    """The default return path must be byte-identical to Phase 0-2 behaviour."""
    bp = _bp(small_h)
    torch.manual_seed(0)
    channel_llr = torch.randn(4, small_h.shape[1]) * 3.0

    plain = bp(channel_llr)
    traced, trace = bp(channel_llr, return_trace=True)
    torch.testing.assert_close(plain, traced)

    # return_messages keeps working, alone and combined with return_trace.
    post, v2c, c2v = bp(channel_llr, return_messages=True)
    torch.testing.assert_close(post, plain)
    post2, v2c2, c2v2, trace2 = bp(channel_llr, return_messages=True, return_trace=True)
    torch.testing.assert_close(post2, plain)
    torch.testing.assert_close(v2c2, v2c)
    torch.testing.assert_close(c2v2, c2v)
    torch.testing.assert_close(trace2["llr_trace"], trace["llr_trace"])


def test_trace_shapes(small_h) -> None:
    n_checks, n_variables = small_h.shape
    bp = _bp(small_h)
    channel_llr = torch.randn(3, n_variables)
    _, trace = bp(channel_llr, return_trace=True)

    assert trace["channel_llr"].shape == (3, n_variables)
    assert trace["llr_trace"].shape == (3, ITERATIONS, n_variables)
    assert trace["check_parity_trace"].shape == (3, ITERATIONS, n_checks)


def test_last_trace_entry_is_the_final_posterior(small_h) -> None:
    bp = _bp(small_h)
    channel_llr = torch.randn(2, small_h.shape[1]) * 2.0
    posterior, trace = bp(channel_llr, return_trace=True)
    torch.testing.assert_close(trace["llr_trace"][:, -1], posterior)


def test_soft_parity_is_plus_one_on_a_valid_codeword(small_h) -> None:
    """A satisfied even-parity check has soft parity +1."""
    bp = _bp(small_h)
    # The all-zero codeword satisfies every check; large positive LLR => bit 0.
    confident = torch.full((1, small_h.shape[1]), 30.0)
    parity = bp.soft_parity(confident)
    torch.testing.assert_close(parity, torch.ones_like(parity), atol=1e-4, rtol=0)

    # Flipping one bit violates exactly the checks containing that bit.
    flipped = confident.clone()
    flipped[0, 1] = -30.0
    parity = bp.soft_parity(flipped)
    expected = torch.where(
        torch.as_tensor(small_h[:, 1] != 0), -torch.ones(4), torch.ones(4)
    )
    torch.testing.assert_close(parity[0], expected, atol=1e-4, rtol=0)


def test_oscillating_bit_gets_nonzero_sign_flip_count() -> None:
    """A hand-built oscillating trace must be detected, a settled one must not."""
    # Bit 0 alternates sign every iteration, bit 1 is stable, bit 2 flips once.
    trace = torch.tensor(
        [[[+1.0, +5.0, -5.0], [-1.0, +5.0, -5.0], [+1.0, +5.0, +5.0], [-1.0, +5.0, +5.0]]]
    )  # [B=1, K=4, N=3]
    derived = derive_features(trace)

    assert derived["sign_flip_count"].shape == (1, 3)
    assert derived["sign_flip_count"][0, 0].item() == 3.0  # oscillates every step
    assert derived["sign_flip_count"][0, 1].item() == 0.0  # settled
    assert derived["sign_flip_count"][0, 2].item() == 1.0  # one flip

    torch.testing.assert_close(derived["final_llr"][0], torch.tensor([-1.0, 5.0, 5.0]))
    torch.testing.assert_close(derived["confidence"][0], torch.tensor([1.0, 5.0, 5.0]))
    torch.testing.assert_close(
        derived["min_confidence"][0], torch.tensor([1.0, 5.0, 5.0])
    )


def test_derive_features_accepts_unbatched_trace() -> None:
    derived = derive_features(torch.tensor([[1.0, -2.0], [-1.0, -2.0]]))
    assert derived["sign_flip_count"].shape == (1, 2)
    assert derived["sign_flip_count"].tolist() == [[1.0, 0.0]]


def test_derive_features_handles_zero_iterations() -> None:
    derived = derive_features(torch.zeros(2, 0, 7))
    for key in ("sign_flip_count", "final_llr", "confidence", "min_confidence"):
        assert derived[key].shape == (2, 7)
        assert torch.all(derived[key] == 0)


def test_real_bp_produces_some_oscillation_at_low_snr(ldpc49_code) -> None:
    """On the real code at low SNR, BP genuinely fails to settle on some bits."""
    _, parity_check = ldpc49_code
    bp = _bp(parity_check, iterations=8)
    torch.manual_seed(0)
    # Eb/N0 near 1 dB: sigma is large, so BP oscillates on the weak bits.
    channel_llr = torch.randn(64, parity_check.shape[1]) * 1.2
    _, trace = bp(channel_llr, return_trace=True)
    flips = derive_features(trace["llr_trace"])["sign_flip_count"]
    assert flips.max().item() > 0.0, "expected unsettled bits at low SNR"


def test_feature_builders_shapes_and_finiteness(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    bp = _bp(parity_check)
    channel_llr = torch.randn(5, graph.n_variables) * 2.0
    _, trace = bp(channel_llr, return_trace=True)

    vn = build_vn_features(graph, trace)
    cn = build_cn_features(graph, trace)
    assert vn.shape == (5, graph.n_variables, VN_FEATURE_DIM)
    assert cn.shape == (5, graph.n_checks, CN_FEATURE_DIM)
    assert torch.isfinite(vn).all() and torch.isfinite(cn).all()


def test_features_stay_finite_at_extreme_llr(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    bp = _bp(parity_check)
    extreme = torch.full((2, graph.n_variables), 1e4)
    extreme[1] *= -1.0
    _, trace = bp(extreme, return_trace=True)
    assert torch.isfinite(build_vn_features(graph, trace)).all()
    assert torch.isfinite(build_cn_features(graph, trace)).all()


def test_run_bp_with_features_detaches_by_default(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    bp = _bp(parity_check)
    channel_llr = (torch.randn(3, graph.n_variables) * 2.0).requires_grad_(True)

    bp_llr, vn, cn, trace = run_bp_with_features(bp, graph, channel_llr)
    assert not bp_llr.requires_grad
    assert not vn.requires_grad and not cn.requires_grad
    assert "sign_flip_count" in trace and "llr_trace" in trace

    bp_llr, vn, _, _ = run_bp_with_features(bp, graph, channel_llr, detach=False)
    assert bp_llr.requires_grad and vn.requires_grad
