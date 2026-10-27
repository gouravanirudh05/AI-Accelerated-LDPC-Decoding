from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.losses import (
    TERM_NAMES,
    CompositeLoss,
    LossConfig,
    bce_loss,
    block_soft_bler_loss,
    build_loss,
    deep_supervision_loss,
    focal_loss,
    hard_mining_loss,
    syndrome_loss,
)
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH, N_BITS = 6, 8


@pytest.fixture
def dummy_batch(small_h):
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    final_llr = (torch.randn(BATCH, N_BITS) * 3.0).requires_grad_(True)
    bp_llr = torch.randn(BATCH, N_BITS) * 3.0
    target = torch.randint(0, 2, (BATCH, N_BITS)).float()
    stage_llrs = [final_llr * 0.5, final_llr]
    return graph, final_llr, bp_llr, target, stage_llrs


@pytest.mark.parametrize("term", TERM_NAMES)
def test_each_term_alone_is_finite_scalar_and_differentiable(term, dummy_batch) -> None:
    """Every term, enabled on its own, must give one usable gradient signal."""
    graph, final_llr, bp_llr, target, stage_llrs = dummy_batch

    # syndrome_aux is barred from being the sole term, so pair it with bce and
    # read its own contribution out of the breakdown.
    flags = {name: False for name in TERM_NAMES}
    flags[term] = True
    # syndrome_aux is barred from standing alone, so pair it with bce.
    if term == "syndrome_aux":
        flags["bce"] = True
    config = LossConfig(**flags)
    loss_fn = CompositeLoss(config, graph)

    total, breakdown = loss_fn(final_llr, target, bp_llr, stage_llrs)

    assert total.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(total), f"{term} produced a non-finite loss"
    assert total.requires_grad
    assert term in breakdown and np.isfinite(breakdown[term])

    total.backward()
    assert final_llr.grad is not None
    assert torch.isfinite(final_llr.grad).all()
    assert final_llr.grad.abs().sum() > 0, f"{term} produced a zero gradient"


def test_breakdown_reports_only_enabled_terms(dummy_batch) -> None:
    graph, final_llr, bp_llr, target, stage_llrs = dummy_batch
    config = LossConfig(bce=True, focal=True, syndrome_aux=True)
    _, breakdown = CompositeLoss(config, graph)(final_llr, target, bp_llr, stage_llrs)
    assert set(breakdown) == {"bce", "focal", "syndrome_aux", "total"}


def test_total_is_the_weighted_sum_of_its_parts(dummy_batch) -> None:
    graph, final_llr, bp_llr, target, stage_llrs = dummy_batch
    config = LossConfig(
        bce=True, bce_weight=0.75, focal=True, focal_weight=2.5,
        syndrome_aux=True, syndrome_aux_weight=0.1,
    )
    total, parts = CompositeLoss(config, graph)(final_llr, target, bp_llr, stage_llrs)
    expected = (
        0.75 * parts["bce"] + 2.5 * parts["focal"] + 0.1 * parts["syndrome_aux"]
    )
    assert total.item() == pytest.approx(expected, rel=1e-5)
    assert parts["total"] == pytest.approx(total.item(), rel=1e-6)


def test_bce_matches_legacy_convention(dummy_batch) -> None:
    """Positive LLR must mean bit 0, as in gat_bp.bce_from_llr."""
    from ldpc_corrector._compat import bce_from_llr

    _, final_llr, _, target, _ = dummy_batch
    torch.testing.assert_close(
        bce_loss(final_llr, target), bce_from_llr(final_llr, target)
    )


def test_bce_rewards_correct_confident_decisions() -> None:
    target = torch.tensor([[0.0, 1.0]])
    correct = torch.tensor([[10.0, -10.0]])  # positive => bit 0
    wrong = torch.tensor([[-10.0, 10.0]])
    assert bce_loss(correct, target).item() < bce_loss(wrong, target).item()


def test_focal_downweights_easy_bits() -> None:
    """With gamma > 0 the easy bits contribute far less than under plain BCE."""
    target = torch.zeros(1, 2)
    llr = torch.tensor([[12.0, 0.2]])  # one very easy bit, one borderline
    plain = bce_loss(llr, target)
    focal = focal_loss(llr, target, gamma=2.0)
    assert focal.item() < plain.item()
    # gamma=0 degenerates to BCE.
    torch.testing.assert_close(focal_loss(llr, target, gamma=0.0), plain)


def test_hard_mining_selects_wrong_or_uncertain_bits() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0]])
    # bit 0: BP right and confident -> excluded.
    # bit 1: BP wrong -> included.
    # bit 2: BP right but unconfident -> included.
    bp_llr = torch.tensor([[9.0, -9.0, 0.2]])
    final_llr = torch.tensor([[5.0, 5.0, 5.0]])

    mined = hard_mining_loss(final_llr, target, bp_llr, threshold=1.0)
    from ldpc_corrector.losses import per_bit_bce

    per_bit = per_bit_bce(final_llr, target)
    torch.testing.assert_close(mined, per_bit[0, 1:].mean())

    # Nothing selected: with every bit right and confident the term is zero,
    # not NaN from an empty-mask division.
    safe = hard_mining_loss(
        torch.tensor([[9.0]]), torch.zeros(1, 1), torch.tensor([[9.0]]), threshold=1.0
    )
    assert torch.isfinite(safe)


def test_block_soft_bler_tracks_the_worst_bit() -> None:
    target = torch.zeros(2, 4)
    good = torch.full((2, 4), 8.0)
    one_bad = good.clone()
    one_bad[0, 2] = -8.0  # a single wrong bit ruins the whole block

    assert block_soft_bler_loss(one_bad, target).item() > block_soft_bler_loss(
        good, target
    ).item()
    # Per-bit BCE barely moves for one bad bit out of four; the block term must
    # react much more strongly. That is the entire point of the term.
    bit_ratio = bce_loss(one_bad, target) / bce_loss(good, target)
    block_ratio = block_soft_bler_loss(one_bad, target) / block_soft_bler_loss(
        good, target
    )
    assert block_ratio.item() > bit_ratio.item()


def test_syndrome_loss_is_minimal_on_a_valid_codeword(small_h) -> None:
    graph = build_tanner_graph(small_h)
    all_zero = torch.full((1, graph.n_variables), 20.0)  # valid codeword
    invalid = all_zero.clone()
    invalid[0, 0] = -20.0  # violates every check containing bit 0
    assert syndrome_loss(all_zero, graph).item() < syndrome_loss(invalid, graph).item()
    assert syndrome_loss(all_zero, graph).item() == pytest.approx(0.0, abs=1e-4)


def test_syndrome_loss_matches_the_legacy_implementation(ldpc49_code) -> None:
    """The vectorized log-domain product must agree with the per-check loop."""
    from ldpc_corrector._compat import train_gat_bp

    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    torch.manual_seed(0)
    llr = torch.randn(4, graph.n_variables) * 2.0

    legacy = train_gat_bp.parity_consistency_loss(
        llr, torch.as_tensor(parity_check, dtype=torch.float32)
    )
    torch.testing.assert_close(syndrome_loss(llr, graph), legacy, atol=1e-5, rtol=1e-4)


def test_product_to_cn_handles_zeros_and_signs(small_h) -> None:
    graph = build_tanner_graph(small_h)
    edge_values = torch.ones(1, graph.n_edges)
    torch.testing.assert_close(
        graph.product_to_cn(edge_values), torch.ones(1, graph.n_checks)
    )

    # One negative edge flips exactly its own check's sign.
    edge_values[0, 0] = -1.0
    product = graph.product_to_cn(edge_values)
    flipped = int(graph.edge_cn[0])
    assert product[0, flipped].item() == pytest.approx(-1.0)

    # One exact zero zeroes exactly its own check.
    edge_values = torch.ones(1, graph.n_edges)
    edge_values[0, 3] = 0.0
    product = graph.product_to_cn(edge_values)
    zeroed = int(graph.edge_cn[3])
    assert product[0, zeroed].item() == pytest.approx(0.0)


def test_deep_supervision_weights_the_last_stage_most() -> None:
    target = torch.zeros(1, 2)
    good, bad = torch.full((1, 2), 8.0), torch.full((1, 2), -8.0)
    # Same stages, opposite order: the version ending well must score better.
    ends_well = deep_supervision_loss([bad, good], target, gamma=0.5)
    ends_badly = deep_supervision_loss([good, bad], target, gamma=0.5)
    assert ends_well.item() < ends_badly.item()

    # A single stage reduces to plain BCE regardless of gamma.
    torch.testing.assert_close(
        deep_supervision_loss([good], target, gamma=0.3), bce_loss(good, target)
    )


def test_deep_supervision_requires_stages(dummy_batch) -> None:
    graph, final_llr, bp_llr, target, _ = dummy_batch
    loss_fn = CompositeLoss(LossConfig(bce=True, deep_supervision=True), graph)
    with pytest.raises(ValueError, match="no stage LLRs"):
        loss_fn(final_llr, target, bp_llr, None)


def test_hard_mining_requires_bp_llr(dummy_batch) -> None:
    graph, final_llr, _, target, stages = dummy_batch
    loss_fn = CompositeLoss(LossConfig(bce=True, hard_mining=True), graph)
    with pytest.raises(ValueError, match="needs the BP"):
        loss_fn(final_llr, target, None, stages)


def test_syndrome_aux_cannot_be_the_only_term() -> None:
    with pytest.raises(ValueError, match="auxiliary only"):
        LossConfig(bce=False, syndrome_aux=True).validate()


def test_config_rejects_empty_and_invalid_settings() -> None:
    with pytest.raises(ValueError, match="At least one loss term"):
        LossConfig(bce=False).validate()
    with pytest.raises(ValueError, match="deep_supervision_gamma"):
        LossConfig(deep_supervision_gamma=0.0).validate()
    with pytest.raises(ValueError, match="temperature"):
        LossConfig(block_soft_bler_temperature=0.0).validate()
    with pytest.raises(KeyError, match="Unknown loss config keys"):
        LossConfig.from_mapping({"bce": True, "nonsense": 1})


def test_syndrome_aux_needs_a_graph() -> None:
    with pytest.raises(ValueError, match="needs the Tanner graph"):
        build_loss({"bce": True, "syndrome_aux": True}, graph=None)


def test_all_terms_together_are_finite(dummy_batch) -> None:
    graph, final_llr, bp_llr, target, stage_llrs = dummy_batch
    config = LossConfig(**{term: True for term in TERM_NAMES})
    total, breakdown = CompositeLoss(config, graph)(
        final_llr, target, bp_llr, stage_llrs
    )
    assert torch.isfinite(total)
    assert set(breakdown) == set(TERM_NAMES) | {"total"}
    total.backward()
    assert torch.isfinite(final_llr.grad).all()


def test_terms_stay_finite_at_extreme_llr(small_h) -> None:
    graph = build_tanner_graph(small_h)
    target = torch.randint(0, 2, (2, N_BITS)).float()
    for magnitude in (0.0, 1e3, 1e6):
        llr = torch.full((2, N_BITS), magnitude).requires_grad_(True)
        config = LossConfig(**{term: True for term in TERM_NAMES})
        total, _ = CompositeLoss(config, graph)(llr, target, llr.detach(), [llr])
        assert torch.isfinite(total), f"non-finite at |LLR|={magnitude}"
        total.backward()
        assert torch.isfinite(llr.grad).all(), f"non-finite grad at |LLR|={magnitude}"
