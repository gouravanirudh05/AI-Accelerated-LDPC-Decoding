"""Per-bit vs per-edge residual (``residual_mode``).

``node`` emits one bounded correction per variable node, so ``|delta| <=
correction_scale``.  ``edge`` emits one bounded correction per *edge* and sums
them onto the variable, so a degree-d bit can be moved by up to
``d * correction_scale`` and each contribution is conditioned on that edge's own
BP messages.  The Phase 0-2 model uses the edge form.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.bp_trace import CN_FEATURE_DIM, EDGE_FEATURE_DIM, VN_FEATURE_DIM
from ldpc_corrector.decoder import CorrectorDecoder
from ldpc_corrector.models import build_corrector
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH = 3
EDGE_CAPABLE = ("gat", "local_gt")


def _model(name, graph, **kwargs):
    base = {
        "graph": graph,
        "vn_dim": VN_FEATURE_DIM,
        "cn_dim": CN_FEATURE_DIM,
        "hidden_dim": 16,
        "heads": 2,
        "head_dim": 8,
    }
    return build_corrector(name, **{**base, **kwargs})


@pytest.fixture
def inputs(small_h):
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    return (
        graph,
        torch.randn(BATCH, graph.n_variables, VN_FEATURE_DIM),
        torch.randn(BATCH, graph.n_checks, CN_FEATURE_DIM),
        torch.randn(BATCH, graph.n_edges, EDGE_FEATURE_DIM),
    )


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_untrained_edge_residual_is_still_exactly_bp(name, inputs) -> None:
    graph, vn, cn, edges = inputs
    model = _model(name, graph, residual_mode="edge")
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    torch.testing.assert_close(delta, torch.zeros_like(delta))


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_edge_residual_can_exceed_the_per_node_bound(name, inputs) -> None:
    """The point of the edge form: enough authority to flip a confident bit."""
    graph, vn, cn, edges = inputs
    scale = 2.0
    model = _model(name, graph, residual_mode="edge", correction_scale=scale)
    with torch.no_grad():
        # Zero weight plus a large bias saturates every per-edge tanh to +1,
        # independently of the hidden state, so the sum is exactly degree*scale.
        model.edge_delta_head[-1].weight.zero_()
        model.edge_delta_head[-1].bias.fill_(50.0)

    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    # Saturated, every incident edge contributes +scale, so delta == degree*scale.
    expected = graph.vn_degrees * scale
    torch.testing.assert_close(delta[0], expected, atol=1e-3, rtol=1e-3)
    assert delta.abs().max() > scale, "edge residual should beat the per-node bound"


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_node_residual_stays_within_the_per_node_bound(name, inputs) -> None:
    graph, vn, cn, edges = inputs
    scale = 2.0
    model = _model(name, graph, residual_mode="node", correction_scale=scale)
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 50.0)
        model.delta_head.bias.fill_(50.0)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    assert delta.abs().max() <= scale + 1e-6


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_edge_residual_is_conditioned_on_the_edge_features(name, inputs) -> None:
    """Each edge's correction must actually depend on that edge's messages."""
    graph, vn, cn, edges = inputs
    model = _model(name, graph, residual_mode="edge").eval()
    with torch.no_grad():
        model.edge_delta_head[-1].weight.normal_(0.0, 0.5)

    first = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    second = model(
        vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges + 1.0
    )
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_edge_mode_gradients_reach_the_edge_head(name, inputs) -> None:
    graph, vn, cn, edges = inputs
    model = _model(name, graph, residual_mode="edge")
    with torch.no_grad():
        model.edge_delta_head[-1].weight.normal_(0.0, 0.1)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    delta.square().mean().backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{parameter_name} got no gradient"
        assert torch.isfinite(parameter.grad).all()
    assert model.edge_delta_head[-1].weight.grad.abs().sum() > 0


def test_edge_mode_costs_more_flops_than_node_mode(small_h) -> None:
    graph = build_tanner_graph(small_h)
    node = _model("gat", graph, residual_mode="node")
    edge = _model("gat", graph, residual_mode="edge")
    assert edge.estimate_flops_per_frame() > node.estimate_flops_per_frame()
    assert edge.num_parameters() > node.num_parameters()


@pytest.mark.parametrize("name", ["gcn", "sparse_global_gt"])
def test_architectures_without_edge_context_reject_edge_mode(name, small_h) -> None:
    """No per-edge context to condition on: fail loudly, not quietly."""
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="edge_dim"):
        build_corrector(
            name, graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            hidden_dim=16, residual_mode="edge",
        )


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_edge_mode_has_no_unused_node_head(name, small_h) -> None:
    """A dead head would silently take no gradient; there must not be one."""
    graph = build_tanner_graph(small_h)
    assert _model(name, graph, residual_mode="edge").delta_head is None
    assert _model(name, graph, residual_mode="node").edge_delta_head is None


def test_invalid_residual_mode_is_rejected(small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="residual_mode must be"):
        _model("gat", graph, residual_mode="sideways")


def test_end_to_end_edge_mode_on_the_real_code(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(
        parity_check,
        corrector="gat",
        bp_iterations=5,
        corrector_kwargs={
            "hidden_dim": 32, "heads": 2, "head_dim": 16, "residual_mode": "edge",
        },
    )
    final_llr, aux = decoder(torch.randn(4, 49) * 2.0, return_aux=True)
    torch.testing.assert_close(final_llr, aux["bp_llr"])  # untrained == BP

    # Parameter count and cost should land near the Aug-26 legacy model, which
    # uses the same edge-wise formulation.
    assert 15_000 < decoder.corrector.num_parameters() < 30_000


@pytest.mark.parametrize("name", EDGE_CAPABLE)
def test_effective_range_exposes_the_mode_dependent_bound(name, small_h) -> None:
    """``correction_scale`` is per-edge in edge mode, so the real bound is
    ``max_degree * scale``. Exposed because conflating the two silently
    over-powers the corrector."""
    graph = build_tanner_graph(small_h)
    max_degree = float(graph.vn_degrees.max())

    node = _model(name, graph, residual_mode="node", correction_scale=8.0)
    assert node.effective_correction_range == pytest.approx(8.0)

    edge = _model(name, graph, residual_mode="edge", correction_scale=8.0)
    assert edge.effective_correction_range == pytest.approx(8.0 * max_degree)

    # Dividing the scale by the degree restores the node-mode bound, which is
    # how the edge configs are set.
    matched = _model(
        name, graph, residual_mode="edge", correction_scale=8.0 / max_degree
    )
    assert matched.effective_correction_range == pytest.approx(8.0)


def test_shipped_edge_configs_keep_a_sane_effective_range() -> None:
    """Guard the actual configs, not just the mechanism."""
    from pathlib import Path

    from ldpc_corrector.ablate import load_sweep, resolve_variant

    sweep_path = (
        Path(__file__).resolve().parents[1] / "configs" / "ablations.yaml"
    )
    sweep = load_sweep(sweep_path)
    common = sweep.get("common") or []
    checked = 0
    for variant in sweep["variants"]:
        overrides = " ".join(variant.get("overrides") or [])
        if "residual_mode=edge" not in overrides:
            continue
        config = resolve_variant(variant, common, sweep_path.parent)
        scale = config.model.correction_scale
        # Mean variable degree on these codes is ~4, so anything above ~4
        # implies an effective range past +/-16 and over-corrects.
        assert scale <= 4.0, (
            f"{config.name}: correction_scale={scale} is a node-mode value; "
            "edge mode sums over incident edges"
        )
        checked += 1
    assert checked >= 3, "expected the edge-mode variants to be covered"
