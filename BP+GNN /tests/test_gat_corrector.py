from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.bp_trace import (
    CN_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    VN_FEATURE_DIM,
    build_edge_features,
)
from ldpc_corrector.decoder import CorrectorDecoder
from ldpc_corrector.models.gat_corrector import GATCorrector
from ldpc_corrector.models.graph_ops import scatter_sum
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH = 3


@pytest.fixture
def gat_inputs(small_h):
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    vn = torch.randn(BATCH, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(BATCH, graph.n_checks, CN_FEATURE_DIM)
    edges = torch.randn(BATCH, graph.n_edges, EDGE_FEATURE_DIM)
    return graph, vn, cn, edges


def _model(graph, **kwargs) -> GATCorrector:
    return GATCorrector(
        graph=graph,
        vn_dim=VN_FEATURE_DIM,
        cn_dim=CN_FEATURE_DIM,
        hidden_dim=16,
        heads=2,
        head_dim=8,
        **kwargs,
    )


def test_legacy_graph_matches_the_new_graph(small_h) -> None:
    """The wrapped block indexes edges the same way the new graph does."""
    graph = build_tanner_graph(small_h)
    model = _model(graph)
    torch.testing.assert_close(model.legacy_graph.edge_variables, graph.edge_vn)
    torch.testing.assert_close(model.legacy_graph.edge_checks, graph.edge_cn)


def test_edge_exclusion_changes_the_output_but_not_the_parameters(gat_inputs) -> None:
    """The ablation must isolate aggregation: same weights, different result."""
    graph, vn, cn, edges = gat_inputs
    torch.manual_seed(7)
    standard = _model(graph, edge_exclusion=False).eval()
    torch.manual_seed(7)
    exclusive = _model(graph, edge_exclusion=True).eval()

    standard_params = dict(standard.named_parameters())
    exclusive_params = dict(exclusive.named_parameters())
    assert set(standard_params) == set(exclusive_params)
    for name, parameter in standard_params.items():
        torch.testing.assert_close(parameter, exclusive_params[name])

    standard_hidden = standard.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn,
                                      edge_features=edges)
    exclusive_hidden = exclusive.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn,
                                        edge_features=edges)
    assert not torch.allclose(standard_hidden, exclusive_hidden), (
        "edge_exclusion had no effect on the hidden state"
    )


def test_exclusive_aggregation_leaves_out_the_edges_own_message(small_h) -> None:
    """Directly verify excl(e) = (total - a_e v_e) / (1 - a_e)."""
    from ldpc_corrector._compat import DirectedEdgeGATv2
    from ldpc_corrector.models.gat_corrector import ConfigurableEdgeGATPass

    graph = build_tanner_graph(small_h)
    torch.manual_seed(3)
    hidden_dim = 8
    layer = ConfigurableEdgeGATPass(
        hidden_dim, EDGE_FEATURE_DIM, heads=1, head_dim=4, edge_exclusion=True
    )
    layer.eval()

    target = torch.randn(1, graph.n_checks, hidden_dim)
    source = torch.randn(1, graph.n_variables, hidden_dim)
    edges = torch.randn(1, graph.n_edges, EDGE_FEATURE_DIM)

    # Regroup edges by check, as the V->C direction does.
    order = graph.check_order
    check_index = graph.edge_cn.index_select(0, order)
    slices = []
    start = 0
    for check in range(graph.n_checks):
        count = int((check_index == check).sum())
        slices.append((start, start + count))
        start += count

    updated, attention = layer(
        target_hidden=target,
        source_hidden=source,
        edge_features=edges,
        source_index=graph.edge_vn,
        target_index=graph.edge_cn,
        target_slices=slices,
        n_targets=graph.n_checks,
        edge_order=order,
        return_attention=True,
    )
    assert updated.shape == (1, graph.n_checks, hidden_dim)
    assert torch.isfinite(updated).all()
    # Attention still normalizes over each target's incident edges.
    sums = scatter_sum(attention, check_index, graph.n_checks)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_degree_one_target_yields_zero_extrinsic_aggregate() -> None:
    """With one incident edge there is no extrinsic information to use."""
    from ldpc_corrector.models.gat_corrector import ConfigurableEdgeGATPass

    # A check with a single variable: attention saturates to 1, so 1 - a = 0.
    single = np.array([[1, 1], [1, 0]], dtype=np.int64)  # check 1 has degree 1
    graph = build_tanner_graph(single)
    layer = ConfigurableEdgeGATPass(
        4, EDGE_FEATURE_DIM, heads=1, head_dim=2, edge_exclusion=True
    ).eval()

    target = torch.randn(1, graph.n_checks, 4)
    source = torch.randn(1, graph.n_variables, 4)
    edges = torch.randn(1, graph.n_edges, EDGE_FEATURE_DIM)
    order = graph.check_order
    check_index = graph.edge_cn.index_select(0, order)
    slices, start = [], 0
    for check in range(graph.n_checks):
        count = int((check_index == check).sum())
        slices.append((start, start + count))
        start += count

    updated = layer(
        target_hidden=target,
        source_hidden=source,
        edge_features=edges,
        source_index=graph.edge_vn,
        target_index=graph.edge_cn,
        target_slices=slices,
        n_targets=graph.n_checks,
        edge_order=order,
    )
    assert torch.isfinite(updated).all(), "degree-1 target produced a non-finite update"


def test_attention_is_normalized_in_both_directions(gat_inputs) -> None:
    graph, vn, cn, edges = gat_inputs
    model = _model(graph).eval()
    attention = model.attention_maps(vn, cn, edges)

    v_to_c = scatter_sum(attention["v_to_c"], graph.edge_cn, graph.n_checks)
    c_to_v = scatter_sum(attention["c_to_v"], graph.edge_vn, graph.n_variables)
    torch.testing.assert_close(v_to_c, torch.ones_like(v_to_c), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_to_v, torch.ones_like(c_to_v), atol=1e-5, rtol=1e-5)


def test_edge_features_are_pulled_from_the_bp_trace(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(
        parity_check,
        corrector="gat",
        bp_iterations=4,
        corrector_kwargs={"hidden_dim": 16, "heads": 2, "head_dim": 8},
    )
    channel_llr = torch.randn(2, decoder.graph.n_variables) * 2.0
    _, trace = decoder.run_bp(channel_llr)

    assert "variable_to_check" in trace and "check_to_variable" in trace
    extra = decoder.corrector.extra_inputs(trace)
    assert extra["edge_features"].shape == (2, decoder.graph.n_edges, EDGE_FEATURE_DIM)
    assert torch.isfinite(extra["edge_features"]).all()


def test_missing_bp_messages_raises_a_clear_error(small_h) -> None:
    graph = build_tanner_graph(small_h)
    model = _model(graph)
    with pytest.raises(KeyError, match="needs the BP messages"):
        model.extra_inputs({"llr_trace": torch.zeros(1, 2, graph.n_variables)})


def test_edge_features_mirror_the_legacy_construction(ldpc49_code) -> None:
    """Same six channels as Phase 0-2, with the raw magnitudes rescaled."""
    from ldpc_corrector.bp_trace import LLR_SCALE

    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    torch.manual_seed(0)
    v2c = torch.randn(2, graph.n_edges)
    c2v = torch.randn(2, graph.n_edges)

    features = build_edge_features(v2c, c2v)
    torch.testing.assert_close(features[..., 0], v2c / LLR_SCALE)
    torch.testing.assert_close(features[..., 1], c2v / LLR_SCALE)
    torch.testing.assert_close(features[..., 2], v2c.abs() / LLR_SCALE)
    torch.testing.assert_close(features[..., 3], c2v.abs() / LLR_SCALE)
    torch.testing.assert_close(features[..., 4], torch.tanh(0.5 * v2c))
    torch.testing.assert_close(features[..., 5], torch.tanh(0.5 * c2v))


def test_wrong_edge_channel_count_is_rejected(gat_inputs) -> None:
    graph, vn, cn, edges = gat_inputs
    model = _model(graph)
    with pytest.raises(ValueError, match="edge channels"):
        model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges[..., :-1])
