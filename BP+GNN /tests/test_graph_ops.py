from __future__ import annotations

import torch

from ldpc_corrector.models.graph_ops import (
    gather_nodes,
    scatter_max,
    scatter_mean,
    scatter_sum,
    segment_softmax,
)
from ldpc_corrector.tanner_graph import build_tanner_graph


def test_scatter_sum_matches_explicit_loop(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    edges = torch.randn(3, graph.n_edges, 5)

    expected = torch.zeros(3, graph.n_variables, 5)
    for e in range(graph.n_edges):
        expected[:, graph.edge_vn[e]] += edges[:, e]
    torch.testing.assert_close(
        scatter_sum(edges, graph.edge_vn, graph.n_variables), expected
    )


def test_scatter_mean_uses_degrees(small_h) -> None:
    graph = build_tanner_graph(small_h)
    edges = torch.ones(2, graph.n_edges, 3)
    result = scatter_mean(edges, graph.edge_cn, graph.n_checks, graph.cn_degrees)
    torch.testing.assert_close(result, torch.ones_like(result))
    # Same answer when degrees are inferred rather than supplied.
    torch.testing.assert_close(
        scatter_mean(edges, graph.edge_cn, graph.n_checks), result
    )


def test_scatter_max(small_h) -> None:
    graph = build_tanner_graph(small_h)
    edges = torch.arange(graph.n_edges, dtype=torch.float32).view(1, -1, 1)
    result = scatter_max(edges, graph.edge_vn, graph.n_variables)
    # Edges are grouped by variable, so each variable's max is its last edge.
    for v in range(graph.n_variables):
        last = int((graph.edge_vn == v).nonzero()[-1])
        assert result[0, v, 0].item() == float(last)


def test_gather_is_scatter_adjoint(small_h) -> None:
    graph = build_tanner_graph(small_h)
    nodes = torch.randn(2, graph.n_variables, 4)
    gathered = gather_nodes(nodes, graph.edge_vn)
    assert gathered.shape == (2, graph.n_edges, 4)
    for e in range(graph.n_edges):
        torch.testing.assert_close(gathered[:, e], nodes[:, graph.edge_vn[e]])


def test_segment_softmax_normalizes_per_target_node(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    logits = torch.randn(4, graph.n_edges, 3) * 5.0
    attention = segment_softmax(logits, graph.edge_vn, graph.n_variables)

    sums = scatter_sum(attention, graph.edge_vn, graph.n_variables)
    torch.testing.assert_close(sums, torch.ones_like(sums))
    assert torch.all(attention >= 0)


def test_segment_softmax_is_numerically_stable(small_h) -> None:
    graph = build_tanner_graph(small_h)
    logits = torch.full((1, graph.n_edges, 1), 1e4)
    logits[0, 0, 0] = -1e4
    attention = segment_softmax(logits, graph.edge_cn, graph.n_checks)
    assert torch.isfinite(attention).all()
    sums = scatter_sum(attention, graph.edge_cn, graph.n_checks)
    torch.testing.assert_close(sums, torch.ones_like(sums))
