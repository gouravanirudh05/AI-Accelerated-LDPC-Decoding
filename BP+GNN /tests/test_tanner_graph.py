from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.tanner_graph import build_tanner_graph


def test_edge_count_matches_nnz(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    assert graph.n_edges == int(np.count_nonzero(small_h))
    assert graph.vn_to_cn.shape == (2, graph.n_edges)
    assert graph.cn_to_vn.shape == (2, graph.n_edges)


def test_degrees_match_row_and_column_sums(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    np.testing.assert_allclose(graph.vn_degrees.numpy(), small_h.sum(axis=0))
    np.testing.assert_allclose(graph.cn_degrees.numpy(), small_h.sum(axis=1))


def test_directed_indices_are_transposes_not_one_undirected_set(
    small_h: np.ndarray,
) -> None:
    """The two directions must stay separate, with source/target swapped."""
    graph = build_tanner_graph(small_h)
    torch.testing.assert_close(graph.vn_to_cn[0], graph.cn_to_vn[1])
    torch.testing.assert_close(graph.vn_to_cn[1], graph.cn_to_vn[0])
    # Every directed edge reproduces a nonzero of H, and nothing else does.
    recovered = np.zeros_like(small_h)
    recovered[graph.edge_cn.numpy(), graph.edge_vn.numpy()] = 1
    np.testing.assert_array_equal(recovered, (small_h != 0).astype(recovered.dtype))


def test_dense_and_sparse_inputs_agree(small_h: np.ndarray) -> None:
    sp = pytest.importorskip("scipy.sparse")
    dense = build_tanner_graph(small_h)
    sparse = build_tanner_graph(sp.csr_matrix(small_h))
    torch_sparse = build_tanner_graph(torch.as_tensor(small_h).float().to_sparse())
    for other in (sparse, torch_sparse):
        assert other.n_edges == dense.n_edges
        torch.testing.assert_close(other.edge_vn, dense.edge_vn)
        torch.testing.assert_close(other.edge_cn, dense.edge_cn)
        torch.testing.assert_close(other.vn_degrees, dense.vn_degrees)


def test_to_dense_round_trips(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    np.testing.assert_array_equal(
        graph.to_dense().numpy().astype(np.int64), (small_h != 0).astype(np.int64)
    )


def test_aggregation_matches_dense_matmul(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    edge_values = torch.randn(3, graph.n_edges)

    # Sum over each variable's incident edges, computed the slow explicit way.
    expected_vn = torch.zeros(3, graph.n_variables)
    expected_cn = torch.zeros(3, graph.n_checks)
    for e in range(graph.n_edges):
        expected_vn[:, graph.edge_vn[e]] += edge_values[:, e]
        expected_cn[:, graph.edge_cn[e]] += edge_values[:, e]

    torch.testing.assert_close(graph.aggregate_to_vn(edge_values), expected_vn)
    torch.testing.assert_close(graph.aggregate_to_cn(edge_values), expected_cn)
    torch.testing.assert_close(
        graph.aggregate_to_vn(edge_values, reduce="mean"),
        expected_vn / graph.vn_degrees,
    )


def test_gather_is_the_adjoint_of_scatter(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    node_values = torch.arange(graph.n_variables, dtype=torch.float32).unsqueeze(0)
    gathered = graph.gather_vn(node_values)
    assert gathered.shape == (1, graph.n_edges)
    torch.testing.assert_close(gathered[0], graph.edge_vn.float())


def test_two_hop_index_is_cached_and_symmetric(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    first = graph.two_hop_index("vn")
    assert first is graph.two_hop_index("vn")  # cached, not recomputed

    # Two variables are 2-hop neighbours exactly when they share a check.
    shared = (small_h.T @ small_h).astype(np.int64)
    np.fill_diagonal(shared, 0)
    expected = set(zip(*np.nonzero(shared)))
    actual = set(zip(first[0].tolist(), first[1].tolist()))
    assert actual == expected
    assert all((b, a) in actual for a, b in actual)


def test_edge_order_aligns_with_legacy_bp(ldpc49_code) -> None:
    """A [B, E] message tensor from the Phase 0-2 BP must index-align."""
    from ldpc_corrector._compat import LegacyTannerGraph

    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    legacy = LegacyTannerGraph.from_parity_check(parity_check)

    assert graph.n_edges == legacy.n_edges
    torch.testing.assert_close(graph.edge_vn, legacy.edge_variables)
    torch.testing.assert_close(graph.edge_cn, legacy.edge_checks)
    torch.testing.assert_close(graph.check_order, legacy.check_order)


def test_rejects_degenerate_matrices() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        build_tanner_graph(np.ones(5))
    with pytest.raises(ValueError, match="no Tanner-graph edges"):
        build_tanner_graph(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="variable node"):
        build_tanner_graph(np.array([[1, 0], [1, 0]]))
    with pytest.raises(ValueError, match="check node"):
        build_tanner_graph(np.array([[1, 1], [0, 0]]))


def test_real_code_shapes(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    assert (graph.n_variables, graph.n_checks) == (49, 28)
    assert graph.n_edges == int(np.count_nonzero(parity_check))


def test_mean_aggregation_is_correct_for_a_square_h() -> None:
    """With n == m, the reduction must not guess which degrees to divide by."""
    square = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    graph = build_tanner_graph(square)
    assert graph.n_variables == graph.n_checks == 4
    # Row and column sums differ, so a wrong pick would show up numerically.
    assert graph.vn_degrees.tolist() != graph.cn_degrees.tolist()

    edge_values = torch.ones(1, graph.n_edges)
    torch.testing.assert_close(
        graph.aggregate_to_vn(edge_values, reduce="mean"),
        torch.ones(1, graph.n_variables),
    )
    torch.testing.assert_close(
        graph.aggregate_to_cn(edge_values, reduce="mean"),
        torch.ones(1, graph.n_checks),
    )

    # And the sums equal the respective degrees, not each other's.
    torch.testing.assert_close(
        graph.aggregate_to_vn(edge_values)[0], graph.vn_degrees
    )
    torch.testing.assert_close(
        graph.aggregate_to_cn(edge_values)[0], graph.cn_degrees
    )


def test_to_device_moves_every_index_tensor(small_h: np.ndarray) -> None:
    graph = build_tanner_graph(small_h)
    graph.two_hop_index("vn")  # populate the cache before moving
    moved = graph.to("cpu")
    for name in ("edge_vn", "edge_cn", "vn_to_cn", "cn_to_vn", "check_order"):
        assert getattr(moved, name).device.type == "cpu"
    # The cached 2-hop index survives the move rather than being dropped.
    assert moved._two_hop_vn is not None
    torch.testing.assert_close(moved.two_hop_index("vn"), graph.two_hop_index("vn"))
