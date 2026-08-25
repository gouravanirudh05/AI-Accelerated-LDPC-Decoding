from __future__ import annotations

import torch

from gat_bp import (
    BidirectionalEdgeGATv2,
    EdgeAwareGAT,
    HybridBPGAT,
    SumProductBP,
    TannerGraph,
)
from train_gat_bp import (
    create_fixed_validation_dataset,
    evaluate,
)


def small_graph() -> torch.Tensor:
    # H = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]].
    return torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )


def small_generator() -> torch.Tensor:
    # Codewords have the form [a, a, a, b] and satisfy small_graph() @ c = 0.
    return torch.tensor(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def test_tanner_edges_match_nonzero_parity_entries() -> None:
    parity_check = small_graph()
    graph = TannerGraph.from_parity_check(parity_check, positional_dim=0)

    actual = set(zip(graph.edge_checks.tolist(), graph.edge_variables.tolist()))
    expected = {
        (check, variable)
        for check in range(parity_check.shape[0])
        for variable in range(parity_check.shape[1])
        if parity_check[check, variable] != 0
    }
    assert actual == expected
    assert graph.n_edges == len(expected)


def test_bp_shapes_gradients_and_high_magnitude_finiteness() -> None:
    graph = TannerGraph.from_parity_check(small_graph(), positional_dim=0)
    bp = SumProductBP(graph, iterations=3)

    channel_llr = torch.randn(5, graph.n_variables, requires_grad=True)
    posterior, v2c, c2v = bp(channel_llr, return_messages=True)
    assert posterior.shape == (5, graph.n_variables)
    assert v2c.shape == (5, graph.n_edges)
    assert c2v.shape == (5, graph.n_edges)
    assert torch.isfinite(posterior).all()
    assert torch.isfinite(v2c).all()
    assert torch.isfinite(c2v).all()

    posterior.square().mean().backward()
    assert channel_llr.grad is not None
    assert torch.isfinite(channel_llr.grad).all()

    extreme = torch.tensor([[100.0, -100.0, 80.0, -80.0]])
    extreme_output = bp(extreme)
    assert torch.isfinite(extreme_output).all()


def test_attention_is_normalized_over_each_variable_neighborhood() -> None:
    graph = TannerGraph.from_parity_check(small_graph(), positional_dim=0)
    attention_layer = EdgeAwareGAT(
        graph,
        variable_dim=4,
        check_dim=5,
        edge_dim=4,
        hidden_dim=8,
        heads=2,
        head_dim=4,
        dropout=0.0,
    ).eval()
    variable_features = torch.randn(3, graph.n_variables, 4)
    check_features = torch.randn(3, graph.n_checks, 5)
    edge_features = torch.randn(3, graph.n_edges, 4)

    _, attention = attention_layer(
        variable_features,
        check_features,
        edge_features,
        return_attention=True,
    )
    assert attention.shape == (3, graph.n_edges, 2)
    for start, end in graph.variable_slices:
        assert torch.allclose(
            attention[:, start:end].sum(dim=1),
            torch.ones(3, 2),
            atol=1e-6,
        )


def test_bidirectional_attention_is_normalized_in_both_directions() -> None:
    graph = TannerGraph.from_parity_check(small_graph(), positional_dim=0)
    attention_layer = BidirectionalEdgeGATv2(
        graph,
        variable_dim=5,
        check_dim=6,
        edge_dim=6,
        hidden_dim=8,
        heads=2,
        head_dim=4,
        dropout=0.0,
    ).eval()
    variable_features = torch.randn(3, graph.n_variables, 5)
    check_features = torch.randn(3, graph.n_checks, 6)
    edge_features = torch.randn(3, graph.n_edges, 6)

    _, _, attention = attention_layer(
        variable_features,
        check_features,
        edge_features,
        return_attention=True,
    )
    v_to_c = attention["v_to_c"].index_select(1, graph.check_order)
    c_to_v = attention["c_to_v"]
    for start, end in graph.check_slices:
        assert torch.allclose(
            v_to_c[:, start:end].sum(dim=1),
            torch.ones(3, 2),
            atol=1e-6,
        )
    for start, end in graph.variable_slices:
        assert torch.allclose(
            c_to_v[:, start:end].sum(dim=1),
            torch.ones(3, 2),
            atol=1e-6,
        )


def test_zero_initialized_correction_is_exact_bp() -> None:
    torch.manual_seed(7)
    parity_check = small_graph()
    model = HybridBPGAT(
        parity_check,
        bp_iterations=3,
        positional_dim=0,
        hidden_dim=8,
        heads=2,
        head_dim=4,
    ).eval()
    channel_llr = torch.randn(4, parity_check.shape[1])

    final_llr, auxiliary = model(channel_llr, return_aux=True)
    inference_llr = model(channel_llr)
    assert torch.equal(auxiliary["correction"], torch.zeros_like(auxiliary["correction"]))
    assert torch.equal(final_llr, auxiliary["bp_llr"])
    assert torch.equal(inference_llr, final_llr)


def test_fixed_validation_dataset_and_evaluation_are_repeatable() -> None:
    parity_check = small_graph()
    generator = small_generator()
    ebn0_values = [2.0, 3.0]

    first = create_fixed_validation_dataset(
        generator,
        parity_check,
        ebn0_values,
        frames_per_snr=12,
        seed=123,
    )
    second = create_fixed_validation_dataset(
        generator,
        parity_check,
        ebn0_values,
        frames_per_snr=12,
        seed=123,
    )
    for key in ("2", "3"):
        assert torch.equal(first[key]["channel_llr"], second[key]["channel_llr"])
        assert torch.equal(first[key]["target"], second[key]["target"])

    model = HybridBPGAT(
        parity_check,
        bp_iterations=3,
        positional_dim=0,
        hidden_dim=8,
        heads=2,
        head_dim=4,
    ).eval()
    metrics_one = evaluate(
        model,
        generator,
        parity_check,
        ebn0_values,
        batches=1,
        batch_size=5,
        device=torch.device("cpu"),
        fixed_validation=first,
    )
    metrics_two = evaluate(
        model,
        generator,
        parity_check,
        ebn0_values,
        batches=1,
        batch_size=5,
        device=torch.device("cpu"),
        fixed_validation=first,
    )
    assert metrics_one == metrics_two
