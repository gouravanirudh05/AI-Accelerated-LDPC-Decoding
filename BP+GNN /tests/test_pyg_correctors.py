"""Correctors built on PyTorch Geometric.

Covers the batching adapter, the bipartite wiring, and -- using PyG's own
``GATConv`` / ``GATv2Conv`` -- that the static-vs-dynamic attention distinction
holds with the library's implementations rather than only with mine.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.nn import GATConv, GATv2Conv  # noqa: E402
from torch_geometric.utils import scatter  # noqa: E402

from ldpc_corrector.bp_trace import (  # noqa: E402
    CN_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    VN_FEATURE_DIM,
)
from ldpc_corrector.decoder import CorrectorDecoder  # noqa: E402
from ldpc_corrector.models import build_corrector  # noqa: E402
from ldpc_corrector.models.pyg_correctors import ExtrinsicGATConv  # noqa: E402
from ldpc_corrector.pyg_graph import (  # noqa: E402
    BipartiteBatcher,
    flatten_edges,
    flatten_nodes,
    unflatten_nodes,
)
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH, HIDDEN, HEADS = 3, 16, 2

PYG_CASES = {
    "pyg_gcn": [{"num_layers": 1}, {"num_layers": 2}],
    "pyg_gat": [
        {"heads": 2, "attention_variant": "v2"},
        {"heads": 2, "attention_variant": "v1"},
        {"heads": 2, "edge_exclusion": True},
    ],
    "pyg_local_gt": [{"heads": 2, "hop_count": 1}, {"heads": 2, "hop_count": 2}],
}


def _cases():
    for name, variants in PYG_CASES.items():
        for index, extra in enumerate(variants):
            yield pytest.param(name, extra, id=f"{name}-{index}")


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


def _model(name, graph, extra, **overrides):
    return build_corrector(
        name,
        graph=graph,
        vn_dim=VN_FEATURE_DIM,
        cn_dim=CN_FEATURE_DIM,
        hidden_dim=HIDDEN,
        **extra,
        **overrides,
    )


# ------------------------------------------------------------ batching


def test_replicated_edge_index_offsets_each_batch_element(small_h) -> None:
    graph = build_tanner_graph(small_h)
    batcher = BipartiteBatcher(graph)
    index = batcher.edge_index("vn_to_cn", BATCH, torch.device("cpu"))

    assert index.shape == (2, BATCH * graph.n_edges)
    assert batcher.size("vn_to_cn", BATCH) == (
        BATCH * graph.n_variables,
        BATCH * graph.n_checks,
    )
    # Element b must occupy its own disjoint block of node ids.
    for b in range(BATCH):
        block = slice(b * graph.n_edges, (b + 1) * graph.n_edges)
        torch.testing.assert_close(
            index[0, block], graph.edge_vn + b * graph.n_variables
        )
        torch.testing.assert_close(
            index[1, block], graph.edge_cn + b * graph.n_checks
        )


def test_replicated_index_is_cached(small_h) -> None:
    batcher = BipartiteBatcher(build_tanner_graph(small_h))
    first = batcher.edge_index("cn_to_vn", 4, torch.device("cpu"))
    assert batcher.edge_index("cn_to_vn", 4, torch.device("cpu")) is first
    # A different batch size is a different index.
    assert batcher.edge_index("cn_to_vn", 5, torch.device("cpu")) is not first


def test_flatten_round_trips(small_h) -> None:
    graph = build_tanner_graph(small_h)
    nodes = torch.randn(BATCH, graph.n_variables, 5)
    assert torch.equal(unflatten_nodes(flatten_nodes(nodes), BATCH), nodes)
    edges = torch.randn(BATCH, graph.n_edges, EDGE_FEATURE_DIM)
    assert flatten_edges(edges).shape == (BATCH * graph.n_edges, EDGE_FEATURE_DIM)


def test_batch_elements_stay_independent_under_pyg(inputs) -> None:
    """The disjoint union must not leak between frames."""
    graph, vn, cn, edges = inputs
    model = _model("pyg_gat", graph, {"heads": HEADS}).eval()
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.5)

    full = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    single = model(
        vn[1:2], cn[1:2], graph.vn_to_cn, graph.cn_to_vn, edge_features=edges[1:2]
    )
    torch.testing.assert_close(full[1:2], single, atol=1e-5, rtol=1e-5)


# ------------------------------------------------- shared model battery


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_delta_shape_and_zero_init(name, extra, inputs) -> None:
    graph, vn, cn, edges = inputs
    model = _model(name, graph, extra)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    assert delta.shape == (BATCH, graph.n_variables)
    assert torch.isfinite(delta).all()
    # Untrained: exactly BP.
    torch.testing.assert_close(delta, torch.zeros_like(delta))


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_gradients_reach_every_parameter(name, extra, inputs) -> None:
    graph, vn, cn, edges = inputs
    model = _model(name, graph, extra)
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.1)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    delta.square().mean().backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{parameter_name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), parameter_name


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_learned_gate_is_per_bit(name, extra, inputs) -> None:
    graph, vn, cn, edges = inputs
    model = _model(name, graph, extra, use_gate=True)
    output = model(
        vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges, return_gate=True
    )
    assert output.gate_logits.shape == (BATCH, graph.n_variables)
    gate = torch.sigmoid(output.gate_logits)
    assert torch.all((gate > 0) & (gate < 1))


def test_hidden_dim_must_divide_by_heads(small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="divisible by heads"):
        build_corrector(
            "pyg_gat", graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            hidden_dim=15, heads=2,
        )


def test_pyg_gcn_rejects_edge_residual(small_h) -> None:
    """SAGEConv carries no edge features to condition an edge residual on."""
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="edge_dim"):
        build_corrector(
            "pyg_gcn", graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            hidden_dim=HIDDEN, residual_mode="edge",
        )


# ------------------------------------- GATv1 vs GATv2, PyG's own layers


def _pyg_attention(layer, graph, x_src, x_dst, edge_attr):
    """Attention weights from a PyG GAT layer on the C->V direction."""
    edge_index = torch.stack((graph.edge_cn, graph.edge_vn), dim=0)
    _, (returned_index, alpha) = layer(
        (x_src, x_dst), edge_index, edge_attr, return_attention_weights=True
    )
    return returned_index, alpha


def _orderings(index, alpha, n_targets):
    """Ordering of each target's incident edges by attention weight."""
    target = index[1]
    orderings = []
    for node in range(n_targets):
        positions = (target == node).nonzero().flatten()
        if positions.numel() < 2:
            continue
        orderings.append(tuple(torch.argsort(alpha[positions, 0]).tolist()))
    return orderings


@pytest.mark.parametrize(
    "layer_class,variant,expect_static",
    [(GATConv, "v1", True), (GATv2Conv, "v2", False)],
)
def test_library_gat_static_vs_dynamic(
    small_h, layer_class, variant, expect_static
) -> None:
    """PyG's GATConv is static, GATv2Conv dynamic -- checked on their own code.

    Within one target's neighbourhood GATv1's query term is an additive
    constant and LeakyReLU is monotonic, so the neighbour ordering cannot
    change with the query. GATv2 puts the nonlinearity inside, so it can.
    """
    graph = build_tanner_graph(small_h)
    torch.manual_seed(3)
    layer = layer_class(
        (HIDDEN, HIDDEN), HIDDEN, heads=1, edge_dim=EDGE_FEATURE_DIM,
        add_self_loops=False,
    ).eval()

    x_src = torch.randn(graph.n_checks, HIDDEN)
    edge_attr = torch.randn(graph.n_edges, EDGE_FEATURE_DIM)

    observed = set()
    for shift in (-5.0, -1.0, 0.0, 1.0, 5.0):
        x_dst = torch.randn(graph.n_variables, HIDDEN) + shift
        index, alpha = _pyg_attention(layer, graph, x_src, x_dst, edge_attr)
        observed.add(tuple(_orderings(index, alpha, graph.n_variables)))

    if expect_static:
        assert len(observed) == 1, f"{variant} should not reorder neighbours"
    else:
        assert len(observed) > 1, f"{variant} should be able to reorder neighbours"


def test_v1_and_v2_correctors_differ(inputs) -> None:
    graph, vn, cn, edges = inputs
    torch.manual_seed(5)
    v1 = _model("pyg_gat", graph, {"heads": HEADS, "attention_variant": "v1"}).eval()
    torch.manual_seed(5)
    v2 = _model("pyg_gat", graph, {"heads": HEADS, "attention_variant": "v2"}).eval()
    assert not torch.allclose(
        v1.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
        v2.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
    )


# ------------------------------------------------ extrinsic aggregation


def test_extrinsic_conv_normalizes_and_stays_finite(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(7)
    layer = ExtrinsicGATConv(
        (HIDDEN, HIDDEN), HIDDEN, heads=1, edge_dim=EDGE_FEATURE_DIM
    ).eval()

    x_src = torch.randn(graph.n_checks, HIDDEN)
    x_dst = torch.randn(graph.n_variables, HIDDEN)
    edge_attr = torch.randn(graph.n_edges, EDGE_FEATURE_DIM)
    edge_index = torch.stack((graph.edge_cn, graph.edge_vn), dim=0)

    out = layer((x_src, x_dst), edge_index, edge_attr)
    assert out.shape == (graph.n_variables, HIDDEN)
    assert torch.isfinite(out).all()


def test_extrinsic_conv_handles_degree_one_targets() -> None:
    """One incident edge means no extrinsic information: must not divide by 0."""
    single = np.array([[1, 1], [1, 0]], dtype=np.int64)  # variable 1 has degree 1
    graph = build_tanner_graph(single)
    layer = ExtrinsicGATConv((4, 4), 4, heads=1, edge_dim=EDGE_FEATURE_DIM).eval()

    edge_index = torch.stack((graph.edge_cn, graph.edge_vn), dim=0)
    out = layer(
        (torch.randn(graph.n_checks, 4), torch.randn(graph.n_variables, 4)),
        edge_index,
        torch.randn(graph.n_edges, EDGE_FEATURE_DIM),
    )
    assert torch.isfinite(out).all()


# ---------------------------------------------------------- end to end


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_end_to_end_untrained_equals_bp(name, extra, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(
        parity_check,
        corrector=name,
        bp_iterations=5,
        corrector_kwargs={"hidden_dim": HIDDEN, **extra},
    )
    final_llr, aux = decoder(torch.randn(4, 49) * 2.0, return_aux=True)
    torch.testing.assert_close(final_llr, aux["bp_llr"])
    assert decoder.estimate_flops_per_frame() > 0


def test_two_hop_pass_adds_capacity(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    hop1 = _model("pyg_local_gt", graph, {"heads": HEADS, "hop_count": 1})
    hop2 = _model("pyg_local_gt", graph, {"heads": HEADS, "hop_count": 2})
    assert hop2.num_parameters() > hop1.num_parameters()
    assert hop2.to_vv is not None and hop1.to_vv is None
    assert hop2.estimate_flops_per_frame() > hop1.estimate_flops_per_frame()
