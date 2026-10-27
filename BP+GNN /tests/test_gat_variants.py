"""GATv1 (static attention) vs GATv2 (dynamic attention).

The mathematical distinction is about *ranking*, not about the exact weights.

GATv1 scores with ``LeakyReLU(a_t^T Wh_t + a_s^T Wh_s + a_e^T We)``.  Within one
target's incident edges the target term is an additive constant, and LeakyReLU
is strictly monotonic, so the *ordering* of the neighbours is fixed by the
source and edge terms alone -- identical for every query.  That is static
attention.

GATv2 scores with ``a^T LeakyReLU(Wh_t + Wh_s + We)``.  The nonlinearity sits
inside, before the attention vector, so the target's contribution does not
factor out and the ordering can differ per query.  That is dynamic attention.

Note that the *values* alone do not separate them: if every pre-activation is
positive, LeakyReLU is the identity and both variants collapse to static
attention.  These tests therefore use random weights, so the nonlinearity is
genuinely active, and compare orderings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ldpc_corrector.bp_trace import CN_FEATURE_DIM, EDGE_FEATURE_DIM, VN_FEATURE_DIM
from ldpc_corrector.models.gat_corrector import ConfigurableEdgeGATPass, GATCorrector
from ldpc_corrector.models.graph_ops import scatter_sum
from ldpc_corrector.models.graph_transformer_local import GraphTransformerLocal
from ldpc_corrector.tanner_graph import build_tanner_graph

HIDDEN, HEADS, HEAD_DIM = 8, 1, 4


def _pass_inputs(graph):
    """Target-grouped indices and slices for the C->V direction."""
    target_index = graph.edge_vn
    slices, start = [], 0
    for variable in range(graph.n_variables):
        count = int((target_index == variable).sum())
        slices.append((start, start + count))
        start += count
    return target_index, slices


def _make_layer(variant: str, seed: int = 0) -> ConfigurableEdgeGATPass:
    """A randomly initialized layer with the additive edge bias removed.

    ``edge_bias`` is added *after* the nonlinearity, so zeroing it isolates the
    attention scoring itself -- otherwise a per-edge constant added post-hoc
    could reorder v1's neighbours and mask the property under test.
    """
    torch.manual_seed(seed)
    layer = ConfigurableEdgeGATPass(
        HIDDEN, EDGE_FEATURE_DIM, HEADS, HEAD_DIM, attention_variant=variant
    )
    layer.eval()
    with torch.no_grad():
        layer.edge_bias.weight.zero_()
    return layer


def _neighbour_orderings(attention, graph):
    """Ordering of each target's incident edges by attention weight."""
    orderings = []
    for variable in range(graph.n_variables):
        positions = (graph.edge_vn == variable).nonzero().flatten()
        if positions.numel() < 2:
            continue  # a single neighbour has no ordering to change
        weights = attention[0, positions, 0]
        orderings.append(tuple(torch.argsort(weights).tolist()))
    return orderings


def _attention(layer, graph, target_hidden, source_hidden, edge_features):
    target_index, slices = _pass_inputs(graph)
    _, attention = layer(
        target_hidden=target_hidden,
        source_hidden=source_hidden,
        edge_features=edge_features,
        source_index=graph.edge_cn,
        target_index=target_index,
        target_slices=slices,
        n_targets=graph.n_variables,
        return_attention=True,
    )
    return attention


def test_v1_neighbour_ordering_is_query_independent(small_h) -> None:
    """GATv1 is static: changing the query cannot reorder its neighbours."""
    graph = build_tanner_graph(small_h)
    layer = _make_layer("v1")

    torch.manual_seed(3)
    source_hidden = torch.randn(1, graph.n_checks, HIDDEN)
    edge_features = torch.randn(1, graph.n_edges, EDGE_FEATURE_DIM)

    reference = None
    # Sweep the query side hard: none of it may change the ordering.
    for shift in (-5.0, -1.0, 0.0, 1.0, 5.0):
        target_hidden = torch.randn(1, graph.n_variables, HIDDEN) + shift
        attention = _attention(
            layer, graph, target_hidden, source_hidden, edge_features
        )
        ordering = _neighbour_orderings(attention, graph)
        assert ordering, "test needs at least one multi-neighbour target"
        if reference is None:
            reference = ordering
        else:
            assert ordering == reference, (
                "GATv1 attention reordered neighbours when the query changed"
            )


def test_v2_neighbour_ordering_depends_on_the_query(small_h) -> None:
    """GATv2 is dynamic: the query can reorder its neighbours."""
    graph = build_tanner_graph(small_h)
    layer = _make_layer("v2")

    torch.manual_seed(3)
    source_hidden = torch.randn(1, graph.n_checks, HIDDEN)
    edge_features = torch.randn(1, graph.n_edges, EDGE_FEATURE_DIM)

    orderings = set()
    for shift in (-5.0, -1.0, 0.0, 1.0, 5.0):
        target_hidden = torch.randn(1, graph.n_variables, HIDDEN) + shift
        attention = _attention(
            layer, graph, target_hidden, source_hidden, edge_features
        )
        orderings.add(tuple(_neighbour_orderings(attention, graph)))

    assert len(orderings) > 1, (
        "GATv2 attention should be able to reorder neighbours per query"
    )


def test_v2_attention_values_respond_to_the_query(small_h) -> None:
    graph = build_tanner_graph(small_h)
    layer = _make_layer("v2")
    torch.manual_seed(5)
    source_hidden = torch.randn(1, graph.n_checks, HIDDEN)
    edge_features = torch.randn(1, graph.n_edges, EDGE_FEATURE_DIM)
    target_a = torch.randn(1, graph.n_variables, HIDDEN)
    target_b = target_a + 4.0

    first = _attention(layer, graph, target_a, source_hidden, edge_features)
    second = _attention(layer, graph, target_b, source_hidden, edge_features)
    assert not torch.allclose(first, second, atol=1e-4)


def test_both_variants_normalize_attention_per_target(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(1)
    target_hidden = torch.randn(2, graph.n_variables, HIDDEN)
    source_hidden = torch.randn(2, graph.n_checks, HIDDEN)
    edge_features = torch.randn(2, graph.n_edges, EDGE_FEATURE_DIM)

    for variant in ("v1", "v2"):
        layer = ConfigurableEdgeGATPass(
            HIDDEN, EDGE_FEATURE_DIM, HEADS, HEAD_DIM, attention_variant=variant
        ).eval()
        attention = _attention(
            layer, graph, target_hidden, source_hidden, edge_features
        )
        sums = scatter_sum(attention, graph.edge_vn, graph.n_variables)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5
        )
        assert torch.all(attention >= 0)


def test_default_path_delegates_to_the_legacy_implementation(small_h) -> None:
    """v2 without exclusion must be the Phase 0-2 code, bit for bit."""
    from ldpc_corrector._compat import DirectedEdgeGATv2

    graph = build_tanner_graph(small_h)
    torch.manual_seed(4)
    configurable = ConfigurableEdgeGATPass(
        HIDDEN, EDGE_FEATURE_DIM, HEADS, HEAD_DIM
    ).eval()
    assert configurable.is_legacy_path

    legacy = DirectedEdgeGATv2(HIDDEN, EDGE_FEATURE_DIM, HEADS, HEAD_DIM).eval()
    legacy.load_state_dict(configurable.state_dict())

    target_hidden = torch.randn(2, graph.n_variables, HIDDEN)
    source_hidden = torch.randn(2, graph.n_checks, HIDDEN)
    edge_features = torch.randn(2, graph.n_edges, EDGE_FEATURE_DIM)
    torch.testing.assert_close(
        _attention(configurable, graph, target_hidden, source_hidden, edge_features),
        _attention(legacy, graph, target_hidden, source_hidden, edge_features),
    )


def test_v1_and_v2_differ_but_share_their_common_parameters(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(9)
    v2 = GATCorrector(
        graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM, hidden_dim=16,
        heads=2, head_dim=8, attention_variant="v2",
    ).eval()
    torch.manual_seed(9)
    v1 = GATCorrector(
        graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM, hidden_dim=16,
        heads=2, head_dim=8, attention_variant="v1",
    ).eval()

    v1_params = dict(v1.named_parameters())
    v2_params = dict(v2.named_parameters())
    # v1 adds exactly its two extra attention vectors per directed pass.
    extra = set(v1_params) - set(v2_params)
    assert extra == {
        "block.v_to_c.attn_vector_source", "block.v_to_c.attn_vector_edge",
        "block.c_to_v.attn_vector_source", "block.c_to_v.attn_vector_edge",
    }
    # Everything they share is initialized identically, so the comparison is
    # between attention rules and not between two random inits.
    for name, parameter in v2_params.items():
        torch.testing.assert_close(parameter, v1_params[name])

    torch.manual_seed(0)
    vn = torch.randn(3, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(3, graph.n_checks, CN_FEATURE_DIM)
    edges = torch.randn(3, graph.n_edges, EDGE_FEATURE_DIM)
    assert not torch.allclose(
        v1.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
        v2.encode(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
    )


@pytest.mark.parametrize("variant", ["v1", "v2"])
@pytest.mark.parametrize("exclusion", [False, True])
def test_all_four_flag_combinations_are_trainable(small_h, variant, exclusion) -> None:
    """The two flags are orthogonal, so every combination must work."""
    graph = build_tanner_graph(small_h)
    model = GATCorrector(
        graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM, hidden_dim=16,
        heads=2, head_dim=8, attention_variant=variant, edge_exclusion=exclusion,
    )
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.1)

    torch.manual_seed(0)
    vn = torch.randn(2, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(2, graph.n_checks, CN_FEATURE_DIM)
    edges = torch.randn(2, graph.n_edges, EDGE_FEATURE_DIM)

    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges)
    assert delta.shape == (2, graph.n_variables)
    assert torch.isfinite(delta).all()
    delta.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all()


def test_local_transformer_supports_both_variants(small_h) -> None:
    graph = build_tanner_graph(small_h)
    for variant in ("v1", "v2"):
        model = GraphTransformerLocal(
            graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            hidden_dim=16, heads=2, head_dim=8, hop_count=2,
            attention_variant=variant,
        )
        assert model.attention_variant == variant
        # The 2-hop pass must use the same variant as the bipartite passes.
        assert model.v_to_v.attention_variant == variant
        assert model.v_to_c.attention_variant == variant


def test_invalid_variant_is_rejected(small_h) -> None:
    graph = build_tanner_graph(small_h)
    for factory in (GATCorrector, GraphTransformerLocal):
        with pytest.raises(ValueError, match="attention_variant must be"):
            factory(
                graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
                hidden_dim=16, heads=2, head_dim=8, attention_variant="v3",
            )
    with pytest.raises(ValueError, match="attention_variant must be"):
        ConfigurableEdgeGATPass(
            HIDDEN, EDGE_FEATURE_DIM, HEADS, HEAD_DIM, attention_variant="v9"
        )


def test_local_gt_hop1_is_the_same_model_as_gat(ldpc49_code) -> None:
    """``local_gt`` with ``hop_count=1`` *is* ``gat``.

    The 2-hop pass is the only thing that distinguishes the local transformer
    from the bipartite GAT block, so ``hop_count=1`` collapses onto it exactly.
    Pinned here because it is easy to read the two as separate architectures
    and then attribute a seed-variance gap between them to design.
    """
    from ldpc_corrector.models import build_corrector

    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    shared = dict(
        graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
        hidden_dim=32, heads=2, head_dim=16,
    )
    gat = build_corrector("gat", **shared).eval()
    local = build_corrector("local_gt", hop_count=1, **shared).eval()

    assert gat.num_parameters() == local.num_parameters()

    # Same parameter set, modulo the encoder attribute names.
    renamed = {}
    for key, value in gat.state_dict().items():
        key = key.replace("block.", "")
        key = key.replace("variable_encoder", "vn_encoder")
        key = key.replace("check_encoder", "cn_encoder")
        renamed[key] = value
    incompatible = local.load_state_dict(renamed, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    with torch.no_grad():  # lift the shared head off zero
        gat.delta_head.weight.normal_(0.0, 0.3)
        gat.delta_head.bias.fill_(0.1)
        local.delta_head.weight.copy_(gat.delta_head.weight)
        local.delta_head.bias.copy_(gat.delta_head.bias)

    torch.manual_seed(0)
    vn = torch.randn(4, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(4, graph.n_checks, CN_FEATURE_DIM)
    edges = torch.randn(4, graph.n_edges, EDGE_FEATURE_DIM)
    torch.testing.assert_close(
        gat(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
        local(vn, cn, graph.vn_to_cn, graph.cn_to_vn, edge_features=edges),
    )


def test_hop2_is_genuinely_different_from_gat(ldpc49_code) -> None:
    """Only hop_count=2 makes the local transformer a distinct architecture."""
    from ldpc_corrector.models import build_corrector

    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    shared = dict(
        graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
        hidden_dim=32, heads=2, head_dim=16,
    )
    hop1 = build_corrector("local_gt", hop_count=1, **shared)
    hop2 = build_corrector("local_gt", hop_count=2, **shared)
    assert hop2.num_parameters() > hop1.num_parameters()
    assert hop2.v_to_v is not None and hop1.v_to_v is None
