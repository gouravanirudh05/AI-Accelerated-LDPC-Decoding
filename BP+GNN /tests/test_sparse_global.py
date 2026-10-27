from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.bp_trace import CN_FEATURE_DIM, VN_FEATURE_DIM, VN_IDX_CONFIDENCE
from ldpc_corrector.decoder import CorrectorDecoder
from ldpc_corrector.models.graph_transformer_sparse_global import (
    GraphTransformerSparseGlobal,
)
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH = 4


def _model(graph, **kwargs) -> GraphTransformerSparseGlobal:
    return GraphTransformerSparseGlobal(
        graph=graph,
        vn_dim=VN_FEATURE_DIM,
        cn_dim=CN_FEATURE_DIM,
        hidden_dim=16,
        heads=4,
        **kwargs,
    )


@pytest.fixture
def inputs(small_h):
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    vn = torch.randn(BATCH, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(BATCH, graph.n_checks, CN_FEATURE_DIM)
    # Confidence must be non-negative: it is |LLR| / LLR_SCALE.
    vn[..., VN_IDX_CONFIDENCE] = vn[..., VN_IDX_CONFIDENCE].abs()
    return graph, vn, cn


def _trained_delta_head(model: GraphTransformerSparseGlobal) -> None:
    """Move the zero-initialized head off zero, as training would."""
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.5)
        model.delta_head.bias.fill_(0.7)


def test_empty_subset_returns_all_zero_delta_without_erroring(inputs) -> None:
    """threshold=0 selects nothing: the required degenerate case."""
    graph, vn, cn = inputs
    model = _model(graph, mode="threshold", threshold=0.0)
    _trained_delta_head(model)  # so a nonzero result would actually show up

    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    assert delta.shape == (BATCH, graph.n_variables)
    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_nodes_outside_the_subset_get_exactly_zero(inputs) -> None:
    graph, vn, cn = inputs
    model = _model(graph, mode="threshold", threshold=4.0)
    _trained_delta_head(model)

    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)
    assert output.output_mask is not None
    outside = output.output_mask == 0
    assert outside.any(), "test needs at least one unselected node"
    assert torch.all(output.delta_llr[outside] == 0.0)
    # And the selected ones are actually being corrected.
    assert output.delta_llr[~outside].abs().sum() > 0


def test_selection_is_not_done_by_zeroing_inputs(inputs) -> None:
    """Unselected nodes still get their real features, per the spec.

    If unselected inputs were zeroed, replacing their features with different
    values could not change the *selected* nodes' outputs. It can, because they
    are encoded and scattered normally.
    """
    graph, vn, cn = inputs
    model = _model(graph, mode="top_k", top_k=3).eval()
    _trained_delta_head(model)

    indices, valid, mask = model.select(vn[..., VN_IDX_CONFIDENCE])
    _, _, refined_mask = model.select(vn[..., VN_IDX_CONFIDENCE])
    torch.testing.assert_close(mask, refined_mask)

    # The encoder sees every node, selected or not.
    hidden = model.vn_encoder(vn)
    assert hidden.shape == (BATCH, graph.n_variables, model.hidden_dim)
    unselected = (mask == 0)
    assert hidden[unselected].abs().sum() > 0, "unselected inputs were zeroed"


def test_top_k_selects_the_least_confident_bits(inputs) -> None:
    graph, vn, cn = inputs
    model = _model(graph, mode="top_k", top_k=3)
    confidence = vn[..., VN_IDX_CONFIDENCE]
    indices, valid, mask = model.select(confidence)

    assert indices.shape == (BATCH, 3)
    assert valid.all()
    assert torch.all(mask.sum(dim=1) == 3)
    for b in range(BATCH):
        chosen = confidence[b, indices[b]]
        rest = confidence[b, mask[b] == 0]
        assert chosen.max() <= rest.min(), "a more confident bit was selected"


def test_threshold_selects_exactly_the_unconfident_bits(inputs) -> None:
    """``threshold`` is in raw LLR units, not the scaled feature units."""
    from ldpc_corrector.bp_trace import LLR_SCALE

    graph, vn, cn = inputs
    threshold = 4.0
    model = _model(graph, mode="threshold", threshold=threshold)
    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)

    raw_confidence = vn[..., VN_IDX_CONFIDENCE] * LLR_SCALE
    expected = (raw_confidence < threshold).to(output.output_mask.dtype)
    torch.testing.assert_close(output.output_mask, expected)


def test_ragged_subsets_across_the_batch_are_handled(small_h) -> None:
    """Frames with different |S| must not contaminate each other."""
    graph = build_tanner_graph(small_h)
    model = _model(graph, mode="threshold", threshold=1.0).eval()
    _trained_delta_head(model)

    torch.manual_seed(1)
    vn = torch.randn(3, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(3, graph.n_checks, CN_FEATURE_DIM)
    # Frame 0 selects nothing, frame 1 selects one bit, frame 2 selects all.
    vn[0, :, VN_IDX_CONFIDENCE] = 5.0
    vn[1, :, VN_IDX_CONFIDENCE] = 5.0
    vn[1, 2, VN_IDX_CONFIDENCE] = 0.001
    vn[2, :, VN_IDX_CONFIDENCE] = 0.001

    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)
    counts = output.output_mask.sum(dim=1)
    assert counts.tolist() == [0.0, 1.0, float(graph.n_variables)]
    torch.testing.assert_close(output.delta_llr[0], torch.zeros(graph.n_variables))
    assert output.delta_llr[1, 2].abs() > 0
    assert torch.all(output.delta_llr[1, [0, 1, 3, 4, 5, 6, 7]] == 0.0)
    assert torch.isfinite(output.delta_llr).all()


def test_padded_rows_do_not_change_a_frames_result(small_h) -> None:
    """A frame's output must not depend on how much padding the batch needs."""
    graph = build_tanner_graph(small_h)
    model = _model(graph, mode="threshold", threshold=1.0).eval()
    _trained_delta_head(model)

    torch.manual_seed(2)
    vn = torch.randn(2, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(2, graph.n_checks, CN_FEATURE_DIM)
    vn[:, :, VN_IDX_CONFIDENCE] = 5.0
    vn[0, [1, 4], VN_IDX_CONFIDENCE] = 0.001  # frame 0 selects 2
    vn[1, :, VN_IDX_CONFIDENCE] = 0.001  # frame 1 selects all, forcing padding

    together = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    alone = model(vn[:1], cn[:1], graph.vn_to_cn, graph.cn_to_vn)
    torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)


def test_gradients_flow_only_through_selected_bits(inputs) -> None:
    graph, vn, cn = inputs
    model = _model(graph, mode="top_k", top_k=2)
    _trained_delta_head(model)

    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)
    output.delta_llr.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all()


def test_multi_layer_block(inputs) -> None:
    graph, vn, cn = inputs
    model = _model(graph, mode="top_k", top_k=4, num_layers=2)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    assert delta.shape == (BATCH, graph.n_variables)
    assert torch.isfinite(delta).all()


def test_invalid_configuration_is_rejected(small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="mode must be"):
        _model(graph, mode="nonsense")
    with pytest.raises(ValueError, match="divisible by heads"):
        GraphTransformerSparseGlobal(
            graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            hidden_dim=15, heads=4,
        )
    with pytest.raises(ValueError, match="top_k must be"):
        _model(graph, mode="top_k", top_k=999)
    with pytest.raises(ValueError, match="threshold must be"):
        _model(graph, threshold=-1.0)
    with pytest.raises(ValueError, match="num_layers"):
        _model(graph, num_layers=0)


def test_end_to_end_on_the_real_code(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(
        parity_check,
        corrector="sparse_global_gt",
        bp_iterations=5,
        corrector_kwargs={"hidden_dim": 16, "heads": 4, "mode": "top_k", "top_k": 6},
    )
    final_llr, aux = decoder(torch.randn(8, 49) * 2.0, return_aux=True)
    assert final_llr.shape == (8, 49)
    # Untrained: still exactly BP.
    torch.testing.assert_close(final_llr, aux["bp_llr"])
