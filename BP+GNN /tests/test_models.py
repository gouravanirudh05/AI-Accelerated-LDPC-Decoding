"""Shape, gradient, and zero-init checks for every corrector architecture.

Each new model class is registered in ``CORRECTOR_CASES`` and inherits the
whole battery below before it is allowed near the training loop.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector.bp_trace import CN_FEATURE_DIM, VN_FEATURE_DIM
from ldpc_corrector.models import CORRECTOR_REGISTRY, build_corrector
from ldpc_corrector.models.base_corrector import combine
from ldpc_corrector.tanner_graph import build_tanner_graph

BATCH = 3
HIDDEN = 16

# name -> extra constructor kwargs beyond the common ones.
CORRECTOR_CASES = {
    "gcn": [{"num_layers": 1}, {"num_layers": 2}],
    "gat": [
        {"heads": 2, "head_dim": 8, "edge_exclusion": False},
        {"heads": 2, "head_dim": 8, "edge_exclusion": True},
        {"heads": 2, "head_dim": 8, "attention_variant": "v1"},
        {"heads": 2, "head_dim": 8, "attention_variant": "v1", "edge_exclusion": True},
    ],
    "local_gt": [
        {"heads": 2, "head_dim": 8, "hop_count": 1},
        {"heads": 2, "head_dim": 8, "hop_count": 2},
        {"heads": 2, "head_dim": 8, "hop_count": 1, "edge_exclusion": True},
        {"heads": 2, "head_dim": 8, "hop_count": 2, "edge_exclusion": True},
        {"heads": 2, "head_dim": 8, "hop_count": 2, "attention_variant": "v1"},
    ],
    "sparse_global_gt": [
        {"heads": 4, "mode": "threshold", "threshold": 10.0},
        {"heads": 4, "mode": "top_k", "top_k": 4},
    ],
    "resnet1d": [
        {"num_blocks": 1, "kernel_size": 3},
        {"num_blocks": 4, "kernel_size": 5, "dropout": 0.1},
    ],
    "resnet2d": [
        {"num_blocks": 1},
        {"num_blocks": 3, "dropout": 0.1},
    ],
}


def _instantiate(name: str, graph, extra: dict, **overrides):
    kwargs = {
        "graph": graph,
        "vn_dim": VN_FEATURE_DIM,
        "cn_dim": CN_FEATURE_DIM,
        "hidden_dim": HIDDEN,
        **extra,
        **overrides,
    }
    return build_corrector(name, **kwargs)


def _cases():
    for name, variants in CORRECTOR_CASES.items():
        for index, extra in enumerate(variants):
            yield pytest.param(name, extra, id=f"{name}-{index}")


@pytest.fixture
def dummy_inputs(small_h):
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    vn = torch.randn(BATCH, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(BATCH, graph.n_checks, CN_FEATURE_DIM)
    return graph, vn, cn


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_delta_llr_shape(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    assert delta.shape == (BATCH, graph.n_variables)
    assert torch.isfinite(delta).all()


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_untrained_corrector_is_exactly_bp(name, extra, dummy_inputs) -> None:
    """Zero-initialized residual head: an untrained corrector cannot hurt BP."""
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    torch.testing.assert_close(delta, torch.zeros_like(delta))

    bp_llr = torch.randn(BATCH, graph.n_variables)
    torch.testing.assert_close(combine(bp_llr, delta), bp_llr)


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_gradients_reach_every_parameter(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra)
    # The delta head starts at zero, so perturb it to get a live gradient path.
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.1)

    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    delta.square().mean().backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{parameter_name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{parameter_name} grad not finite"


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_learned_gate_shape_and_range(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra, use_gate=True)
    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)

    assert output.delta_llr.shape == (BATCH, graph.n_variables)
    assert output.gate_logits is not None
    # Per-bit, not one global scalar.
    assert output.gate_logits.shape == (BATCH, graph.n_variables)
    gate = torch.sigmoid(output.gate_logits)
    assert torch.all((gate > 0.0) & (gate < 1.0))


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_fixed_gate_has_no_gate_head(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra, use_gate=False)
    output = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn, return_gate=True)
    assert output.gate_logits is None
    assert not any(name.startswith("gate_head") for name, _ in model.named_parameters())


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_correction_is_bounded_by_correction_scale(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra, correction_scale=1.5)
    with torch.no_grad():  # drive the head hard to hit the tanh saturation
        model.delta_head.weight.normal_(0.0, 50.0)
        model.delta_head.bias.normal_(0.0, 50.0)
    delta = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    assert delta.abs().max().item() <= 1.5 + 1e-6


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_batch_elements_are_independent(name, extra, dummy_inputs) -> None:
    """A fixed graph with batched signals must not leak across batch items."""
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra)
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.5)
    model.eval()

    full = model(vn, cn, graph.vn_to_cn, graph.cn_to_vn)
    single = model(vn[1:2], cn[1:2], graph.vn_to_cn, graph.cn_to_vn)
    torch.testing.assert_close(full[1:2], single, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name,extra", list(_cases()))
def test_rejects_mismatched_feature_shapes(name, extra, dummy_inputs) -> None:
    graph, vn, cn = dummy_inputs
    model = _instantiate(name, graph, extra)
    with pytest.raises(ValueError, match="variable channels"):
        model(vn[..., :-1], cn, graph.vn_to_cn, graph.cn_to_vn)
    with pytest.raises(ValueError, match="check channels"):
        model(vn, cn[..., :-1], graph.vn_to_cn, graph.cn_to_vn)
    with pytest.raises(ValueError, match="variable nodes"):
        model(vn[:, :-1], cn, graph.vn_to_cn, graph.cn_to_vn)
    with pytest.raises(ValueError, match="\\[batch, nodes, channels\\]"):
        model(vn[0], cn, graph.vn_to_cn, graph.cn_to_vn)


def test_registry_covers_every_tested_case() -> None:
    assert set(CORRECTOR_CASES) <= set(CORRECTOR_REGISTRY)


def test_build_corrector_rejects_unknown_name(small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(KeyError, match="Unknown corrector"):
        build_corrector("nope", graph=graph, vn_dim=1, cn_dim=1)


def test_gcn_rejects_bad_layer_count(small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="1 or 2 layers"):
        build_corrector(
            "gcn", graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM,
            num_layers=3,
        )
