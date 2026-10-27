"""CNN-specific checks for the ResNet correctors.

The generic battery in ``test_models.py`` already covers shape, zero-init,
gradients, gate, bound and batch independence.  These pin down what is
particular to the convolutional encoders: what each one can and cannot see,
and that the receptive field really covers the code.
"""

from __future__ import annotations

import pytest
import torch

from ldpc_corrector.bp_trace import CN_FEATURE_DIM, VN_FEATURE_DIM
from ldpc_corrector.decoder import BPBaseline, CorrectorDecoder
from ldpc_corrector.models import build_corrector
from ldpc_corrector.tanner_graph import build_tanner_graph


def _build(name, graph, **kwargs):
    return build_corrector(
        name, graph=graph, vn_dim=VN_FEATURE_DIM, cn_dim=CN_FEATURE_DIM, **kwargs
    )


def _live(model):
    """Perturb the zero-initialized head so the output depends on the input."""
    with torch.no_grad():
        model.delta_head.weight.normal_(0.0, 0.5)
    return model.eval()


@pytest.mark.parametrize("name", ["resnet1d", "resnet2d"])
def test_default_receptive_field_covers_ldpc49(name, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    model = _build(name, graph)
    longest_axis = graph.n_variables if name == "resnet1d" else max(
        graph.n_variables, graph.n_checks
    )
    assert model.receptive_field >= longest_axis


def test_dilations_are_clamped_on_short_codes(small_h) -> None:
    """Circular padding cannot wrap more than once; tiny codes must still build."""
    graph = build_tanner_graph(small_h)
    model = _build("resnet1d", graph, kernel_size=5, dilations=(1, 2, 4, 8, 16))
    half_kernel = 2
    assert all(d * half_kernel <= graph.n_variables for d in model.block_dilations)


def test_resnet1d_ignores_check_features(small_h) -> None:
    """resnet1d has no access to H: check-node features cannot reach it."""
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    model = _live(_build("resnet1d", graph, hidden_dim=8, num_blocks=2))
    vn = torch.randn(2, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(2, graph.n_checks, CN_FEATURE_DIM)

    base = model(vn, cn)
    torch.testing.assert_close(model(vn, torch.randn_like(cn)), base)


def test_resnet2d_uses_check_features(small_h) -> None:
    graph = build_tanner_graph(small_h)
    torch.manual_seed(0)
    model = _live(_build("resnet2d", graph, hidden_dim=8, num_blocks=2))
    vn = torch.randn(2, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(2, graph.n_checks, CN_FEATURE_DIM)

    assert not torch.allclose(model(vn, cn), model(vn, torch.randn_like(cn)))


def test_resnet2d_image_is_zero_off_the_edges(small_h) -> None:
    graph = build_tanner_graph(small_h)
    model = _build("resnet2d", graph, hidden_dim=8)
    vn = torch.randn(2, graph.n_variables, VN_FEATURE_DIM)
    cn = torch.randn(2, graph.n_checks, CN_FEATURE_DIM)
    image = model.build_image(vn, cn)

    h = torch.as_tensor(small_h, dtype=torch.float32)
    assert image.shape == (2, 1 + VN_FEATURE_DIM + CN_FEATURE_DIM, *h.shape)
    torch.testing.assert_close(image[:, 0], h.expand(2, -1, -1))
    assert torch.all(image[:, 1:][..., h == 0] == 0)
    # On an edge (j, i), the bit channels carry bit i's features.
    j, i = 0, 4
    assert small_h[j, i] == 1
    torch.testing.assert_close(image[:, 1 : 1 + VN_FEATURE_DIM, j, i], vn[:, i])


@pytest.mark.parametrize("name", ["resnet1d", "resnet2d"])
def test_edge_residual_mode_is_rejected(name, small_h) -> None:
    graph = build_tanner_graph(small_h)
    with pytest.raises(ValueError, match="edge_dim"):
        _build(name, graph, residual_mode="edge")


@pytest.mark.parametrize("name", ["resnet1d", "resnet2d"])
def test_untrained_cnn_decoder_is_exactly_bp5(name, ldpc49_code) -> None:
    """End to end on the real code: zero-init means BP(5) bit for bit."""
    generator, parity_check = ldpc49_code
    torch.manual_seed(0)
    channel_llr = 2.0 + 1.5 * torch.randn(64, parity_check.shape[1])

    decoder = CorrectorDecoder(parity_check, corrector=name).eval()
    bp = BPBaseline(parity_check, bp_iterations=5).eval()
    with torch.no_grad():
        torch.testing.assert_close(decoder(channel_llr), bp(channel_llr))


@pytest.mark.parametrize("name", ["resnet1d", "resnet2d"])
def test_flop_estimate_is_positive_and_scales_with_depth(name, ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    graph = build_tanner_graph(parity_check)
    shallow = _build(name, graph, num_blocks=1).estimate_flops_per_frame()
    deep = _build(name, graph, num_blocks=4).estimate_flops_per_frame()
    assert 0 < shallow < deep
