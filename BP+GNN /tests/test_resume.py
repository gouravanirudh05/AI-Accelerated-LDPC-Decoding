"""Resuming BP after the correction (``bp.resume_iterations``).

The invariant everything rests on: resuming is *continuing*. BP(j) then k
resumed iterations on the same channel is BP(j + k), so an untrained corrector
with resumed BP is exactly BP(j + k) and any difference is the correction.
"""

from __future__ import annotations

import pytest
import torch

from ldpc_corrector._compat import LegacyTannerGraph, SumProductBP
from ldpc_corrector.config import load_config
from ldpc_corrector.decoder import BPBaseline, CorrectorDecoder
from ldpc_corrector.evaluate import load_checkpoint_decoder

SPARSE = ["model.corrector=pyg_sparse_global_gt", "model.hidden_dim=16",
          "model.heads=2", "model.num_layers=1", "model.mode=top_k", "model.top_k=4"]


@pytest.fixture
def channel_llr():
    torch.manual_seed(0)
    return 1.5 + 2.0 * torch.randn(64, 49)


def test_resumed_bp_is_continued_bp(ldpc49_code, channel_llr) -> None:
    _, parity_check = ldpc49_code
    graph = LegacyTannerGraph.from_parity_check(parity_check)
    _, v2c, c2v = SumProductBP(graph, 5)(channel_llr, return_messages=True)
    resumed = SumProductBP(graph, 5)(channel_llr, initial_messages=(v2c, c2v))
    torch.testing.assert_close(resumed, SumProductBP(graph, 10)(channel_llr), rtol=0, atol=0)


def test_initial_messages_shape_is_checked(ldpc49_code, channel_llr) -> None:
    _, parity_check = ldpc49_code
    bp = SumProductBP(LegacyTannerGraph.from_parity_check(parity_check), 2)
    wrong = torch.zeros(64, 3)
    with pytest.raises(ValueError, match="initial_messages"):
        bp(channel_llr, initial_messages=(wrong, wrong))


def test_untrained_corrector_with_resume_is_exactly_bp10(ldpc49_code, channel_llr) -> None:
    """Zero-initialised correction + 5 resumed iterations == BP(10), bit for bit."""
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(parity_check, corrector="pyg_sparse_global_gt",
                               corrector_kwargs={"hidden_dim": 16, "heads": 2,
                                                 "mode": "top_k", "top_k": 4},
                               resume_iterations=5).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            decoder(channel_llr), BPBaseline(parity_check, 10).eval()(channel_llr),
            rtol=0, atol=0,
        )


def test_correction_enters_the_resumed_bp_as_a_prior(ldpc49_code, channel_llr) -> None:
    """A live correction must change the resumed output, through BP."""
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(parity_check, corrector="pyg_sparse_global_gt",
                               corrector_kwargs={"hidden_dim": 16, "heads": 2,
                                                 "mode": "top_k", "top_k": 4},
                               resume_iterations=5).eval()
    with torch.no_grad():
        decoder.corrector.delta_head.weight.normal_(0.0, 0.5)
        baseline = BPBaseline(parity_check, 10).eval()(channel_llr)
        corrected = decoder(channel_llr)
    changed = (corrected - baseline).abs() > 1e-4
    # Only 4 bits per frame are corrected, but BP spreads the change further.
    assert changed.sum(dim=1).float().mean() > 4


def test_training_backpropagates_through_resumed_bp(ldpc49_code, channel_llr) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(parity_check, corrector="pyg_sparse_global_gt",
                               corrector_kwargs={"hidden_dim": 16, "heads": 2,
                                                 "mode": "top_k", "top_k": 4},
                               resume_iterations=3)
    with torch.no_grad():
        decoder.corrector.delta_head.weight.normal_(0.0, 0.1)
    decoder(channel_llr).square().mean().backward()
    grad = decoder.corrector.delta_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_name_and_cost_include_the_resumed_iterations(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    plain = CorrectorDecoder(parity_check, corrector="gcn")
    resumed = CorrectorDecoder(parity_check, corrector="gcn", resume_iterations=5)
    assert resumed.name == plain.name + "+BP(5)"
    bp5 = BPBaseline(parity_check, 5).estimate_flops_per_frame()
    assert resumed.estimate_flops_per_frame() == pytest.approx(
        plain.estimate_flops_per_frame() + bp5
    )


def test_checkpoint_can_be_swept_with_resume_added(tmp_path, ldpc49_code, channel_llr) -> None:
    """A corrector trained without resume can be evaluated with it, unchanged weights."""
    from ldpc_corrector.train import build_model

    _, parity_check = ldpc49_code
    config = load_config(None, ["name=probe", *SPARSE])
    assert config.bp.resume_iterations == 0
    model = build_model(config, parity_check)
    path = tmp_path / "probe.pt"
    torch.save({"step": 1, "config": config.to_dict(), "model_state": model.state_dict()}, path)

    label, plain = load_checkpoint_decoder(path, parity_check, torch.device("cpu"))
    label_r, resumed = load_checkpoint_decoder(
        path, parity_check, torch.device("cpu"), resume_iterations=5
    )
    assert label_r != label and "+BP(5)" in label_r
    assert resumed.resume_iterations == 5 and plain.resume_iterations == 0


def test_resume_setting_survives_a_checkpoint(tmp_path, ldpc49_code) -> None:
    from ldpc_corrector.train import build_model

    _, parity_check = ldpc49_code
    config = load_config(None, ["name=probe", "bp.resume_iterations=4", *SPARSE])
    model = build_model(config, parity_check)
    path = tmp_path / "probe.pt"
    torch.save({"step": 1, "config": config.to_dict(), "model_state": model.state_dict()}, path)
    _, restored = load_checkpoint_decoder(path, parity_check, torch.device("cpu"))
    assert restored.resume_iterations == 4
