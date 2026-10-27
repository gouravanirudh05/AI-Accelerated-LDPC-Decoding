"""End-to-end pipeline tests: BP(k) -> features -> corrector -> combine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldpc_corrector._compat import sample_channel_batch
from ldpc_corrector.decoder import (
    BPBaseline,
    CorrectorDecoder,
    bp_flops_per_frame,
    decode_bits,
)
from ldpc_corrector.losses import LossConfig, build_loss

ALL_CORRECTORS = {
    "gcn": {"hidden_dim": 16, "num_layers": 2},
    "gat": {"hidden_dim": 16, "heads": 2, "head_dim": 8},
    "local_gt": {"hidden_dim": 16, "heads": 2, "head_dim": 8, "hop_count": 2},
    "sparse_global_gt": {"hidden_dim": 16, "heads": 4, "mode": "top_k", "top_k": 6},
}


@pytest.fixture(params=sorted(ALL_CORRECTORS))
def decoder(request, ldpc49_code):
    _, parity_check = ldpc49_code
    return CorrectorDecoder(
        parity_check,
        corrector=request.param,
        bp_iterations=5,
        corrector_kwargs=ALL_CORRECTORS[request.param],
    )


def test_untrained_decoder_reproduces_bp_exactly(decoder, ldpc49_code) -> None:
    """The zero-init residual means the corrector cannot damage BP(5)."""
    _, parity_check = ldpc49_code
    bp_only = BPBaseline(parity_check, 5)
    torch.manual_seed(0)
    channel_llr = torch.randn(16, 49) * 2.0

    final_llr = decoder(channel_llr)
    torch.testing.assert_close(final_llr, bp_only(channel_llr))


def test_bp_stage_is_frozen_by_default(decoder) -> None:
    assert decoder.freeze_bp
    assert all(not p.requires_grad for p in decoder.bp.parameters())

    channel_llr = (torch.randn(4, 49) * 2.0).requires_grad_(True)
    bp_llr, trace = decoder.run_bp(channel_llr)
    assert not bp_llr.requires_grad
    assert not trace["llr_trace"].requires_grad


def test_gradient_reaches_the_corrector_but_not_bp(decoder) -> None:
    with torch.no_grad():  # lift the residual head off its zero init
        decoder.corrector.delta_head.weight.normal_(0.0, 0.1)

    channel_llr = torch.randn(8, 49) * 2.0
    target = torch.randint(0, 2, (8, 49))
    loss_fn = build_loss(LossConfig(bce=True), decoder.graph)

    final_llr, aux = decoder(channel_llr, return_aux=True)
    loss, _ = loss_fn(final_llr, target, aux["bp_llr"], aux["stage_llrs"])
    loss.backward()

    corrector_grads = [
        p.grad for p in decoder.corrector.parameters() if p.grad is not None
    ]
    assert corrector_grads, "no gradient reached the corrector"
    assert any(g.abs().sum() > 0 for g in corrector_grads)


def test_aux_payload_is_complete(decoder) -> None:
    final_llr, aux = decoder(torch.randn(5, 49) * 2.0, return_aux=True)
    for key in (
        "bp_llr",
        "delta_llr",
        "gate",
        "stage_llrs",
        "confidence",
        "sign_flip_count",
        "check_parity_trace",
    ):
        assert key in aux, f"aux is missing {key}"
    assert aux["stage_llrs"], "deep supervision needs at least one stage"
    torch.testing.assert_close(aux["stage_llrs"][-1], final_llr)
    assert aux["bp_llr"].shape == final_llr.shape


def test_every_loss_term_trains_one_step(decoder) -> None:
    """The full pipeline must take a step under every term simultaneously."""
    from ldpc_corrector.losses.losses import TERM_NAMES

    config = LossConfig(**{term: True for term in TERM_NAMES})
    loss_fn = build_loss(config, decoder.graph)
    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad], lr=1e-3
    )

    channel_llr = torch.randn(8, 49) * 2.0
    target = torch.randint(0, 2, (8, 49))
    final_llr, aux = decoder(channel_llr, return_aux=True)
    loss, breakdown = loss_fn(final_llr, target, aux["bp_llr"], aux["stage_llrs"])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert np.isfinite(breakdown["total"])
    assert set(breakdown) == set(TERM_NAMES) | {"total"}


def test_learned_gate_is_per_bit_and_bounded(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    decoder = CorrectorDecoder(
        parity_check,
        corrector="gcn",
        bp_iterations=5,
        corrector_kwargs={"hidden_dim": 16, "num_layers": 1, "use_gate": True},
    )
    _, aux = decoder(torch.randn(4, 49) * 2.0, return_aux=True)
    assert aux["gate"].shape == (4, 49)
    assert torch.all((aux["gate"] > 0) & (aux["gate"] < 1))


def test_fixed_gate_scales_the_residual(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    torch.manual_seed(0)
    channel_llr = torch.randn(4, 49) * 2.0

    deltas = {}
    for gate in (1.0, 0.25):
        torch.manual_seed(11)
        decoder = CorrectorDecoder(
            parity_check,
            corrector="gcn",
            bp_iterations=5,
            fixed_gate=gate,
            corrector_kwargs={"hidden_dim": 16, "num_layers": 1},
        )
        with torch.no_grad():
            decoder.corrector.delta_head.weight.normal_(0.0, 0.3)
            decoder.corrector.delta_head.bias.fill_(0.2)
        decoder.eval()
        final_llr, aux = decoder(channel_llr, return_aux=True)
        deltas[gate] = final_llr - aux["bp_llr"]

    torch.testing.assert_close(deltas[0.25], 0.25 * deltas[1.0], atol=1e-6, rtol=1e-5)


def test_more_bp_iterations_decode_better(ldpc49_code) -> None:
    """Sanity check on the frozen baseline: BP(20) must beat BP(5)."""
    generator, parity_check = ldpc49_code
    generator = torch.as_tensor(generator, dtype=torch.float32)
    rng = torch.Generator().manual_seed(5)
    channel_llr, codewords, _, _ = sample_channel_batch(
        generator, 2048, 3.0, torch.device("cpu"), rng=rng
    )

    errors = {}
    for iterations in (5, 20):
        predicted = decode_bits(BPBaseline(parity_check, iterations)(channel_llr))
        errors[iterations] = (predicted != codewords).any(dim=1).float().mean().item()
    assert errors[20] < errors[5], f"BP(20) did not beat BP(5): {errors}"


def test_flops_grow_with_bp_iterations(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    bp5, bp20 = BPBaseline(parity_check, 5), BPBaseline(parity_check, 20)
    assert bp20.estimate_flops_per_frame() == pytest.approx(
        4 * bp5.estimate_flops_per_frame()
    )
    assert bp_flops_per_frame(bp5.graph, 0) == 0.0


def test_corrector_flops_include_the_bp_stage(decoder) -> None:
    bp_only = bp_flops_per_frame(decoder.graph, decoder.bp_iterations)
    assert decoder.estimate_flops_per_frame() > bp_only


def test_decoder_names_are_descriptive(ldpc49_code) -> None:
    _, parity_check = ldpc49_code
    assert BPBaseline(parity_check, 7).name == "BP(7)"
    decoder = CorrectorDecoder(
        parity_check, corrector="gat", bp_iterations=5,
        corrector_kwargs={"hidden_dim": 8, "heads": 1, "head_dim": 4},
    )
    assert decoder.name == "BP(5)+GAT"


def test_moving_the_decoder_keeps_the_graph_consistent(decoder) -> None:
    """`.to(device)` must move the dataclass graph, not just module buffers."""
    moved = decoder.to("cpu")
    assert moved.graph.edge_vn.device == moved.corrector.vn_to_cn.device
    assert moved.corrector.graph is moved.graph
    # Still decodes after the move.
    assert torch.isfinite(moved(torch.randn(2, 49) * 2.0)).all()
