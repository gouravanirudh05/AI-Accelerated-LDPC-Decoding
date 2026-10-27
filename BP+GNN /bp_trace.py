"""Features derived from a truncated BP(k) run.

``SumProductBP.forward(..., return_trace=True)`` returns the per-iteration LLR
and soft-parity history.  This module turns that history into the variable-node
and check-node feature tensors the correctors consume.

All tensors are batched: the spec's ``(K_bp, N)`` trace is ``[B, K_bp, N]``
here, and its ``(N,)`` per-bit features are ``[B, N]``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .tanner_graph import TannerGraph

# LLRs grow without bound at high SNR; dividing by this keeps the raw-magnitude
# features in a range the networks can use while leaving the sign informative.
LLR_SCALE = 8.0

VN_FEATURE_DIM = 8
CN_FEATURE_DIM = 6
EDGE_FEATURE_DIM = 6

# Named channel positions in the VN feature vector, so downstream models can
# reach a specific signal without hard-coding a bare integer.
VN_IDX_SOFT_CHANNEL = 0
VN_IDX_SOFT_FINAL = 1
VN_IDX_CHANNEL_LLR = 2
VN_IDX_FINAL_LLR = 3
VN_IDX_CONFIDENCE = 4
VN_IDX_SIGN_FLIPS = 5
VN_IDX_DEGREE = 6
VN_IDX_VIOLATION = 7


def derive_features(llr_trace: Tensor) -> Dict[str, Tensor]:
    """Reduce an LLR trace to per-bit features.

    ``llr_trace`` is ``[B, K, N]`` (or ``[K, N]``, which is promoted).  Returns
    ``sign_flip_count``, ``final_llr``, and ``confidence = |final_llr|``, each
    ``[B, N]``, plus ``min_confidence`` over the trace.

    A bit whose hard decision oscillates across iterations gets a nonzero
    ``sign_flip_count``; that is the signal telling the corrector which bits BP
    never settled.
    """
    if llr_trace.ndim == 2:
        llr_trace = llr_trace.unsqueeze(0)
    if llr_trace.ndim != 3:
        raise ValueError("llr_trace must be [B, K, N] or [K, N].")

    batch, iterations, n_variables = llr_trace.shape
    if iterations == 0:
        zeros = llr_trace.new_zeros((batch, n_variables))
        return {
            "sign_flip_count": zeros,
            "final_llr": zeros.clone(),
            "confidence": zeros.clone(),
            "min_confidence": zeros.clone(),
        }

    # Hard decision per iteration, matching gat_bp.hard_decode (bit 1 <=> L<0).
    bits = (llr_trace < 0).to(llr_trace.dtype)
    if iterations > 1:
        flips = (bits[:, 1:] != bits[:, :-1]).to(llr_trace.dtype).sum(dim=1)
    else:
        flips = llr_trace.new_zeros((batch, n_variables))

    final_llr = llr_trace[:, -1]
    magnitude = final_llr.abs()
    return {
        "sign_flip_count": flips,
        "final_llr": final_llr,
        "confidence": magnitude,
        "min_confidence": llr_trace.abs().amin(dim=1),
    }


def build_vn_features(
    graph: TannerGraph,
    trace: Dict[str, Tensor],
    derived: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Assemble ``[B, n, VN_FEATURE_DIM]`` variable-node features."""
    if derived is None:
        derived = derive_features(trace["llr_trace"])

    channel_llr = trace["channel_llr"]
    final_llr = derived["final_llr"]
    iterations = trace["llr_trace"].shape[1]
    flip_norm = max(iterations - 1, 1)

    parity_trace = trace["check_parity_trace"]
    if parity_trace.shape[1] > 0:
        # Soft violation probability per check, spread back onto variables.
        violation = 0.5 * (1.0 - parity_trace[:, -1])
        incident_violation = graph.aggregate_to_vn(
            graph.gather_cn(violation), reduce="mean"
        )
    else:
        incident_violation = torch.zeros_like(channel_llr)

    degree_norm = graph.vn_degrees / graph.vn_degrees.max().clamp_min(1.0)

    features = torch.stack(
        (
            torch.tanh(0.5 * channel_llr),
            torch.tanh(0.5 * final_llr),
            channel_llr / LLR_SCALE,
            final_llr / LLR_SCALE,
            derived["confidence"] / LLR_SCALE,
            derived["sign_flip_count"] / flip_norm,
            degree_norm.expand_as(channel_llr),
            incident_violation,
        ),
        dim=-1,
    )
    if features.shape[-1] != VN_FEATURE_DIM:  # pragma: no cover - guard
        raise RuntimeError("VN_FEATURE_DIM is out of sync with build_vn_features.")
    return features


def build_cn_features(graph: TannerGraph, trace: Dict[str, Tensor]) -> Tensor:
    """Assemble ``[B, m, CN_FEATURE_DIM]`` check-node features."""
    parity_trace = trace["check_parity_trace"]
    channel_llr = trace["channel_llr"]
    batch = channel_llr.shape[0]

    if parity_trace.shape[1] > 0:
        final_parity = parity_trace[:, -1]
        mean_parity = parity_trace.mean(dim=1)
        # How much the check's parity moved across iterations: a check that
        # never settled is where the corrector should look.
        parity_spread = (
            parity_trace.std(dim=1, unbiased=False)
            if parity_trace.shape[1] > 1
            else torch.zeros_like(final_parity)
        )
    else:
        final_parity = channel_llr.new_zeros((batch, graph.n_checks))
        mean_parity = final_parity.clone()
        parity_spread = final_parity.clone()

    llr_trace = trace["llr_trace"]
    final_llr = llr_trace[:, -1] if llr_trace.shape[1] > 0 else channel_llr
    soft_bits = torch.tanh(0.5 * final_llr)
    mean_incident_magnitude = graph.aggregate_to_cn(
        graph.gather_vn(soft_bits.abs()), reduce="mean"
    )
    degree_norm = graph.cn_degrees / graph.cn_degrees.max().clamp_min(1.0)

    features = torch.stack(
        (
            final_parity,
            0.5 * (1.0 - final_parity),
            mean_parity,
            parity_spread,
            degree_norm.expand_as(final_parity),
            mean_incident_magnitude,
        ),
        dim=-1,
    )
    if features.shape[-1] != CN_FEATURE_DIM:  # pragma: no cover - guard
        raise RuntimeError("CN_FEATURE_DIM is out of sync with build_cn_features.")
    return features


def build_edge_features(
    variable_to_check: Tensor, check_to_variable: Tensor
) -> Tensor:
    """Assemble ``[B, E, EDGE_FEATURE_DIM]`` features for the two BP channels.

    Mirrors the Phase 0-2 edge features -- signed, magnitude, and soft-bit
    views of both directed BP messages -- with the raw magnitudes divided by
    ``LLR_SCALE``, since those grow without bound at high SNR.
    """
    features = torch.stack(
        (
            variable_to_check / LLR_SCALE,
            check_to_variable / LLR_SCALE,
            variable_to_check.abs() / LLR_SCALE,
            check_to_variable.abs() / LLR_SCALE,
            torch.tanh(0.5 * variable_to_check),
            torch.tanh(0.5 * check_to_variable),
        ),
        dim=-1,
    )
    if features.shape[-1] != EDGE_FEATURE_DIM:  # pragma: no cover - guard
        raise RuntimeError("EDGE_FEATURE_DIM is out of sync with build_edge_features.")
    return features


def run_bp_with_features(
    bp: torch.nn.Module,
    graph: TannerGraph,
    channel_llr: Tensor,
    detach: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Run frozen BP(k) and build corrector inputs.

    Returns ``(bp_llr, vn_features, cn_features, trace)``.  ``detach=True``
    blocks gradient flow into BP, which is the default training regime: the BP
    stage is frozen and only the corrector learns.
    """
    context = torch.no_grad() if detach else torch.enable_grad()
    with context:
        bp_llr, trace = bp(channel_llr, return_trace=True)

    if detach:
        bp_llr = bp_llr.detach()
        trace = {key: value.detach() for key, value in trace.items()}

    derived = derive_features(trace["llr_trace"])
    vn_features = build_vn_features(graph, trace, derived)
    cn_features = build_cn_features(graph, trace)
    trace = {**trace, **derived}
    return bp_llr, vn_features, cn_features, trace
