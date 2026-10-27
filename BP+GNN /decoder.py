"""End-to-end decoders: BP(k) alone, and BP(k) + a graph residual corrector.

Both expose ``forward(channel_llr) -> final_llr`` so the evaluation harness can
sweep them uniformly, plus ``estimate_flops_per_frame()`` so the cost/accuracy
tradeoff against BP(20) is a number rather than a hand wave.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ._compat import HybridBPGAT, LegacyTannerGraph, SumProductBP, hard_decode
from .bp_trace import (
    CN_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    VN_FEATURE_DIM,
    build_cn_features,
    build_vn_features,
    derive_features,
)
from .models import build_corrector
from .models.base_corrector import BaseCorrector, combine
from .tanner_graph import TannerGraph, build_tanner_graph


class BPBaseline(nn.Module):
    """Plain BP(k). Used for the BP(5) and BP(20) reference curves."""

    def __init__(
        self,
        parity_check: np.ndarray | Tensor,
        bp_iterations: int = 5,
        bp_damping: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph = build_tanner_graph(parity_check)
        self.bp = SumProductBP(
            LegacyTannerGraph.from_parity_check(parity_check),
            iterations=bp_iterations,
            damping=bp_damping,
        )
        self.bp_iterations = int(bp_iterations)

    def _apply(self, fn, *args, **kwargs):
        # ``self.graph`` is a plain dataclass, so ``.to(device)`` would move the
        # module buffers and leave the graph's index tensors behind.
        module = super()._apply(fn, *args, **kwargs)
        module.graph = module.graph.to(module.bp.edge_variables.device)
        return module

    def forward(self, channel_llr: Tensor) -> Tensor:
        return self.bp(channel_llr)

    @property
    def name(self) -> str:
        return f"BP({self.bp_iterations})"

    def estimate_flops_per_frame(self) -> float:
        return bp_flops_per_frame(self.graph, self.bp_iterations)


def bp_flops_per_frame(graph: TannerGraph, iterations: int) -> float:
    """Rough FLOP count for one frame of flooding sum-product BP.

    Per iteration and per edge the check update does a tanh, a forward/backward
    cumulative product, and an atanh; the variable update is a scatter-add plus
    a subtraction.  ``TANH_COST`` prices the transcendental calls, which
    dominate.  This is an order-of-magnitude model, consistent across variants,
    not a hardware-accurate count.
    """
    tanh_cost = 10.0
    per_edge = 2.0 * tanh_cost + 6.0  # tanh + atanh + cumprod/scatter arithmetic
    return float(iterations) * graph.n_edges * per_edge


class CorrectorDecoder(nn.Module):
    """Frozen BP(k), then one graph residual corrector, then optionally more BP.

    The BP stage runs under ``no_grad`` unless ``freeze_bp=False``, so by
    default no gradient flows into BP and only the corrector learns.

    With ``resume_iterations = r > 0`` the correction is not the end: it is
    added to the *channel* LLR of the bits it touched and BP resumes from its
    stopped messages for ``r`` more iterations, so a nudge on an uncertain bit
    can propagate through the code. With a zero correction this is exactly
    BP(k + r). The resumed BP stays differentiable, so training backpropagates
    through it into the corrector.
    """

    def __init__(
        self,
        parity_check: np.ndarray | Tensor,
        corrector: str = "gcn",
        bp_iterations: int = 5,
        bp_damping: float = 0.0,
        freeze_bp: bool = True,
        fixed_gate: float = 1.0,
        corrector_kwargs: Optional[Dict[str, Any]] = None,
        resume_iterations: int = 0,
    ) -> None:
        super().__init__()
        if resume_iterations < 0:
            raise ValueError("resume_iterations must be non-negative.")
        self.graph = build_tanner_graph(parity_check)
        legacy_graph = LegacyTannerGraph.from_parity_check(parity_check)
        self.bp = SumProductBP(legacy_graph, iterations=bp_iterations, damping=bp_damping)
        self.bp_iterations = int(bp_iterations)
        self.resume_iterations = int(resume_iterations)
        self.resume_bp = (
            SumProductBP(legacy_graph, iterations=resume_iterations, damping=bp_damping)
            if resume_iterations
            else None
        )
        self.freeze_bp = bool(freeze_bp)
        self.corrector_name = corrector

        kwargs = dict(corrector_kwargs or {})
        kwargs.setdefault("vn_dim", VN_FEATURE_DIM)
        kwargs.setdefault("cn_dim", CN_FEATURE_DIM)
        kwargs.setdefault("fixed_gate", fixed_gate)
        kwargs["graph"] = self.graph
        self.corrector: BaseCorrector = build_corrector(corrector, **kwargs)

        if self.freeze_bp:
            for parameter in self.bp.parameters():
                parameter.requires_grad_(False)

    # ------------------------------------------------------------- plumbing

    def _apply(self, fn, *args, **kwargs):
        # See BPBaseline._apply: keep the dataclass graph on the module device.
        module = super()._apply(fn, *args, **kwargs)
        device = module.corrector.vn_to_cn.device
        module.graph = module.graph.to(device)
        module.corrector.graph = module.graph
        return module

    def run_bp(self, channel_llr: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run the BP stage and return ``(bp_llr, trace)``.

        The trace carries the final directed messages as well as the LLR
        history, so edge-conditioned architectures can build edge features.
        """
        if self.freeze_bp:
            with torch.no_grad():
                bp_llr, v2c, c2v, trace = self.bp(
                    channel_llr, return_messages=True, return_trace=True
                )
            trace = {k: v.detach() for k, v in trace.items()}
            trace["variable_to_check"] = v2c.detach()
            trace["check_to_variable"] = c2v.detach()
            return bp_llr.detach(), trace

        bp_llr, v2c, c2v, trace = self.bp(
            channel_llr, return_messages=True, return_trace=True
        )
        trace = dict(trace)
        trace["variable_to_check"] = v2c
        trace["check_to_variable"] = c2v
        return bp_llr, trace

    def resume(
        self, channel_llr: Tensor, bp_llr: Tensor, corrected_llr: Tensor,
        trace: Dict[str, Tensor],
    ) -> Tensor:
        """Continue BP for ``resume_iterations`` with the correction folded in.

        ``corrected_llr - bp_llr`` is the gated correction; adding it to the
        channel LLR makes it a prior that every later variable-node update
        sees. No-op when ``resume_iterations == 0``.
        """
        if self.resume_bp is None:
            return corrected_llr
        prior = channel_llr + (corrected_llr - bp_llr)
        return self.resume_bp(
            prior,
            initial_messages=(trace["variable_to_check"], trace["check_to_variable"]),
        )

    def features(
        self, trace: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        derived = derive_features(trace["llr_trace"])
        vn_features = build_vn_features(self.graph, trace, derived)
        cn_features = build_cn_features(self.graph, trace)
        return vn_features, cn_features, derived

    # -------------------------------------------------------------- forward

    def forward(
        self, channel_llr: Tensor, return_aux: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        bp_llr, trace = self.run_bp(channel_llr)
        vn_features, cn_features, derived = self.features(trace)

        output = self.corrector(
            vn_features,
            cn_features,
            self.graph.vn_to_cn,
            self.graph.cn_to_vn,
            return_gate=True,
            return_stages=return_aux,
            **self.corrector.extra_inputs(trace),
        )
        final_llr = combine(
            bp_llr,
            output.delta_llr,
            output.gate_logits,
            fixed_gate=self.corrector.fixed_gate,
        )
        final_llr = self.resume(channel_llr, bp_llr, final_llr, trace)

        if not return_aux:
            return final_llr

        stage_llrs: List[Tensor] = [
            combine(
                bp_llr,
                stage_delta,
                output.gate_logits,
                fixed_gate=self.corrector.fixed_gate,
            )
            for stage_delta in (output.stage_deltas or [output.delta_llr])
        ]
        gate = (
            torch.sigmoid(output.gate_logits)
            if output.gate_logits is not None
            else torch.full_like(bp_llr, self.corrector.fixed_gate)
        )
        aux: Dict[str, Any] = {
            "bp_llr": bp_llr,
            "delta_llr": output.delta_llr,
            "gate": gate,
            "stage_llrs": stage_llrs,
            "confidence": derived["confidence"],
            "sign_flip_count": derived["sign_flip_count"],
            "check_parity_trace": trace["check_parity_trace"],
        }
        return final_llr, aux

    # ----------------------------------------------------------------- meta

    @property
    def name(self) -> str:
        name = f"BP({self.bp_iterations})+{self.corrector_name.upper()}"
        return name + (f"+BP({self.resume_iterations})" if self.resume_iterations else "")

    def estimate_flops_per_frame(self) -> float:
        """BP cost plus the corrector's dense-layer cost."""
        total = bp_flops_per_frame(self.graph, self.bp_iterations + self.resume_iterations)
        total += self.corrector.estimate_flops_per_frame()
        return total


class LegacyGATBPDecoder(nn.Module):
    """The Phase 0-2 ``HybridBPGAT`` behind this package's decoder interface.

    Lets the already-trained Aug-26 checkpoint be swept on exactly the same
    frames as the new correctors, which is the only way to tell whether the new
    architectures are an improvement on what the project already had.
    """

    def __init__(self, checkpoint: str | Path, parity_check=None) -> None:
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config")
        if config is None:
            raise ValueError(f"{checkpoint} has no embedded config.")

        if parity_check is None:
            parity_check = payload.get("parity_check")
        if parity_check is None:
            raise ValueError(
                f"{checkpoint} carries no parity-check matrix; pass one explicitly."
            )
        parity_check = np.asarray(parity_check)

        self.graph = build_tanner_graph(parity_check)
        self.bp_iterations = int(config.get("bp_iterations", 5))
        self.gat_every = int(config.get("gat_every", 0))
        self.hidden_dim = int(config.get("hidden_dim", 32))
        self.heads = int(config.get("heads", 2))
        self.head_dim = int(config.get("head_dim", 16))

        self.model = HybridBPGAT(
            parity_check,
            bp_iterations=self.bp_iterations,
            bp_damping=float(config.get("bp_damping", 0.0)),
            positional_dim=int(config.get("positional_dim", 0)),
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            head_dim=self.head_dim,
            dropout=float(config.get("dropout", 0.0)),
            correction_scale=float(config.get("correction_scale", 2.0)),
            gat_every=self.gat_every,
        )
        state = payload.get("model") or payload.get("model_state")
        if state is None:
            raise ValueError(f"{checkpoint} has no model weights.")

        # Checkpoints saved before iterative mode existed lack the V->C / C->V
        # message-correction heads. Those heads are unused when gat_every == 0,
        # so tolerate exactly those and nothing else -- silently dropping any
        # other weight would give a quietly wrong decoder.
        incompatible = self.model.load_state_dict(state, strict=False)
        allowed_prefixes = ("v2c_message_correction", "c2v_message_correction")
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"{checkpoint} does not match HybridBPGAT: "
                f"missing {unexpected_missing}, unexpected {incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys and self.gat_every:
            raise ValueError(
                f"{checkpoint} lacks the message-correction heads that "
                f"gat_every={self.gat_every} needs."
            )
        self.missing_unused_heads = tuple(incompatible.missing_keys)
        self.recorded_validation_fer = payload.get("validation_fer")

    def forward(self, channel_llr: Tensor) -> Tensor:
        return self.model(channel_llr)

    @property
    def name(self) -> str:
        mode = "iterative" if self.gat_every else "one-shot"
        return f"BP({self.bp_iterations})+GAT[legacy,{mode}]"

    def estimate_flops_per_frame(self) -> float:
        """Same accounting as GATCorrector, plus the per-edge correction MLP."""
        from .models.graph_ops import linear_flops

        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        projected = self.heads * self.head_dim
        edge_dim = 6
        # The legacy block is applied once per scheduled GAT round.
        rounds = max(1, self.bp_iterations // self.gat_every) if self.gat_every else 1

        block = 0.0
        for n_targets, n_sources in ((n_cn, n_vn), (n_vn, n_cn)):
            block += linear_flops(self.hidden_dim, projected, n_targets)
            block += 2 * linear_flops(self.hidden_dim, projected, n_sources)
            block += 2 * linear_flops(edge_dim, projected, n_edges)
            block += linear_flops(edge_dim, self.heads, n_edges)
            block += linear_flops(projected, self.hidden_dim, n_targets)
            block += 2 * linear_flops(self.hidden_dim, 2 * self.hidden_dim, n_targets)
            block += 6.0 * n_edges * projected

        # Correction heads run on 2*hidden + edge_dim inputs, once per edge.
        head_input = 2 * self.hidden_dim + edge_dim
        heads_used = 2 if self.gat_every else 1  # v2c+c2v deltas, or one posterior
        correction = heads_used * (
            linear_flops(head_input, self.hidden_dim, n_edges)
            + linear_flops(self.hidden_dim, 1, n_edges)
        )

        total = bp_flops_per_frame(self.graph, self.bp_iterations)
        return total + rounds * (block + correction)


def decode_bits(llr: Tensor) -> Tensor:
    """Hard decision, matching the Phase 0-2 convention (bit 1 <=> L < 0)."""
    return hard_decode(llr)
