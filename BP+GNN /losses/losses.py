"""Config-driven loss composition for the residual correctors.

Every term is independently toggleable and carries its own weight, so an
ablation is a config change rather than a new code path.  ``CompositeLoss``
returns the total *and* each term separately -- without the per-term breakdown
there is no way to tell which term is actually contributing.

Sign convention throughout: LLR is positive for bit 0, so the BCE logit for
"this bit is 1" is ``-llr``, and ``signed_llr = llr * (1 - 2 * target)`` is
positive exactly when the decision is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..tanner_graph import TannerGraph

TERM_NAMES = (
    "bce",
    "deep_supervision",
    "focal",
    "hard_mining",
    "block_soft_bler",
    "syndrome_aux",
)


@dataclass
class LossConfig:
    """Which terms are on, and with what weight."""

    bce: bool = True
    bce_weight: float = 1.0

    deep_supervision: bool = False
    deep_supervision_weight: float = 1.0
    deep_supervision_gamma: float = 0.5  # discount applied to earlier layers

    focal: bool = False
    focal_weight: float = 1.0
    focal_gamma: float = 2.0

    hard_mining: bool = False
    hard_mining_weight: float = 1.0
    hard_mining_threshold: float = 1.0  # |LLR| below this counts as uncertain

    block_soft_bler: bool = False
    block_soft_bler_weight: float = 1.0
    block_soft_bler_temperature: float = 1.0

    # Auxiliary only: a small weight nudging the output toward a valid
    # codeword. Never a standalone objective -- it is minimized by any
    # codeword, including the wrong one.
    syndrome_aux: bool = False
    syndrome_aux_weight: float = 0.05

    def enabled_terms(self) -> List[str]:
        return [name for name in TERM_NAMES if getattr(self, name)]

    def validate(self) -> None:
        enabled = self.enabled_terms()
        if not enabled:
            raise ValueError("At least one loss term must be enabled.")
        if enabled == ["syndrome_aux"]:
            raise ValueError(
                "syndrome_aux is auxiliary only and cannot be the sole loss term: "
                "it is minimized by any valid codeword, including a wrong one."
            )
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if not 0.0 < self.deep_supervision_gamma <= 1.0:
            raise ValueError("deep_supervision_gamma must be in (0, 1].")
        if self.block_soft_bler_temperature <= 0.0:
            raise ValueError("block_soft_bler_temperature must be positive.")
        if self.hard_mining_threshold < 0.0:
            raise ValueError("hard_mining_threshold must be non-negative.")

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, object]]) -> "LossConfig":
        if not mapping:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(mapping) - known
        if unknown:
            raise KeyError(f"Unknown loss config keys: {sorted(unknown)}")
        return cls(**dict(mapping))  # type: ignore[arg-type]


# --------------------------------------------------------------- primitives


def per_bit_bce(llr: Tensor, target: Tensor) -> Tensor:
    """Elementwise BCE on ``sigmoid(-llr)``; shape matches ``llr``."""
    return F.binary_cross_entropy_with_logits(
        -llr, target.to(llr.dtype), reduction="none"
    )


def signed_llr(llr: Tensor, target: Tensor) -> Tensor:
    """Positive where the hard decision is correct, negative where it is wrong."""
    return llr * (1.0 - 2.0 * target.to(llr.dtype))


def bce_loss(llr: Tensor, target: Tensor) -> Tensor:
    return per_bit_bce(llr, target).mean()


def focal_loss(llr: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    """Focal reweighting: down-weight bits the decoder already gets right."""
    bce = per_bit_bce(llr, target)
    # p_t is the probability assigned to the true bit value.
    probability_true = torch.sigmoid(signed_llr(llr, target))
    return ((1.0 - probability_true).pow(gamma) * bce).mean()


def hard_mining_loss(
    llr: Tensor,
    target: Tensor,
    bp_llr: Tensor,
    threshold: float = 1.0,
) -> Tensor:
    """BCE restricted to the bits the corrector actually has to fix.

    A bit is selected when BP(k) decided it wrongly, or when BP(k) was
    unconfident (``|LLR| < threshold``).  The mask is detached: it selects
    bits, it does not carry gradient.
    """
    with torch.no_grad():
        bp_wrong = (bp_llr < 0).to(torch.bool) != target.to(torch.bool)
        uncertain = bp_llr.abs() < threshold
        mask = (bp_wrong | uncertain).to(llr.dtype)

    bce = per_bit_bce(llr, target)
    denominator = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denominator


def block_soft_bler_loss(
    llr: Tensor, target: Tensor, temperature: float = 1.0
) -> Tensor:
    """Differentiable block-error surrogate.

    A block is wrong if *any* bit is wrong, so the relevant quantity is a
    soft-min over per-bit correctness -- equivalently a log-sum-exp soft-max
    over the per-bit error logits.  ``softplus`` of that turns it into a
    per-block BCE on the worst bit.  Lower ``temperature`` is closer to a true
    max, and therefore closer to true BLER but with sparser gradients.
    """
    error_logit = -signed_llr(llr, target)
    worst = temperature * torch.logsumexp(error_logit / temperature, dim=-1)
    return F.softplus(worst).mean()


def syndrome_loss(llr: Tensor, graph: TannerGraph) -> Tensor:
    """Soft penalty for violating even parity.

    ``tanh(L/2)`` is a soft +/-1 bit, so the product over a check should reach
    +1.  Auxiliary only: it says nothing about *which* codeword.
    """
    soft_bits = torch.tanh(0.5 * llr)
    parity = graph.product_to_cn(graph.gather_vn(soft_bits))
    even_probability = (0.5 * (1.0 + parity)).clamp_min(1e-6)
    return -torch.log(even_probability).mean()


def deep_supervision_loss(
    stage_llrs: Sequence[Tensor], target: Tensor, gamma: float = 0.5
) -> Tensor:
    """BCE at every unrolled layer, discounting earlier layers by ``gamma``.

    The final layer always carries weight 1; layer ``S - 1 - i`` carries
    ``gamma ** i``.  Weights are normalized so the term does not grow with
    depth.
    """
    if not stage_llrs:
        raise ValueError("deep_supervision needs at least one stage LLR.")

    total = stage_llrs[-1].new_zeros(())
    weight_sum = 0.0
    last = len(stage_llrs) - 1
    for index, stage in enumerate(stage_llrs):
        weight = gamma ** (last - index)
        total = total + weight * bce_loss(stage, target)
        weight_sum += weight
    return total / weight_sum


# --------------------------------------------------------------- composite


class CompositeLoss(nn.Module):
    """Weighted sum of the enabled terms, with a per-term breakdown."""

    def __init__(self, config: LossConfig, graph: Optional[TannerGraph] = None) -> None:
        super().__init__()
        config.validate()
        if config.syndrome_aux and graph is None:
            raise ValueError("syndrome_aux needs the Tanner graph.")
        self.config = config
        self.graph = graph

    def forward(
        self,
        final_llr: Tensor,
        target: Tensor,
        bp_llr: Optional[Tensor] = None,
        stage_llrs: Optional[Sequence[Tensor]] = None,
    ) -> tuple[Tensor, Dict[str, float]]:
        config = self.config
        target = target.to(final_llr.dtype)
        total = final_llr.new_zeros(())
        breakdown: Dict[str, float] = {}

        def add(name: str, value: Tensor, weight: float) -> None:
            nonlocal total
            total = total + weight * value
            breakdown[name] = float(value.detach())

        if config.bce:
            add("bce", bce_loss(final_llr, target), config.bce_weight)

        if config.deep_supervision:
            if not stage_llrs:
                raise ValueError(
                    "deep_supervision is enabled but no stage LLRs were provided; "
                    "call the decoder with return_aux=True."
                )
            add(
                "deep_supervision",
                deep_supervision_loss(
                    stage_llrs, target, config.deep_supervision_gamma
                ),
                config.deep_supervision_weight,
            )

        if config.focal:
            add(
                "focal",
                focal_loss(final_llr, target, config.focal_gamma),
                config.focal_weight,
            )

        if config.hard_mining:
            if bp_llr is None:
                raise ValueError("hard_mining needs the BP(k) LLR.")
            add(
                "hard_mining",
                hard_mining_loss(
                    final_llr, target, bp_llr, config.hard_mining_threshold
                ),
                config.hard_mining_weight,
            )

        if config.block_soft_bler:
            add(
                "block_soft_bler",
                block_soft_bler_loss(
                    final_llr, target, config.block_soft_bler_temperature
                ),
                config.block_soft_bler_weight,
            )

        if config.syndrome_aux:
            assert self.graph is not None  # guaranteed by __init__
            add(
                "syndrome_aux",
                syndrome_loss(final_llr, self.graph),
                config.syndrome_aux_weight,
            )

        breakdown["total"] = float(total.detach())
        return total, breakdown


def build_loss(
    config: Optional[Mapping[str, object] | LossConfig],
    graph: Optional[TannerGraph] = None,
) -> CompositeLoss:
    if isinstance(config, LossConfig):
        return CompositeLoss(config, graph)
    return CompositeLoss(LossConfig.from_mapping(config), graph)
