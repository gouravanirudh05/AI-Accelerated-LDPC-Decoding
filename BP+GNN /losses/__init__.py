"""Config-driven loss terms for the residual correctors."""

from __future__ import annotations

from .losses import (
    TERM_NAMES,
    CompositeLoss,
    LossConfig,
    bce_loss,
    block_soft_bler_loss,
    build_loss,
    deep_supervision_loss,
    focal_loss,
    hard_mining_loss,
    per_bit_bce,
    signed_llr,
    syndrome_loss,
)

__all__ = [
    "TERM_NAMES",
    "CompositeLoss",
    "LossConfig",
    "build_loss",
    "bce_loss",
    "block_soft_bler_loss",
    "deep_supervision_loss",
    "focal_loss",
    "hard_mining_loss",
    "per_bit_bce",
    "signed_llr",
    "syndrome_loss",
]
