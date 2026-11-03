"""BPSK over AWGN, producing channel LLRs.

Sign convention used everywhere in this package: bit 0 -> symbol +1, and a
**positive LLR means bit 0**. The hard decision is ``llr < 0``.

Frames carry the all-zero codeword. For a linear code, a symmetric channel
(BPSK/AWGN) and a symmetric decoder (sum-product BP), the error probability is
the same for every transmitted codeword, so this is exact rather than an
approximation. It is also what the AFF3CT reference curves in
``codes/references`` use (``--src-type AZCW``). A learned decoder is *not*
automatically symmetric, so training/evaluating one on the all-zero codeword
needs care; that is handled where the model is introduced, not here.
"""

from __future__ import annotations

import math

import torch


def ebn0_to_sigma(ebn0_db: float | torch.Tensor, rate: float) -> float | torch.Tensor:
    """Noise standard deviation for unit-energy BPSK at a given Eb/N0 (dB).

    Es = R * Eb and sigma^2 = N0 / 2 with Es = 1, so sigma^2 = 1 / (2 R Eb/N0).
    """
    if isinstance(ebn0_db, torch.Tensor):
        return torch.sqrt(1.0 / (2.0 * rate * 10.0 ** (ebn0_db / 10.0)))
    return math.sqrt(1.0 / (2.0 * rate * 10.0 ** (ebn0_db / 10.0)))


def awgn_llr(
    batch: int,
    n: int,
    ebn0_db: float | torch.Tensor,
    rate: float,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Channel LLRs ``[batch, n]`` for the all-zero codeword.

    ``ebn0_db`` may be a scalar or a ``[batch]`` tensor (one SNR per frame,
    for training over an SNR range).
    """
    sigma = ebn0_to_sigma(ebn0_db, rate)
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=device, dtype=torch.float32).view(-1, 1)
    noise = torch.randn(batch, n, device=device, generator=generator)
    y = 1.0 + sigma * noise
    return 2.0 * y / sigma**2
