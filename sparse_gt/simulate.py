"""Monte-Carlo BER/FER simulation and AFF3CT reference parsing."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

from .bp import SumProductBP
from .channel import awgn_llr
from .codes import LDPCCode


@torch.no_grad()
def simulate_bp(
    code: LDPCCode,
    ebn0_db: Iterable[float],
    iterations: int,
    batch: int = 4096,
    min_frame_errors: int = 100,
    max_frames: int = 1_000_000,
    device: str = "cpu",
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """BER/FER of sum-product BP with syndrome early stopping.

    Each SNR point runs until ``min_frame_errors`` frame errors or
    ``max_frames`` frames, whichever comes first (AFF3CT's stopping rule).
    """
    bp = SumProductBP(code).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for snr in ebn0_db:
        frames = bit_err = frame_err = undetected = iters = 0
        start = time.perf_counter()
        while frame_err < min_frame_errors and frames < max_frames:
            llr = awgn_llr(batch, code.n, snr, code.rate, device, gen)
            res = bp(llr, iterations)
            errors = (res.llr < 0).sum(1)  # all-zero codeword: any 1 is an error
            bit_err += int(errors.sum())
            frame_err += int((errors > 0).sum())
            undetected += int(((errors > 0) & res.converged).sum())
            iters += int(res.iterations.sum())
            frames += batch
        row = {
            "ebn0": float(snr),
            "frames": frames,
            "bit_errors": bit_err,
            "frame_errors": frame_err,
            "undetected_errors": undetected,
            "ber": bit_err / (frames * code.n),
            "fer": frame_err / frames,
            "avg_iterations": iters / frames,
            "seconds": time.perf_counter() - start,
        }
        rows.append(row)
        if log:
            log(f"  {code.name} BP{iterations} Eb/N0={snr:5.2f} dB  BER={row['ber']:.3e}  "
                f"FER={row['fer']:.3e}  ({frame_err} FE / {frames} frames, "
                f"avg it {row['avg_iterations']:.2f}, {row['seconds']:.1f}s)")
    return rows


def read_aff3ct_reference(path: str | Path) -> list[dict]:
    """Parse the data table of an AFF3CT ``error_rate_references`` file.

    Columns: Es/N0 | Eb/N0 | FRA | BE/FE | FE | BER | FER || ...
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith("#") or "|" not in line:
            continue
        cells = [c.strip() for c in re.split(r"\|\|?", line)]
        try:
            esn0, ebn0, fra, be_fe, fe, ber, fer = (float(c) for c in cells[:7])
        except ValueError:
            continue
        rows.append({"ebn0": ebn0, "frames": int(fra), "frame_errors": int(fe),
                     "ber": ber, "fer": fer})
    return rows


def fer_interval(frame_errors: int, frames: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a frame error rate."""
    if frames == 0:
        return 0.0, 1.0
    p = frame_errors / frames
    denom = 1 + z * z / frames
    centre = (p + z * z / (2 * frames)) / denom
    half = z * math.sqrt(p * (1 - p) / frames + z * z / (4 * frames * frames)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
