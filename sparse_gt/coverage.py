"""Does the top-k least-confident set after BP(T) contain BP's errors?

The sparse-GT corrector can only change bits inside the selected set, so the
fraction of errors that land there is a hard ceiling on what it can fix. This
module measures that ceiling with plain BP, no learning involved.

For every frame still in error after ``T`` iterations, each bit is ranked by
``|LLR|`` ascending (rank 0 = least confident). Two histograms are kept:

* ``err_rank``: the rank of every erroneous bit -> **bit coverage**(k), the
  fraction of bit errors with rank < k.
* ``k_needed``: per failed frame, 1 + the largest rank among its errors, i.e.
  the smallest k whose set contains *all* of that frame's errors ->
  **frame coverage**(k). This is what matters for FER: a frame is only
  fixable if every wrong bit is selected.

Both are split by whether a long BP run (``reference_iterations``, default
100) eventually decodes the frame:

* *recoverable*: more BP iterations alone would fix it, so the corrector is
  competing with cheap extra iterations;
* *stuck*: BP never fixes it (trapping sets / oscillation). This is where a
  corrector can do something BP cannot.

Histograms let coverage be read off for any k afterwards, so k need not be
fixed before the run.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import torch

from .bp import SumProductBP
from .channel import awgn_llr
from .codes import LDPCCode

GROUPS = ("recoverable", "stuck")


def k_grid(n: int) -> list[int]:
    """k values reported in summaries: fixed sizes and fractions of n."""
    absolute = [4, 8, 16, 32, 64, 128, 256]
    fractions = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    ks = {k for k in absolute if k <= n} | {max(1, round(f * n)) for f in fractions}
    return sorted(ks)


class _Accumulator:
    """Coverage statistics for one (code, SNR, T)."""

    def __init__(self, n: int, m: int, T: int):
        self.n, self.m, self.T = n, m, T
        self.frames = 0
        self.failed = {g: 0 for g in GROUPS}
        self.err_rank = {g: np.zeros(n, np.int64) for g in GROUPS}
        self.k_needed = {g: np.zeros(n + 1, np.int64) for g in GROUPS}
        self.errors_per_frame = np.zeros(n + 1, np.int64)
        self.unsat_per_frame = np.zeros(m + 1, np.int64)
        # Sign-flip statistics over iterations 0..T, failed frames only.
        self.flipped_err = 0      # erroneous bits that flipped at least once
        self.err_bits = 0
        self.flipped_ok = 0       # correct bits that flipped at least once
        self.ok_bits = 0

    def update(self, trace: torch.Tensor, stuck: torch.Tensor, unsat: torch.Tensor) -> None:
        """``trace``: [T+1, B, n] outputs; ``stuck``: [B] bool; ``unsat``: [B] int."""
        out = trace[self.T]
        self.frames += out.shape[0]
        err = out < 0
        failed = err.any(1)
        if not bool(failed.any()):
            return

        err, out, stuck, unsat = err[failed], out[failed], stuck[failed], unsat[failed]
        sub_trace = trace[: self.T + 1, failed]
        ranks = out.abs().argsort(1).argsort(1)  # 0 = least confident
        k_needed = torch.where(err, ranks + 1, torch.zeros_like(ranks)).max(1).values
        n_err = err.sum(1)

        for group, mask in (("recoverable", ~stuck), ("stuck", stuck)):
            if not bool(mask.any()):
                continue
            self.failed[group] += int(mask.sum())
            self.err_rank[group] += torch.bincount(
                ranks[mask][err[mask]], minlength=self.n).cpu().numpy()
            self.k_needed[group] += torch.bincount(
                k_needed[mask], minlength=self.n + 1).cpu().numpy()

        self.errors_per_frame += torch.bincount(n_err, minlength=self.n + 1).cpu().numpy()
        self.unsat_per_frame += torch.bincount(unsat, minlength=self.m + 1).cpu().numpy()

        signs = sub_trace < 0
        flipped = (signs[1:] != signs[:-1]).any(0) if self.T > 0 else torch.zeros_like(err)
        self.flipped_err += int((flipped & err).sum())
        self.err_bits += int(err.sum())
        self.flipped_ok += int((flipped & ~err).sum())
        self.ok_bits += int((~err).sum())

    def summary(self) -> dict:
        ks = k_grid(self.n)
        total_failed = sum(self.failed.values())
        out = {
            "T": self.T,
            "frames": self.frames,
            "failed_frames": total_failed,
            "fer": total_failed / max(self.frames, 1),
            "failed_by_group": dict(self.failed),
            "k": ks,
            "bit_coverage": {},
            "frame_coverage": {},
        }
        for scope in ("all",) + GROUPS:
            groups = GROUPS if scope == "all" else (scope,)
            ranks = sum(self.err_rank[g] for g in groups)
            needed = sum(self.k_needed[g] for g in groups)
            cum_ranks = np.cumsum(ranks)
            cum_needed = np.cumsum(needed)
            n_err, n_frames = cum_ranks[-1], cum_needed[-1]
            out["bit_coverage"][scope] = [float(cum_ranks[k - 1] / n_err) if n_err else None for k in ks]
            out["frame_coverage"][scope] = [float(cum_needed[k] / n_frames) if n_frames else None for k in ks]

        def quantiles(hist):
            cdf = np.cumsum(hist)
            if cdf[-1] == 0:
                return None
            return {f"p{q}": int(np.searchsorted(cdf, q / 100 * cdf[-1])) for q in (10, 50, 90, 99)} | {
                "mean": float((np.arange(hist.size) * hist).sum() / cdf[-1])}

        out["errors_per_failed_frame"] = quantiles(self.errors_per_frame)
        out["unsatisfied_checks_per_failed_frame"] = quantiles(self.unsat_per_frame)
        out["k_needed_quantiles"] = {g: quantiles(self.k_needed[g]) for g in GROUPS}
        out["flips"] = {
            "p_flipped_given_error": self.flipped_err / max(self.err_bits, 1),
            "p_flipped_given_correct": self.flipped_ok / max(self.ok_bits, 1),
            "p_error_given_flipped": self.flipped_err / max(self.flipped_err + self.flipped_ok, 1),
            "p_error_baseline": self.err_bits / max(self.err_bits + self.ok_bits, 1),
        }
        return out

    def histograms(self) -> dict[str, np.ndarray]:
        h = {}
        for g in GROUPS:
            h[f"err_rank_{g}"] = self.err_rank[g]
            h[f"k_needed_{g}"] = self.k_needed[g]
        h["errors_per_frame"] = self.errors_per_frame
        h["unsat_per_frame"] = self.unsat_per_frame
        return h


@torch.no_grad()
def coverage_sweep(
    code: LDPCCode,
    ebn0_db: Iterable[float],
    T_values: Iterable[int] = (2, 3, 5, 8, 10, 15, 20),
    reference_iterations: int = 100,
    batch: int = 2048,
    target_failed: int = 1000,
    max_frames: int = 500_000,
    device: str = "cpu",
    seed: int = 1,
    log: Callable[[str], None] | None = print,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Coverage statistics for every (SNR, T).

    Each SNR point runs until BP(max T) has ``target_failed`` failed frames
    (so every smaller T has at least that many) or ``max_frames`` is hit.
    Returns per-(SNR, T) summaries and the raw histograms keyed
    ``"<ebn0>/<T>/<name>"``.
    """
    T_values = sorted(T_values)
    T_max = T_values[-1]
    bp = SumProductBP(code).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    summaries, hists = [], {}

    for snr in ebn0_db:
        accs = {T: _Accumulator(code.n, code.m, T) for T in T_values}
        frames = 0
        ref_fail = 0
        start = time.perf_counter()
        while accs[T_max].failed["recoverable"] + accs[T_max].failed["stuck"] < target_failed \
                and frames < max_frames:
            llr = awgn_llr(batch, code.n, snr, code.rate, device, gen)
            short = bp(llr, T_max, return_trace=True)
            long = bp(llr, reference_iterations)
            stuck = (long.llr < 0).any(1)
            ref_fail += int(stuck.sum())
            for T, acc in accs.items():
                unsat = bp.syndrome_weight(short.trace[T]).long()
                acc.update(short.trace, stuck, unsat)
            frames += batch

        for T, acc in accs.items():
            s = acc.summary()
            s.update(ebn0=float(snr), reference_iterations=reference_iterations,
                     fer_reference=ref_fail / frames)
            summaries.append(s)
            for name, h in acc.histograms().items():
                hists[f"{snr}/{T}/{name}"] = h
        if log:
            row = {T: accs[T].summary() for T in T_values}
            fers = "  ".join(f"T{T}:{row[T]['fer']:.2e}" for T in T_values)
            log(f"  {code.name} Eb/N0={snr:5.2f}  frames={frames}  FER {fers}  "
                f"BP{reference_iterations}:{ref_fail / frames:.2e}  "
                f"({time.perf_counter() - start:.1f}s)")
    return summaries, hists
