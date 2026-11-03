"""GPU inference timing: the currency of the equal-latency comparison.

Every decoder variant (plain BP now, BP + sparse GT later) is timed with the
same harness, :func:`gpu_time`, so their costs are directly comparable:

* CUDA events around the call, after warm-up, on a synchronised device;
* median and p10/p90 over repeats (GPU timings have a long upper tail);
* both **latency** (batch 1, what a single frame waits) and **throughput**
  (frames/s at large batch, what a base station cares about).

Timing must be done on the same GPU, in the same process, for any two
numbers that are compared; absolute values differ between Kaggle machines.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import torch

from .bp import SumProductBP
from .channel import awgn_llr
from .codes import LDPCCode


def gpu_time(fn: Callable[[], object], device: str, warmup: int = 3, repeats: int = 20) -> dict:
    """Wall-clock milliseconds of ``fn()`` (median, p10, p90) on ``device``."""
    cuda = device.startswith("cuda")
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    t = np.array(times)
    return {"ms": float(np.median(t)), "p10": float(np.percentile(t, 10)),
            "p90": float(np.percentile(t, 90))}


def select_least_confident(llr: torch.Tensor, k: int) -> torch.Tensor:
    """Indices ``[B, k]`` of the k smallest ``|LLR|`` per frame (the GT's input set)."""
    return torch.topk(llr.abs(), k, dim=1, largest=False, sorted=False).indices


@torch.no_grad()
def inference_timing(
    code: LDPCCode,
    device: str,
    batches: Iterable[int] = (1, 64, 1024, 8192),
    T_values: Iterable[int] = (2, 3, 5, 8, 10, 15, 20, 50, 100),
    k_values: Iterable[int] = (),
    early_stop_snr: float | None = None,
    early_stop_iterations: int = 100,
    max_elements: int = 128_000_000,
    repeats: int = 20,
) -> list[dict]:
    """Time BP and top-k selection on ``device``.

    Rows have ``kind`` in ``bp_fixed`` (exactly T iterations, no early stop:
    the worst-case / fixed-latency cost), ``topk`` (selection only), and
    ``bp_early_stop`` (up to ``early_stop_iterations`` with syndrome stopping
    at ``early_stop_snr``: the average cost in practice, including the
    per-iteration device->host sync that early stopping needs).
    """
    bp = SumProductBP(code).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    rows = []
    for batch in batches:
        if batch * code.num_edges > max_elements:
            continue
        llr = awgn_llr(batch, code.n, early_stop_snr if early_stop_snr is not None else 2.0,
                       code.rate, device, gen)
        for T in T_values:
            reps = repeats if T <= 20 else max(3, repeats // 4)
            t = gpu_time(lambda: bp(llr, T, early_stop=False), device, repeats=reps)
            rows.append({"kind": "bp_fixed", "batch": batch, "T": T, **t,
                         "frames_per_s": batch / t["ms"] * 1e3})
        for k in k_values:
            if k > code.n:
                continue
            t = gpu_time(lambda: select_least_confident(llr, k), device, repeats=repeats)
            rows.append({"kind": "topk", "batch": batch, "k": k, **t})
        if early_stop_snr is not None:
            # Fresh noise per repeat would change the work; fix one draw so the
            # repeats time the same decode, and report its iteration count.
            used = bp(llr, early_stop_iterations).iterations.float()
            t = gpu_time(lambda: bp(llr, early_stop_iterations), device,
                         repeats=max(3, repeats // 2))
            rows.append({"kind": "bp_early_stop", "batch": batch, "ebn0": early_stop_snr,
                         "max_iterations": early_stop_iterations, **t,
                         "mean_iterations": float(used.mean()),
                         "max_iterations_used": int(used.max()),
                         "frames_per_s": batch / t["ms"] * 1e3})
    return rows
