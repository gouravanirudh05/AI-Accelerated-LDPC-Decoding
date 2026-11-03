"""GPU energy per call, from NVML, plus CUDA-graph capture.

Energy is read from the GPU's cumulative energy counter
(``nvmlDeviceGetTotalEnergyConsumption``, millijoules, Volta and later; the
Kaggle T4 is Turing). If that is unavailable, average power is sampled in a
background thread and integrated. Each measurement runs the workload in a
loop for at least ``min_seconds`` so the counter's ~10-100 ms update period
is negligible, and reports both total and above-idle (dynamic) energy.

``cuda_graph`` captures a fixed-shape, sync-free callable once and replays
it, removing per-kernel launch overhead -- the dominant cost of message
passing at small batch on a GPU.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:  # no NVML (CPU smoke runs, or the package is missing)
    pynvml = None
    _HANDLE = None
    _NVML = False


def nvml_available() -> bool:
    return _NVML


def _energy_mj() -> float | None:
    try:
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE))
    except Exception:
        return None


def _power_w() -> float:
    return pynvml.nvmlDeviceGetPowerUsage(_HANDLE) / 1000.0


def idle_power_w(seconds: float = 2.0) -> float | None:
    if not _NVML:
        return None
    torch.cuda.synchronize()
    samples, end = [], time.time() + seconds
    while time.time() < end:
        samples.append(_power_w())
        time.sleep(0.02)
    return sum(samples) / len(samples)


def measure_energy(fn: Callable[[], object], min_seconds: float = 3.0,
                   idle_w: float | None = None) -> dict | None:
    """Joules per call of ``fn`` on GPU 0 (None without NVML)."""
    if not _NVML:
        return None
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(_power_w())
            time.sleep(0.01)

    thread = threading.Thread(target=sampler, daemon=True)
    e0 = _energy_mj()
    thread.start()
    t0 = time.perf_counter()
    calls = 0
    while True:
        fn()
        calls += 1
        if calls % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= min_seconds:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    e1 = _energy_mj()
    stop.set()
    thread.join()

    mean_power = sum(samples) / max(len(samples), 1)
    total_j = (e1 - e0) / 1000.0 if e0 is not None and e1 is not None else mean_power * elapsed
    out = {"joules_per_call": total_j / calls, "mean_power_w": mean_power,
           "seconds": elapsed, "calls": calls,
           "source": "energy_counter" if e0 is not None else "power_sampling"}
    if idle_w is not None:
        out["dynamic_joules_per_call"] = max(total_j - idle_w * elapsed, 0.0) / calls
        out["idle_power_w"] = idle_w
    return out


def cuda_graph(fn: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor):
    """Capture ``fn(example)`` in a CUDA graph; returns ``run(x) -> output``.

    ``fn`` must use static shapes and no host synchronisation. The returned
    output tensor is reused across replays (clone it to keep a result).
    """
    static_in = example.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn(static_in)

    def run(x: torch.Tensor | None = None) -> torch.Tensor:
        if x is not None:
            static_in.copy_(x)
        graph.replay()
        return static_out

    return run
