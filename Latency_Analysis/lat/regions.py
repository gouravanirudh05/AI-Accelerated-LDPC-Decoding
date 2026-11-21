"""Named timing regions with three modes.

* ``off``      : no-op, the code runs as the original decoder.
* ``sync``     : device synchronise at every region boundary and wall-clock the
                 region. Gives the end-to-end cost of each component *including*
                 its kernel-launch overhead, at the price of removing CPU/GPU
                 overlap (the sum of regions is larger than the plain run; the
                 report shows both).
* ``profiler`` : torch.autograd.profiler.record_function ranges, analysed by
                 ``attribute()`` from a torch.profiler trace: per region the GPU
                 kernel time, the number of kernels launched and the host (CPU)
                 time. Host time much larger than kernel time means the region is
                 launch/overhead bound, not compute bound.

Regions nest; results are keyed by the path "stage1/bp/cn".
"""

import contextlib
import time
from collections import defaultdict

import torch

PREFIX = "R::"


class Regions:
    def __init__(self, mode="off", device=None):
        assert mode in ("off", "sync", "profiler")
        self.mode = mode
        self.device = device
        self.stack = []
        self.total = defaultdict(float)   # ms, inclusive
        self.calls = defaultdict(int)

    def _sync(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def __call__(self, name):
        if self.mode == "off":
            yield
            return
        self.stack.append(name)
        path = "/".join(self.stack)
        try:
            if self.mode == "profiler":
                with torch.autograd.profiler.record_function(PREFIX + path):
                    yield
            else:
                self._sync()
                t0 = time.perf_counter()
                yield
                self._sync()
                self.total[path] += (time.perf_counter() - t0) * 1e3
                self.calls[path] += 1
        finally:
            self.stack.pop()

    def reset(self):
        self.total.clear()
        self.calls.clear()

    def table(self, reps=1):
        """{path: dict(incl_ms, self_ms, calls)} per decode call (divided by reps)."""
        out = {}
        for p, t in self.total.items():
            kids = sum(v for q, v in self.total.items()
                       if q.startswith(p + "/") and q.count("/") == p.count("/") + 1)
            out[p] = dict(incl_ms=t / reps, self_ms=(t - kids) / reps, calls=self.calls[p] / reps)
        return out


OFF = Regions("off")


def _dev_self(e):
    for a in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, a, None)
        if v is not None:
            return float(v)
    return 0.0


def _is_cpu(e):
    dt = getattr(e, "device_type", None)
    return dt is None or "CPU" in str(dt)


LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch")
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy")


def attribute(prof, reps=1):
    """Per region path: GPU kernel time (self, ms), kernels launched, host time
    (inclusive, ms) and time the host spent blocked in synchronising CUDA calls.

    Every CPU event is assigned to its innermost enclosing region (walking
    cpu_parent); kernel time is the event's self device time, so nothing is
    counted twice."""
    res = defaultdict(lambda: dict(gpu_ms=0.0, kernels=0, host_ms=0.0, host_calls=0, sync_ms=0.0,
                                   syncs=0, ops=0))
    for e in prof.events():
        if not _is_cpu(e):
            continue
        if e.name.startswith(PREFIX):
            r = res[e.name[len(PREFIX):]]
            r["host_ms"] += e.cpu_time_total / 1e3
            r["host_calls"] += 1
            continue
        top_level = e.cpu_parent is None or e.cpu_parent.name.startswith(PREFIX)
        p = e.cpu_parent
        while p is not None and not p.name.startswith(PREFIX):
            p = p.cpu_parent
        key = p.name[len(PREFIX):] if p is not None else "(outside regions)"
        r = res[key]
        r["gpu_ms"] += _dev_self(e) / 1e3
        if top_level and e.name.startswith("aten::"):
            r["ops"] += 1
        if e.name in LAUNCH_NAMES:
            r["kernels"] += 1
        if e.name in SYNC_NAMES:
            r["syncs"] += 1
            r["sync_ms"] += e.cpu_time_total / 1e3
    out = {}
    for k, v in res.items():
        out[k] = {a: (b / reps if isinstance(b, float) or a in ("kernels", "host_calls", "syncs", "ops") else b)
                  for a, b in v.items()}
    return out


def count_host_syncs(fn, device):
    """Number of synchronising CUDA calls fn() makes (torch sync debug mode)."""
    if device.type != "cuda":
        return None
    import warnings
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(1)
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    return sum("synchroniz" in str(x.message).lower() for x in w)
