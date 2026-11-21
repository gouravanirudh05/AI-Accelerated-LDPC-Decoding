"""Timing helpers: interleaved repeated measurements in one process."""

import statistics
import time

import torch

from common.profiling import cuda_graphed, latency
from .regions import Regions, attribute


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, x, reps=20, warmup=3, max_seconds=6.0):
    """Median wall ms per call with sync (common.profiling.latency)."""
    return latency(fn, x, warmup=warmup, reps=reps, max_seconds=max_seconds)["median_ms"]


def interleaved(cases, x, rounds=2, **kw):
    """{name: fn}: every case is timed ``rounds`` times, round robin, so slow drifts
    of the GPU (clocks, temperature) hit all cases alike. Returns
    {name: dict(median_ms, rounds_ms, spread)} where spread = max/min - 1 of the rounds."""
    res = {n: [] for n in cases}
    for _ in range(rounds):
        for n, fn in cases.items():
            try:
                res[n].append(timed(fn, x, **kw))
            except Exception as e:  # noqa: BLE001 (e.g. OOM or capture failure: record and go on)
                res[n].append(float("nan"))
                print(f"      {n}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                if x.device.type == "cuda":
                    torch.cuda.empty_cache()
    out = {}
    for n, v in res.items():
        ok = [t for t in v if t == t]
        out[n] = dict(median_ms=statistics.median(ok) if ok else float("nan"), rounds_ms=v,
                      spread=(max(ok) / min(ok) - 1) if len(ok) > 1 else 0.0)
    return out


def graphed_or_none(fn, x, log=print):
    if x.device.type != "cuda":
        return None
    try:
        return cuda_graphed(fn, x)
    except Exception as e:  # noqa: BLE001
        log(f"      CUDA graph capture failed: {type(e).__name__}: {str(e)[:160]}")
        return None


def sync_breakdown(run, dev, reps=5):
    """run(R) once for warm-up, then ``reps`` times in sync mode; per-call table."""
    R = Regions("sync", dev)
    run(R)
    R.reset()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        run(R)
    sync(dev)
    wall = (time.perf_counter() - t0) * 1e3 / reps
    return dict(wall_ms=wall, regions=R.table(reps))


def profiler_breakdown(run, dev, reps=5, trace=None):
    """run(R) under torch.profiler with record_function regions; per-call table of
    GPU kernel time, kernel launches, host time and host syncs per region."""
    from torch.profiler import ProfilerActivity, profile
    R = Regions("profiler", dev)
    run(R)
    sync(dev)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    t0 = time.perf_counter()
    with profile(activities=acts) as prof:
        for _ in range(reps):
            run(R)
        sync(dev)
    wall = (time.perf_counter() - t0) * 1e3 / reps
    if trace:
        try:
            prof.export_chrome_trace(trace)
        except Exception:  # noqa: BLE001
            pass
    try:
        top = prof.key_averages().table(sort_by="self_cuda_time_total" if dev.type == "cuda"
                                        else "self_cpu_time_total", row_limit=25)
    except Exception:  # noqa: BLE001
        top = ""
    return dict(wall_ms=wall, regions=attribute(prof, reps), top_ops=top)
