"""Inference latency / throughput and GPU power / energy measurement.

GPU power uses NVML: the board energy counter (nvmlDeviceGetTotalEnergyConsumption,
Volta and newer, incl. T4) when available, otherwise power samples integrated
over time. Idle power is measured first so we can report the *dynamic* energy
attributable to decoding. CPU energy uses Intel RAPL when the container exposes it.
"""

import glob
import os
import statistics
import subprocess
import threading
import time

import torch


class GPUPowerMeter:
    def __init__(self, index=0):
        self.nvml = None
        self.has_energy = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.name = pynvml.nvmlDeviceGetName(self.h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
                self.has_energy = True
            except Exception:
                pass
        except Exception:
            self.name = "unknown"
        self.index = index

    @property
    def available(self):
        return self.nvml is not None

    def power_w(self):
        if self.nvml is not None:
            return self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        out = subprocess.run(["nvidia-smi", f"--id={self.index}", "--query-gpu=power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return float(out.stdout.strip().split("\n")[0])

    def energy_j(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0

    def measure(self, fn, seconds, sample_dt=0.02):
        """Run fn() repeatedly for ~seconds. Returns dict(calls, seconds, energy_j, avg_w, peak_w)."""
        samples, stop = [], threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    samples.append(self.power_w())
                except Exception:
                    pass
                time.sleep(sample_dt)

        th = threading.Thread(target=sampler, daemon=True)
        torch.cuda.synchronize()
        e0 = self.energy_j() if self.has_energy else None
        th.start()
        t0 = time.perf_counter()
        calls = 0
        while time.perf_counter() - t0 < seconds:
            fn()
            calls += 1
            if calls % 4 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        e1 = self.energy_j() if self.has_energy else None
        stop.set()
        th.join()
        avg_sample = statistics.fmean(samples) if samples else float("nan")
        energy = (e1 - e0) if self.has_energy else avg_sample * dt
        return dict(calls=calls, seconds=dt, energy_j=energy, avg_w=energy / dt,
                    peak_w=max(samples) if samples else float("nan"),
                    method="nvml_energy_counter" if self.has_energy else "power_sampling")

    def idle_w(self, seconds=3.0):
        torch.cuda.synchronize()
        return self.measure(lambda: time.sleep(0.01), seconds)["avg_w"]


class RAPL:
    """Package energy from /sys/class/powercap (often unavailable in containers)."""

    def __init__(self):
        self.files = [f for f in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
                      if os.access(f, os.R_OK)]

    @property
    def available(self):
        return bool(self.files)

    def energy_j(self):
        return sum(int(open(f).read()) for f in self.files) / 1e6


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(fn, llr, warmup=5, reps=30, max_seconds=5.0):
    """Wall-clock latency per call (ms) with device synchronisation."""
    dev = llr.device
    for _ in range(warmup):
        fn(llr)
    _sync(dev)
    times = []
    t_start = time.perf_counter()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(llr)
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
        if time.perf_counter() - t_start > max_seconds and len(times) >= 5:
            break
    times.sort()
    return dict(mean_ms=statistics.fmean(times), median_ms=statistics.median(times),
                p95_ms=times[min(len(times) - 1, int(0.95 * len(times)))], reps=len(times))


def cuda_graphed(fn, example):
    """Capture fn(example) into a CUDA graph (removes kernel-launch overhead).

    Only valid for decoders without data-dependent control flow (fixed iterations).
    """
    static_in = example.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn(static_in)

    def run(x):
        static_in.copy_(x)
        graph.replay()
        return static_out
    return run


def _maybe_graph(name, fn, llr, log):
    if "CUDA graph" not in name or llr.device.type != "cuda":
        return fn
    try:
        return cuda_graphed(fn, llr)
    except Exception as e:  # fall back to eager execution
        log(f"    CUDA graph capture failed for {name}: {e}")
        return fn


@torch.no_grad()
def profile_decoders(decoders, code, ebno, batch_sizes, energy_batch, meter=None, idle_w=None,
                     energy_seconds=5.0, cpu_batch_sizes=(1, 64), cpu_decoders=None, log=print):
    """decoders: {name: fn(llr)}. Times each decoder on the GPU at several batch
    sizes, measures GPU energy at ``energy_batch`` and CPU latency/energy."""
    out = {}
    dev = code.device
    torch.manual_seed(7)
    for name, fn in decoders.items():
        rec = dict(gpu={}, cpu={})
        for bs in batch_sizes:
            _, llr = code.sample(bs, ebno)
            f = _maybe_graph(name, fn, llr, log)
            lat = latency(f, llr)
            lat["throughput_cw_s"] = bs / (lat["median_ms"] / 1e3)
            lat["info_mbps"] = lat["throughput_cw_s"] * code.k / 1e6
            rec["gpu"][bs] = lat
            del f
        if meter is not None and dev.type == "cuda":
            _, llr = code.sample(energy_batch, ebno)
            f = _maybe_graph(name, fn, llr, log)
            f(llr)
            e = meter.measure(lambda: f(llr), energy_seconds)
            del f
            frames = e["calls"] * energy_batch
            e["batch"] = energy_batch
            e["energy_per_cw_uj"] = e["energy_j"] / frames * 1e6
            e["energy_per_info_bit_nj"] = e["energy_j"] / (frames * code.k) * 1e9
            if idle_w is not None:
                dyn = max(e["energy_j"] - idle_w * e["seconds"], 0.0)
                e["idle_w"] = idle_w
                e["dynamic_energy_per_cw_uj"] = dyn / frames * 1e6
            rec["gpu_energy"] = e
        out[name] = rec
        g = rec.get("gpu_energy", {})
        log(f"    {name:28s} | B=1 {rec['gpu'][batch_sizes[0]]['median_ms']:7.3f} ms | "
            f"B={batch_sizes[-1]} {rec['gpu'][batch_sizes[-1]]['throughput_cw_s']:10.0f} cw/s | "
            f"{g.get('avg_w', float('nan')):6.1f} W | {g.get('energy_per_cw_uj', float('nan')):8.2f} uJ/cw")
    if cpu_decoders:
        rapl = RAPL()
        for name, (fn, cpu_code) in cpu_decoders.items():
            rec = out.setdefault(name, dict(gpu={}, cpu={}))
            for bs in cpu_batch_sizes:
                _, llr = cpu_code.sample(bs, ebno)
                e0 = rapl.energy_j() if rapl.available else None
                lat = latency(fn, llr, warmup=2, reps=20, max_seconds=4.0)
                if e0 is not None:
                    lat["rapl_energy_per_cw_uj"] = (rapl.energy_j() - e0) / (lat["reps"] * bs) * 1e6
                lat["throughput_cw_s"] = bs / (lat["median_ms"] / 1e3)
                rec["cpu"][bs] = lat
            log(f"    [cpu] {name:22s} | " + " | ".join(
                f"B={bs} {rec['cpu'][bs]['median_ms']:.2f} ms" for bs in cpu_batch_sizes))
    return out
