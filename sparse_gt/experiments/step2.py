"""Step 2: validate BP at scale and measure top-k coverage on every code.

Three sections, each saved as soon as it finishes (so a partial run still
leaves usable results in ``--out``):

1. ``curves``   BP(100) BER/FER curve for every code. WiMAX-576 and
                10G-Ethernet-2048 are compared against AFF3CT's published
                curves (same decoder, same stopping rule, same codeword).
2. ``timing``   GPU inference cost per code and batch size: BP at fixed T
                (latency at batch 1, throughput at large batch), top-k
                selection, and early-stopped BP at the FER~1e-2 SNR. The
                currency of the later equal-latency comparison.
3. ``coverage`` top-k coverage of BP(T)'s errors for T in T_VALUES at the
                SNRs where BP(100) has FER ~ 1e-1, 1e-2, 1e-3 (picked from
                section 1's curve), split into recoverable / stuck frames.

Run::

    python -m sparse_gt.experiments.step2 --device cuda --out results/step2
    python -m sparse_gt.experiments.step2 --smoke          # seconds, CPU
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from ..codes import available_codes, load_code
from ..coverage import coverage_sweep
from ..simulate import simulate_bp
from ..timing import inference_timing

# Eb/N0 grid (start, stop, step) for each code's BP(100) curve. WiMAX and
# Ethernet use the AFF3CT reference grids so points compare one-to-one.
CURVE_GRID = {
    "MACKAY_N204_K102": (1.0, 5.0, 0.25),
    "WIMAX_N576_K288": (0.5, 3.5, 0.25),
    "WIFI_N648_K540": (2.5, 6.0, 0.25),
    "MACKAY_N816_K410": (0.5, 4.0, 0.25),
    "MACKAY_N1008_K504": (0.5, 4.0, 0.25),
    "PEG_N1008_K504": (0.5, 4.0, 0.25),
    "ETH10G_N2048_K1723": (2.4, 4.4, 0.2),
}
REFERENCES = {
    "WIMAX_N576_K288": "LDPC_N576_K288_flooding_SPA_i100_WiMAX.txt",
    "ETH10G_N2048_K1723": "LDPC_N2048_K1723_flooding_SPA_i100_eth_10Gbps.txt",
}
REFERENCE_ITERATIONS = 100
T_VALUES = (2, 3, 5, 8, 10, 15, 20)
COVERAGE_FER_TARGETS = (1e-1, 1e-2, 1e-3)
# Keep B * E around this many elements per BP batch (GPU memory / speed).
ELEMENTS_PER_BATCH = 32_000_000


def batch_for(code, elements: int = ELEMENTS_PER_BATCH, cap: int = 16384) -> int:
    return int(min(cap, max(256, elements // code.num_edges)))


def pick_snrs(curve: list[dict], targets=COVERAGE_FER_TARGETS) -> list[float]:
    """Eb/N0 where the BP(100) FER curve crosses each target (log-linear interp)."""
    pts = [(r["ebn0"], r["fer"]) for r in curve if r["fer"] > 0]
    snrs = []
    for target in targets:
        for (s0, f0), (s1, f1) in zip(pts, pts[1:]):
            if f0 >= target >= f1 and f0 > f1:
                frac = (math.log(f0) - math.log(target)) / (math.log(f0) - math.log(f1))
                snrs.append(round((s0 + frac * (s1 - s0)) * 20) / 20)  # 0.05 dB grid
                break
    return sorted(set(snrs))


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))


def run_curves(codes, args, out: Path, deadline: float) -> dict:
    path = out / "curves.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in codes:
        if name in results:
            continue
        if time.time() > deadline:
            print(f"[curves] deadline reached, skipping {name}")
            break
        code = load_code(name)
        start, stop, step = CURVE_GRID[name]
        grid = np.round(np.arange(start, stop + 1e-9, step), 3).tolist()
        if args.smoke:
            grid = grid[::4][:3]
        validated = name in REFERENCES
        rows = []
        print(f"[curves] {code}")
        for snr in grid:
            row = simulate_bp(
                code, [snr], REFERENCE_ITERATIONS,
                batch=batch_for(code) if not args.smoke else 256,
                min_frame_errors=(200 if validated else 100) if not args.smoke else 10,
                max_frames=(20_000_000 if validated else 2_000_000) if not args.smoke else 1024,
                device=args.device, seed=zlib.crc32(f"{name}/{snr}".encode()),
            )[0]
            rows.append(row)
            # Past this point the remaining SNRs are too costly to resolve.
            if row["frame_errors"] < 20 or time.time() > deadline:
                break
        results[name] = {"n": code.n, "k": code.k, "edges": code.num_edges,
                         "iterations": REFERENCE_ITERATIONS, "rows": rows,
                         "reference": REFERENCES.get(name)}
        save_json(path, results)
    return results


# Operating point (Eb/N0 where BP(100) FER ~ 1e-2) used for early-stop timing
# when no curve from this run is available. Rough values; only the timing
# depends on them.
DEFAULT_OPERATING_SNR = {
    "MACKAY_N204_K102": 3.0, "WIMAX_N576_K288": 2.0, "WIFI_N648_K540": 4.5,
    "MACKAY_N816_K410": 2.25, "MACKAY_N1008_K504": 2.0, "PEG_N1008_K504": 2.0,
    "ETH10G_N2048_K1723": 3.6,
}


def run_timing(codes, curves, args, out: Path) -> dict:
    results = {}
    for name in codes:
        code = load_code(name)
        picked = pick_snrs(curves[name]["rows"], targets=(1e-2,)) if name in curves else []
        snr = picked[0] if picked else DEFAULT_OPERATING_SNR.get(name)
        n = code.n
        k_values = sorted({max(1, round(f * n)) for f in (0.01, 0.02, 0.05, 0.10)} | {32, 64})
        rows = inference_timing(
            code, args.device,
            batches=(1, 64, 1024, 8192) if not args.smoke else (1, 64),
            T_values=(*T_VALUES, 50, REFERENCE_ITERATIONS) if not args.smoke else (2, 5),
            k_values=k_values,
            early_stop_snr=snr,
            early_stop_iterations=REFERENCE_ITERATIONS if not args.smoke else 20,
            repeats=20 if not args.smoke else 2,
        )
        results[name] = {"n": n, "edges": code.num_edges, "operating_ebn0": snr,
                         "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
                         "rows": rows}
        for r in rows:
            if r["kind"] == "bp_fixed" and r["T"] in (5, 20, REFERENCE_ITERATIONS):
                print(f"[timing] {name:20s} B={r['batch']:5d} BP({r['T']:3d}) {r['ms']:8.3f} ms  "
                      f"{r['frames_per_s']:12.0f} frames/s")
            elif r["kind"] == "bp_early_stop":
                print(f"[timing] {name:20s} B={r['batch']:5d} BP early-stop @ {snr} dB {r['ms']:8.3f} ms "
                      f"(mean it {r['mean_iterations']:.1f}, max {r['max_iterations_used']})")
        save_json(out / "timing.json", results)
    return results


def run_coverage(codes, curves, args, out: Path, deadline: float) -> dict:
    path = out / "coverage.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in codes:
        if name in results or name not in curves:
            continue
        if time.time() > deadline:
            print(f"[coverage] deadline reached, skipping {name}")
            break
        code = load_code(name)
        snrs = pick_snrs(curves[name]["rows"]) if not args.smoke else [curves[name]["rows"][0]["ebn0"]]
        print(f"[coverage] {code} at Eb/N0 {snrs}")
        if not snrs:
            continue
        summaries, hists = coverage_sweep(
            code, snrs,
            T_values=T_VALUES if not args.smoke else (2, 5),
            reference_iterations=REFERENCE_ITERATIONS if not args.smoke else 20,
            batch=batch_for(code, ELEMENTS_PER_BATCH // 2, cap=4096) if not args.smoke else 256,
            target_failed=1000 if not args.smoke else 20,
            max_frames=1_000_000 if not args.smoke else 1024,
            device=args.device,
        )
        results[name] = {"n": code.n, "k": code.k, "snrs": snrs, "summaries": summaries}
        np.savez_compressed(out / f"coverage_hist_{name}.npz", **hists)
        save_json(path, results)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/step2")
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--sections", nargs="*", default=["curves", "timing", "coverage"])
    parser.add_argument("--hours", type=float, default=11.0,
                        help="Wall-clock budget; remaining work is skipped after it.")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to check the pipeline.")
    args = parser.parse_args(argv)

    codes = args.codes or (["MACKAY_N204_K102", "WIMAX_N576_K288"] if args.smoke else available_codes())
    codes = sorted(codes, key=lambda c: load_code(c).num_edges)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    print(f"device={args.device} codes={codes} sections={args.sections}")
    if args.device.startswith("cuda"):
        print(torch.cuda.get_device_name(0))

    t0 = time.time()
    curves = {}
    if "curves" in args.sections:
        curves = run_curves(codes, args, out, deadline)
    elif (out / "curves.json").exists():
        curves = json.loads((out / "curves.json").read_text())
    if "timing" in args.sections:
        run_timing(codes, curves, args, out)
    if "coverage" in args.sections:
        run_coverage(codes, curves, args, out, deadline)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
