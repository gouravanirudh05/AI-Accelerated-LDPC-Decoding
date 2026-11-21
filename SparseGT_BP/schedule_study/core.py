"""Shared core of the BP-SGT schedule study (tasks 1, 2, 4 and 5).

* ``parse_schedule``: "5x10" -> [10, 10, 10, 10, 10] (runs x iterations per run),
  "25-25" or "20-10-10-5-5" -> explicit iterations of every BP run.
* ``traced_decode``: the interleaved pipeline of common.bp.staged_decode
  (BP(T_0) -> SGT -> BP(T_1) -> SGT -> ... -> BP(T_R-1)), but with a different
  number of BP iterations per run, and recording for every frame *where* it
  stopped: the BP run (pipeline stage) whose syndrome check succeeded, the
  iteration inside that run, the total BP iterations, and the number of SGT
  calls. A plain BP(T) decoder is the schedule [T] with no refinement, so BP
  baselines and BP-SGT go through the same code path.
* ``simulate_traced``: Monte Carlo over Eb/N0 points with histograms of the
  stopping points (by run, by total iteration, converged-correct vs
  converged-to-a-wrong-codeword vs never converged).
* ``load_sgt``: the trained final BP-SGT model of a code (checkpoint of
  SparseGT_BP/final/run_sgt_final.py; the kwargs are stored in the checkpoint).

Early stopping is the same as everywhere else in the project: a frame leaves the
batch as soon as the hard decision of the current BP iteration satisfies every
parity check, so an SGT stage only runs on frames the previous BP run failed.
"""

import glob
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SGT_ROOT = os.path.dirname(HERE)
if SGT_ROOT not in sys.path:
    sys.path.insert(0, SGT_ROOT)
import sgt  # noqa: E402,F401  (puts the BP-GNN project on sys.path, registers "sgt")
from common.bp import bp_stage  # noqa: E402
from sgt.sgt_decoder import SparseGTBPDecoder  # noqa: E402
from sgt.specs import BANDS, load_any_code  # noqa: E402

CKPT_DIRS = [os.path.join(SGT_ROOT, "results")]


# ----------------------------------------------------------------------------- schedules
def parse_schedule(s):
    s = str(s).strip()
    if "x" in s:
        runs, iters = s.split("x")
        return [int(iters)] * int(runs)
    return [int(t) for t in s.split("-")]


def schedule_str(sched):
    if len(set(sched)) == 1:
        return f"{len(sched)}x{sched[0]}"
    return "-".join(str(t) for t in sched)


# ----------------------------------------------------------------------------- decoding
@torch.no_grad()
def traced_decode(llr, g, schedule, refine=None):
    """Early-stopping decode of ``llr`` with BP runs of ``schedule[r]`` iterations
    and ``refine`` (e.g. SparseGTBPDecoder.refine_chunked) between runs.

    Returns (final LLR [B, n], trace) with trace tensors of shape [B]:
      used      total BP iterations performed on the frame
      stop_run  index of the BP run in which the syndrome was satisfied, -1 = never
      stop_it   iteration inside that run (1-based), 0 if never
      sgt_calls number of refinement (SGT) stages the frame went through
    """
    B, dev = llr.shape[0], llr.device
    out = llr.clone()
    used = torch.zeros(B, dtype=torch.long, device=dev)
    stop_run = torch.full((B,), -1, dtype=torch.long, device=dev)
    stop_it = torch.zeros(B, dtype=torch.long, device=dev)
    calls = torch.zeros(B, dtype=torch.long, device=dev)
    active = torch.arange(B, device=dev)
    x, prior = llr, llr
    last = len(schedule) - 1
    for r, T in enumerate(schedule):
        L, v2c, c2v, ok, u = bp_stage(prior, g, T, early_stop=True)
        out[active] = L
        used[active] += u
        stop_run[active[ok]] = r
        stop_it[active[ok]] = u[ok]
        keep = ~ok
        if r == last or refine is None or not keep.any():
            break
        active, x, prior, L, v2c, c2v = (a[keep] for a in (active, x, prior, L, v2c, c2v))
        calls[active] += 1
        prior = refine(x, prior, L, v2c, c2v, g)
    return out, dict(used=used, stop_run=stop_run, stop_it=stop_it, sgt_calls=calls)


def make_decoder(schedule, model=None):
    """fn(llr, g) -> (LLR, trace); model=None gives plain BP with sum(schedule) iterations."""
    if model is None:
        sched = [sum(schedule)]
        return lambda llr, g: traced_decode(llr, g, sched, None)
    return lambda llr, g: traced_decode(llr, g, schedule, model.refine_chunked)


# ----------------------------------------------------------------------------- Monte Carlo
@torch.no_grad()
def simulate_traced(dec, code, schedule, ebno_list, batch, max_frames, min_frame_errors,
                    max_seconds=None, seed=0, log=print, name=""):
    """Monte Carlo of ``dec`` (from make_decoder) with stopping-point statistics.

    Per Eb/N0 point (all counts are frames):
      run_ok[r]     syndrome satisfied in BP run r and the codeword is the transmitted one
      run_wrong[r]  syndrome satisfied in BP run r but a wrong codeword (undetected error)
      never         syndrome never satisfied (detected failure, used = full budget)
      iter_hist[t]  converged frames that stopped after t total BP iterations (t = 1..budget)
      calls_hist[c] frames that went through c SGT stages
      iters_sum / iters_sq / calls_sum for means and standard deviations
    """
    torch.manual_seed(seed)
    runs = len(schedule)
    budget = sum(schedule)
    g = code.graph
    res = []
    for ebno in ebno_list:
        st = dict(ebno=float(ebno), frames=0, frame_errors=0, bit_errors=0,
                  run_ok=[0] * runs, run_wrong=[0] * runs, never=0, never_correct=0,
                  iter_hist=[0] * (budget + 1), calls_hist=[0] * runs,
                  iters_sum=0, iters_sq=0, calls_sum=0)
        t0 = time.time()
        while st["frames"] < max_frames and st["frame_errors"] < min_frame_errors:
            c, llr = code.sample(batch, ebno)
            L, tr = dec(llr, g)
            err = (L < 0).float() != c
            ferr = err.any(1)
            st["frames"] += batch
            st["frame_errors"] += int(ferr.sum())
            st["bit_errors"] += int(err[:, code.info_pos].sum())
            used, sr, calls = tr["used"], tr["stop_run"], tr["sgt_calls"]
            conv = sr >= 0
            st["never"] += int((~conv).sum())
            st["never_correct"] += int((~conv & ~ferr).sum())
            st["run_ok"] = [a + b for a, b in zip(st["run_ok"], torch.bincount(sr[conv & ~ferr], minlength=runs).tolist())]
            st["run_wrong"] = [a + b for a, b in zip(st["run_wrong"], torch.bincount(sr[conv & ferr], minlength=runs).tolist())]
            st["iter_hist"] = [a + b for a, b in zip(st["iter_hist"], torch.bincount(used[conv], minlength=budget + 1).tolist())]
            st["calls_hist"] = [a + b for a, b in zip(st["calls_hist"], torch.bincount(calls, minlength=runs).tolist())]
            st["iters_sum"] += int(used.sum())
            st["iters_sq"] += int((used * used).sum())
            st["calls_sum"] += int(calls.sum())
            if max_seconds and time.time() - t0 > max_seconds:
                break
        nf = st["frames"]
        mean = st["iters_sum"] / nf
        st.update(bler=st["frame_errors"] / nf, ber=st["bit_errors"] / (nf * code.k),
                  avg_iters=mean, std_iters=max(0.0, st["iters_sq"] / nf - mean * mean) ** 0.5,
                  avg_sgt_calls=st["calls_sum"] / nf, seconds=round(time.time() - t0, 2))
        res.append(st)
        log(f"  {name:28s} {ebno:5.2f} dB  BLER {st['bler']:.2e} ({st['frame_errors']}/{nf})  "
            f"avg it {mean:6.2f}  avg SGT {st['avg_sgt_calls']:.3f}  "
            f"stop/run {[round((a + b) / nf, 4) for a, b in zip(st['run_ok'], st['run_wrong'])]} "
            f"never {st['never'] / nf:.2e}")
    return res


# ----------------------------------------------------------------------------- models / codes
def find_checkpoint(code_name, roots=()):
    pats = [os.path.join(r, "**", f"*_{code_name}_sgt.pt") for r in list(roots) + CKPT_DIRS]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no final BP-SGT checkpoint for {code_name} (searched {pats})")


def load_sgt(code_name, device, roots=(), path=None):
    path = path or find_checkpoint(code_name, roots)
    state = torch.load(path, map_location=device)
    model = SparseGTBPDecoder(**state["kwargs"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, path, state["kwargs"]


def default_ebno(code_name):
    """4 to 7 Eb/N0 points spanning the code's training band (final/code_bands.json),
    on a 0.25 dB grid."""
    lo, hi = BANDS[code_name]["train_ebno"]
    step = 0.25 * math.ceil((hi - lo) / 6 / 0.25 - 1e-9)
    pts, e = [], lo
    while e <= hi + 1e-9:
        pts.append(round(e, 2))
        e += step
    return pts


def frame_budget(n):
    return 3_000_000 if n <= 204 else 2_000_000 if n <= 408 else 1_000_000 if n <= 704 else 500_000


ALL_CODES = ["mackay_96_48", "ccsds_128_64", "mackay_204_102", "mackay_408_204", "wimax_576_288",
             "5G_BG2_k64_n128", "5G_BG2_k128_n256", "5G_BG2_k256_n512", "5G_BG1_k528_n704",
             "5G_BG1_k1056_n1408"]


def results_dir(task, out=None):
    out = out or (os.path.join("/kaggle/working/results/schedule_study", task) if os.path.isdir("/kaggle/working")
                  else os.path.join(HERE, "out", task))
    os.makedirs(out, exist_ok=True)
    return out


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def device_info(device):
    return (f"device {device} | torch {torch.__version__}"
            + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))


__all__ = ["parse_schedule", "schedule_str", "traced_decode", "make_decoder", "simulate_traced",
           "load_sgt", "load_any_code", "default_ebno", "frame_budget", "ALL_CODES", "results_dir",
           "Logger", "device_info", "BANDS"]
