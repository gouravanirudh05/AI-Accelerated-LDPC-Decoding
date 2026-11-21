"""Model and training specs of BP-PGNN for one code.

Codes, Eb/N0 bands (final/code_bands.json of the BP-GNN project) and the
training recipe are those of BP-GNN and BP-SGT, so the three hybrids differ only
in the module between the BP runs, except for the training time: codes with
n >= LONG_N transmitted bits get LONG_MINUTES instead of 80 (Gourav's choice,
2026-11-21). BP-GNN and BP-SGT were trained for 80 min on every code, so on the
long codes the comparison is *not* like-for-like in training budget; the merge
script and README state this next to every such number.
"""

import copy
import json
import math
import os
import re

from . import NV_ROOT
from nr_ldpc import nr_code
from run_final import model_spec
from run_normal_ldpc import load_code

with open(os.path.join(NV_ROOT, "final", "code_bands.json")) as _f:
    BANDS = json.load(_f)

LONG_N, LONG_MINUTES = 512, 160

# training recipe of the tuned BP-GNN / BP-SGT (final/learned_configs.json)
TRAIN = dict(steps=10_000_000, batch=384, lr=1e-3, lr_min=1e-5, lam_syn=0.1, eval_every=500,
             val_frames=16384, grad_clip=1.0, max_minutes=80, schedule="plateau", patience=6,
             band_shift=[-1.5, -1.0], hard_frac=0.75, mining_bp_iters=10, pool_bins=4, pool_new=64,
             pool_init=1024, pool_size=20000, pool_refresh_every=10, mining_chunk=32768,
             mining_max_frames=262144, val_mode="hard", val_hard_bins=3, val_hard_frames=1000,
             val_hard_max_frames=8_000_000)
# d / d_e / hidden of BP-GNN; 1 fine layer before and 1 after the pooled graph (BP-GNN: 2 fine layers)
MODEL = dict(d=16, d_e=8, hidden=48, pre_layers=1, post_layers=1, coarse_layers=2, pool="diff",
             assign="auto", cluster_frac=1 / 16, topk_v=0.125, topk_c=0.25, readout=True, rw_steps=8,
             dropout=0.1, stages=4, bp_iters=10, base="prior")


def load_any_code(name, device):
    m = re.fullmatch(r"5G_(BG[12])_k(\d+)_n(\d+)", name)
    if m:
        return nr_code(int(m[2]), int(m[3]), bg=m[1].lower(), device=device)
    return load_code(name, device)


def model_kwargs(code, kw):
    """Resolve "auto" assignment and cluster counts for one code."""
    kw = dict(kw)
    frac = kw.pop("cluster_frac")
    if kw["assign"] == "auto":
        kw["assign"] = "proto" if "bg" in code.meta else "learned"
    g = code.graph
    kw.setdefault("k_v", max(4, math.ceil(frac * g.n)))
    kw.setdefault("k_c", max(2, math.ceil(frac * g.m)))
    return kw


def label(kw):
    return f"PGNN ({kw['stages'] + 1}x{kw['bp_iters']} BP it, {kw['pool']})"


def train_minutes(code):
    return LONG_MINUTES if code.n_tx >= LONG_N else TRAIN["max_minutes"]


def pgnn_spec(code, overrides=None, quick=False, tag="pgnn", name=None):
    """Model spec for common.experiment.run_experiment. ``overrides`` mixes model and train keys."""
    kw, tr = copy.deepcopy(MODEL), copy.deepcopy(TRAIN)
    tr["max_minutes"] = train_minutes(code)
    for k, v in (overrides or {}).items():
        (kw if k in MODEL or k in ("k_v", "k_c") else tr)[k] = v
    kw = model_kwargs(code, kw)
    t = dict(kind="pgnn", tag=tag, label=name or label(kw), kwargs=kw, train=tr)
    return model_spec(t, code, BANDS[code.name], quick)
