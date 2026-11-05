"""Codes, Eb/N0 bands and model specs shared by tuning and the final runs.

Codes, bands (final/code_bands.json) and the training recipe are those of the
BP-GNN project's final comparison, so BP-SGT and BP-GNN differ only in the
module between the BP runs.
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

# training recipe of the tuned BP-GNN (final/learned_configs.json), minus its time budget
TRAIN = dict(steps=10_000_000, batch=384, lr=1e-3, lr_min=1e-5, lam_syn=0.1, eval_every=500,
             val_frames=16384, grad_clip=1.0, max_minutes=80, schedule="plateau", patience=6,
             band_shift=[-1.5, -1.0], hard_frac=0.75, mining_bp_iters=10, pool_bins=4, pool_new=64,
             pool_init=1024, pool_size=20000, pool_refresh_every=10, mining_chunk=32768,
             mining_max_frames=262144, val_mode="hard", val_hard_bins=3, val_hard_frames=1000,
             val_hard_max_frames=8_000_000)
MODEL = dict(d=64, heads=4, hidden=128, gt_layers=2, k_frac=0.1, k_min=8, stages=4, bp_iters=10,
             attn="global", select="llr", fix="residual", base="prior", struct_bias=True,
             warm=False)


def load_any_code(name, device):
    m = re.fullmatch(r"5G_(BG[12])_k(\d+)_n(\d+)", name)
    if m:
        return nr_code(int(m[2]), int(m[3]), bg=m[1].lower(), device=device)
    return load_code(name, device)


def label(cfg):
    return (f"SGT ({cfg['stages'] + 1}x{cfg['bp_iters']} BP it, k={cfg['k_frac']:g}n, "
            f"L{cfg['gt_layers']} d{cfg['d']} {cfg['attn']})")


def sgt_spec(code, overrides=None, quick=False, tag="sgt", name=None):
    """Model spec for run_experiment. ``overrides`` mixes model kwargs and train keys."""
    kw, tr = copy.deepcopy(MODEL), copy.deepcopy(TRAIN)
    for k, v in (overrides or {}).items():
        (kw if k in MODEL else tr)[k] = v
    t = dict(kind="sgt", tag=tag, label=name or label(kw), kwargs=kw, train=tr)
    t = model_spec(t, code, BANDS[code.name], quick)
    # T4 memory: the attention bias is B * k^2 * heads per stage; cap it for large k
    k = min(code.graph.n, max(kw["k_min"], math.ceil(kw["k_frac"] * code.graph.n)))
    cap = max(32, int(1.2e8 // (k * k * kw["heads"])) // 32 * 32)
    t["train"]["batch"] = min(t["train"]["batch"], cap)
    return t
