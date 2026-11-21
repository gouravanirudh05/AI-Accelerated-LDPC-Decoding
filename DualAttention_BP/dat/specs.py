"""Model and training specs of BP-DAT for one code.

Codes, Eb/N0 bands and the training recipe (including 160 min instead of 80 on
codes with n >= 512) are BP-PGNN's (PooledGNN_BP/pgnn/specs.py), so BP-DAT is
directly comparable with BP-PGNN, and with BP-GNN / BP-SGT up to the training
time on the four long codes (the merge script marks those rows).
"""

import copy

from pgnn.specs import BANDS, TRAIN, load_any_code, train_minutes  # noqa: F401
from run_final import model_spec

# d / d_e / hidden of BP-GNN and BP-PGNN; 2 dual-attention layers (BP-GNN: 2 GNN layers)
MODEL = dict(d=16, d_e=8, hidden=48, heads=2, layers=2, nb_hidden=32, cross="attn", self_attn=True,
             rw_steps=8, dropout=0.1, stages=4, bp_iters=10, base="prior")


def label(kw):
    return f"DAT ({kw['stages'] + 1}x{kw['bp_iters']} BP it, {kw['cross']})"


def dat_spec(code, overrides=None, quick=False, tag="dat", name=None):
    """Model spec for common.experiment.run_experiment. ``overrides`` mixes model and train keys."""
    kw, tr = copy.deepcopy(MODEL), copy.deepcopy(TRAIN)
    tr["max_minutes"] = train_minutes(code)
    for k, v in (overrides or {}).items():
        (kw if k in MODEL else tr)[k] = v
    t = dict(kind="dat", tag=tag, label=name or label(kw), kwargs=kw, train=tr)
    return model_spec(t, code, BANDS[code.name], quick)
