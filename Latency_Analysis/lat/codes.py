"""Codes and models as in the final comparison (SparseGT_BP/final/profile_sgt.py)."""

import glob
import os

import torch

from run_final import code_specs
from sgt.sgt_decoder import SparseGTBPDecoder
from sgt.specs import load_any_code, sgt_spec

# final SGT config (SparseGT_BP/README.md "Winner"): defaults of sgt/specs.py MODEL with k_frac 0.5
FINAL_OVERRIDES = {"k_frac": 0.5}


def load(name, device):
    return load_any_code(name, device)


def prof_ebno(name):
    for grp in ("normal", "5g"):
        for c, _, pe, _ in code_specs(grp, torch.device("cpu")):
            if c.name == name:
                return pe
    raise KeyError(name)


def build_model(code, device, ckpt_root=None, **overrides):
    """Final BP-SGT for ``code``; loads the trained checkpoint when one is found
    under ckpt_root (Kaggle input), otherwise untrained (timing of a fixed-stage
    decode does not depend on the weights; early-stop timing does)."""
    ov = dict(FINAL_OVERRIDES, **overrides)
    kw = sgt_spec(code, ov, quick=False)["kwargs"]
    model = SparseGTBPDecoder(**kw).to(device)
    model.trained_from = None
    model.config_diff = {}
    if ckpt_root:
        hits = glob.glob(os.path.join(ckpt_root, "**", f"*_{code.name}_sgt.pt"), recursive=True)
        if hits:
            # the checkpoint's own kwargs are the trained config (sgt_config.json is not in git)
            state = torch.load(hits[0], map_location=device)
            model = SparseGTBPDecoder(**state["kwargs"]).to(device)
            model.load_state_dict(state["state_dict"])
            model.trained_from = hits[0]
            model.config_diff = {a: (kw.get(a), b) for a, b in state["kwargs"].items() if kw.get(a) != b}
    model.eval()
    return model
