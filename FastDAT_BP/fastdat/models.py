"""The decoders compared in the benchmark, with their trained checkpoints when found.

Checkpoints are searched under ``roots`` (default: the repository's committed
results, plus /kaggle/input on Kaggle):
  BP-GNN  final_learned_*_<code>_bpgnn.pt   (Nvidia_GNN+BP(lLatest)/results, committed)
  BP-SGT  final_sgt_*_<code>_sgt.pt         (SparseGT_BP/results, committed)
  BP-DAT  final_dat_*_<code>_dat*.pt        (DualAttention_BP/results, after its Kaggle runs)
A model without a checkpoint is built untrained (``trained_from = None``). For
DAT that is plain BP with restarts (zero head), unless ``randomise_head``.
"""

import glob
import os

import torch

from common.bp_gnn_decoder import BPGNNDecoder
from dat import DualAttnBPDecoder
from dat.specs import dat_spec

from . import REPO

ROOTS = [os.path.join(REPO, "Nvidia_GNN+BP(lLatest)", "results"), os.path.join(REPO, "SparseGT_BP", "results"),
         os.path.join(REPO, "DualAttention_BP", "results"), "/kaggle/input"]


def find(pattern, roots=None):
    for r in roots or ROOTS:
        if os.path.isdir(r):
            hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
            if hits:
                return hits[0]
    return None


def _load(cls, path, device):
    state = torch.load(path, map_location=device)
    m = cls(**state["kwargs"]).to(device)
    m.load_state_dict(state["state_dict"])
    m.trained_from = path
    return m.eval()


def build_dat(code, device, roots=None, randomise_head=False, **overrides):
    path = None if overrides else find(f"final_dat_*_{code.name}_dat.pt", roots)
    if path:
        return _load(DualAttnBPDecoder, path, device)
    torch.manual_seed(0)
    m = DualAttnBPDecoder(**dat_spec(code, overrides)["kwargs"]).to(device).eval()
    if randomise_head:
        with torch.no_grad():
            m.head[-1].weight.normal_(0, 0.3)
            m.head[-1].bias.normal_(0, 0.3)
    m.trained_from = None
    return m


def build_bpgnn(code, device, roots=None):
    path = find(f"final_learned_*_{code.name}_bpgnn.pt", roots)
    return _load(BPGNNDecoder, path, device) if path else None


def build_sgt(code, device, roots=None):
    path = find(f"final_sgt_*_{code.name}_sgt.pt", roots)
    if not path:
        return None
    from sgt.sgt_decoder import SparseGTBPDecoder
    return _load(SparseGTBPDecoder, path, device)
