"""Bridge to the existing ``GAT+BP`` modules.

``GAT+BP`` contains a ``+``, so it is not a valid Python package name and
cannot be imported normally.  Rather than renaming it -- which would break the
``comparison/`` scripts and the Colab notebooks -- this module loads the proven
BP decoder, Tanner graph, and code utilities directly from their files.

Existing Phase 0-2 scripts keep importing ``gat_bp`` and ``Codes`` the old way;
this shim is additive and does not modify either file's behaviour.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
# ``LDPC_GAT_BP_DIR`` points the shim at a copy of ``GAT+BP`` kept elsewhere,
# e.g. one extracted with ``git archive HEAD GAT+BP | tar -x -C <dir>``.
GAT_BP_DIR = Path(os.environ.get("LDPC_GAT_BP_DIR", REPO_ROOT / "GAT+BP"))


def _load(module_name: str, file_name: str) -> ModuleType:
    """Import ``file_name`` from ``GAT+BP`` under ``module_name``."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = GAT_BP_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Expected the Phase 0-2 module at {path}.")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module can import itself if it ever needs to.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


# ``Codes`` first: it is dependency-free and ``gat_bp`` does not import it.
# ``train_gat_bp`` does ``from Codes import ...``, so register the canonical
# name too and let both point at the same module object.
Codes = _load("Codes", "Codes.py")
sys.modules.setdefault("gat_bp_codes", Codes)
gat_bp = _load("gat_bp", "gat_bp.py")
# Importing the trainer is safe: its CLI is guarded by ``__main__``.
train_gat_bp = _load("train_gat_bp", "train_gat_bp.py")

# Re-export the pieces the corrector package builds on.
LegacyTannerGraph = gat_bp.TannerGraph
SumProductBP = gat_bp.SumProductBP
HybridBPGAT = gat_bp.HybridBPGAT
BidirectionalEdgeGATv2 = gat_bp.BidirectionalEdgeGATv2
DirectedEdgeGATv2 = gat_bp.DirectedEdgeGATv2
hard_decode = gat_bp.hard_decode
bce_from_llr = gat_bp.bce_from_llr

# Channel simulation is reused verbatim so new results stay directly
# comparable with the Phase 0-2 / Aug-26 baseline runs.
sample_channel_batch = train_gat_bp.sample_channel_batch
ebn0_to_sigma = train_gat_bp.ebn0_to_sigma
error_counts = train_gat_bp.error_counts
syndrome_error_count = train_gat_bp.syndrome_error_count
set_seed = train_gat_bp.set_seed

Get_Generator_and_Parity = Codes.Get_Generator_and_Parity
Read_pc_matrix_alist = Codes.Read_pc_matrixrix_alist
EbN0_to_std = Codes.EbN0_to_std

__all__ = [
    "REPO_ROOT",
    "GAT_BP_DIR",
    "Codes",
    "gat_bp",
    "LegacyTannerGraph",
    "SumProductBP",
    "HybridBPGAT",
    "BidirectionalEdgeGATv2",
    "DirectedEdgeGATv2",
    "hard_decode",
    "bce_from_llr",
    "train_gat_bp",
    "sample_channel_batch",
    "ebn0_to_sigma",
    "error_counts",
    "syndrome_error_count",
    "set_seed",
    "Get_Generator_and_Parity",
    "Read_pc_matrix_alist",
    "EbN0_to_std",
]
