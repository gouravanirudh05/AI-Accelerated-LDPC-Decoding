"""BP + dual-attention Tanner-graph transformer (BP-DAT) hybrid decoder.

Reuses the code objects, BP, training loop, evaluation and profiling of the
BP-GNN project (Nvidia_GNN+BP(lLatest), or $NV_ROOT) and the input features,
RWSE and training specs of BP-PGNN (PooledGNN_BP, or $PGNN_ROOT) unchanged, so
every number is directly comparable with BP-GNN, BP-SGT and BP-PGNN. Importing
this package puts those projects on sys.path and registers the model kind "dat".
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGNN_ROOT = os.path.normpath(os.environ.get("PGNN_ROOT", os.path.join(ROOT, "..", "PooledGNN_BP")))
if PGNN_ROOT not in sys.path:
    sys.path.insert(0, PGNN_ROOT)

import pgnn  # noqa: E402  (puts the BP-GNN project on sys.path, registers "pgnn")
from pgnn import NV_ROOT, SGT_ROOT  # noqa: E402,F401
from common import experiment, report  # noqa: E402

from .dat_decoder import DualAttnBPDecoder  # noqa: E402

experiment.MODEL_CLASSES["dat"] = DualAttnBPDecoder

_style = report._style


def _dat_style(name):
    if name.startswith("DAT"):
        return dict(color="tab:red", linestyle="-." if "OSD" in name else "-", marker="D", markersize=5)
    return _style(name)


report._style = _dat_style
