"""BP + sparse graph transformer (BP-SGT) hybrid decoder.

Reuses the code objects, BP, training loop, evaluation and profiling of the
BP-GNN project (Nvidia_GNN+BP(lLatest), or $NV_ROOT) unchanged, so every
number is directly comparable with the BP-GNN results there. Importing this
package puts that project on sys.path and registers the model kind "sgt".
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV_ROOT = os.environ.get("NV_ROOT", os.path.join(ROOT, "..", "Nvidia_GNN+BP(lLatest)"))
for _p in ("final", "5g_ldpc", "normal_ldpc", ""):
    _p = os.path.normpath(os.path.join(NV_ROOT, _p))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import experiment, report  # noqa: E402

from .sgt_decoder import SparseGTBPDecoder  # noqa: E402

experiment.MODEL_CLASSES["sgt"] = SparseGTBPDecoder

_style = report._style


def _sgt_style(name):
    if name.startswith("SGT"):
        return dict(color="tab:purple", linestyle="-." if "OSD" in name else "-", marker="*", markersize=6)
    return _style(name)


report._style = _sgt_style
