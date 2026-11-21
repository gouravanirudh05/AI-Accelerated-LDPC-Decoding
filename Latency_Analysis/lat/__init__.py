"""Component-level latency analysis of BP (common/bp.py) and BP-SGT (SparseGT_BP/sgt).

Importing this package puts SparseGT_BP and the BP-GNN project (common/, final/,
5g_ldpc/, normal_ldpc/) on sys.path, exactly as SparseGT_BP/final/profile_sgt.py does,
so every decoder here is the one used for the published results.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGT_ROOT = os.environ.get("SGT_ROOT", os.path.join(ROOT, "..", "SparseGT_BP"))
if os.path.normpath(SGT_ROOT) not in sys.path:
    sys.path.insert(0, os.path.normpath(SGT_ROOT))

import sgt  # noqa: E402,F401  (registers common/ on sys.path)
