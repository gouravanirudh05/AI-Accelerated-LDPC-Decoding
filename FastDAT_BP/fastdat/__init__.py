"""Latency-optimised BP and BP-DAT for large 5G NR codes.

Nothing in here changes a trained model. ``engine.StagedEngine`` runs the same
BP -> module -> BP schedule as common.bp.staged_decode, with the same early-stop
semantics, but without host synchronisation inside a block of iterations and
with every block captured as one CUDA graph. ``dat_fast.fast_dat_refine`` is
the same DAT module as DualAttention_BP/dat with the scatter operations replaced
by padded gathers (fixed shapes, no atomics). tests/test_equivalence.py checks
both against the original decoders on CPU.

Importing this package puts DualAttention_BP (and through it PooledGNN_BP and
the BP-GNN project), and SparseGT_BP on sys.path.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
for _p in (os.environ.get("DAT_ROOT", os.path.join(REPO, "DualAttention_BP")),
           os.environ.get("SGT_ROOT", os.path.join(REPO, "SparseGT_BP"))):
    _p = os.path.normpath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dat  # noqa: E402,F401  (registers BP-GNN common/ and PooledGNN_BP on sys.path)
