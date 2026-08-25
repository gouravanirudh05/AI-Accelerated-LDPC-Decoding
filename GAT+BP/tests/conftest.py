from __future__ import annotations

import sys
from pathlib import Path


GAT_BP_DIR = Path(__file__).resolve().parents[1]
if str(GAT_BP_DIR) not in sys.path:
    sys.path.insert(0, str(GAT_BP_DIR))
