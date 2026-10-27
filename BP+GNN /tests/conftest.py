from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def small_h() -> np.ndarray:
    """A tiny irregular parity-check matrix: 4 checks, 8 variables, nnz=16."""
    return np.array(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 1],
        ],
        dtype=np.int64,
    )


@pytest.fixture(scope="session")
def ldpc49_code():
    """The real LDPC(49, 24) code used by every baseline in this repo."""
    from ldpc_corrector._compat import Get_Generator_and_Parity

    class _Code:
        n, k, code_type = 49, 24, "LDPC"

    generator, parity_check = Get_Generator_and_Parity(_Code())
    return np.asarray(generator), np.asarray(parity_check)
