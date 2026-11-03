import json

import numpy as np
import pytest
import torch

from sparse_gt.channel import awgn_llr, ebn0_to_sigma
from sparse_gt.codes import CODES_DIR, available_codes, gf2_rank, load_code, read_alist

MANIFEST = {c["file"]: c for c in json.loads((CODES_DIR / "manifest.json").read_text())["codes"]}


def test_all_manifest_codes_are_available():
    assert sorted(f.removesuffix(".alist") for f in MANIFEST) == available_codes()


@pytest.mark.parametrize("name", available_codes())
def test_code_matches_manifest(name):
    code = load_code(name)
    entry = MANIFEST[f"{name}.alist"]
    assert (code.n, code.m, code.k) == (entry["n"], entry["m"], entry["k"])
    assert code.num_edges == entry["edges"]
    edge_vn, edge_cn = code.edges()
    assert edge_vn.numel() == code.num_edges
    assert torch.all(edge_cn[1:] >= edge_cn[:-1]), "edges must be grouped by check"
    # Every edge is a nonzero of H.
    assert code.H[edge_cn.numpy(), edge_vn.numpy()].all()


def test_name_does_not_override_true_dimension():
    code = load_code("MACKAY_N816_K410")
    assert code.k == 410 and abs(code.rate - 410 / 816) < 1e-12


def test_gf2_rank_small():
    H = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)  # row3 = row1 + row2
    assert gf2_rank(H) == 2


def test_read_alist_rejects_inconsistent_file(tmp_path):
    bad = tmp_path / "bad.alist"
    # 2 vars, 1 check; column lists say var 1 is in check 1, row list disagrees.
    bad.write_text("2 1\n1 2\n1 1\n2\n1\n1\n1 1\n")
    with pytest.raises(ValueError):
        read_alist(bad)


def test_sigma_formula():
    assert ebn0_to_sigma(0.0, 0.5) == pytest.approx(1.0)
    assert ebn0_to_sigma(10 * np.log10(2), 0.5) == pytest.approx(np.sqrt(0.5))
    t = ebn0_to_sigma(torch.tensor([0.0, 3.0]), 0.5)
    assert t[0].item() == pytest.approx(1.0)


def test_llr_statistics():
    # For the all-zero codeword, LLR ~ N(2/sigma^2, 4/sigma^2).
    rate, snr = 0.5, 2.0
    sigma = ebn0_to_sigma(snr, rate)
    llr = awgn_llr(4000, 100, snr, rate, generator=torch.Generator().manual_seed(0))
    assert llr.mean().item() == pytest.approx(2 / sigma**2, rel=0.01)
    assert llr.var().item() == pytest.approx(4 / sigma**2, rel=0.02)


def test_per_frame_snr():
    snr = torch.tensor([0.0, 10.0])
    llr = awgn_llr(2, 20000, snr, 0.5, generator=torch.Generator().manual_seed(0))
    # Higher SNR -> larger mean LLR.
    assert llr[1].mean() > 5 * llr[0].mean()
