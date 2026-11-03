import numpy as np
import pytest
import torch

from sparse_gt.bp import SumProductBP
from sparse_gt.channel import awgn_llr
from sparse_gt.codes import LDPCCode, load_code
from sparse_gt.mp_decoder import MessagePassingDecoder, TannerLayout, first_valid, multiloss
from sparse_gt.tests.test_model import nonzero_codeword

IRREGULAR = "WIMAX_N576_K288"  # irregular check degrees exercise the padded layout
REGULAR = "MACKAY_N204_K102"


def naive_minsum(H, llr, iterations, beta=1.0):
    m, n = H.shape
    c2v = np.zeros((m, n))
    post = llr.copy()
    for _ in range(iterations):
        v2c = np.where(H == 1, post[None, :] - c2v, 0.0)
        new = np.zeros((m, n))
        for i in range(m):
            cols = np.nonzero(H[i])[0]
            for j in cols:
                others = v2c[i, [c for c in cols if c != j]]
                new[i, j] = beta * np.prod(np.sign(np.where(others == 0, 1, others))) * np.abs(others).min()
        c2v = new
        post = llr + c2v.sum(0)
    return post


@pytest.mark.parametrize("name", [REGULAR, IRREGULAR])
def test_dense_spa_matches_validated_bp(name):
    code = load_code(name)
    llr = awgn_llr(32, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(0)).double()
    trace = MessagePassingDecoder(code, "spa", 8).double()(llr)
    ref = SumProductBP(code).double()(llr, 8, early_stop=False, return_trace=True).trace
    torch.testing.assert_close(trace, ref[1:], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("variant,beta", [("ms", 1.0), ("nms", 0.75)])
def test_minsum_matches_naive(variant, beta):
    code = load_code(REGULAR)
    llr = awgn_llr(3, code.n, 1.0, code.rate, generator=torch.Generator().manual_seed(1))
    got = MessagePassingDecoder(code, variant, 4)(llr)[-1]
    for b in range(3):
        np.testing.assert_allclose(got[b].numpy(), naive_minsum(code.H, llr[b].double().numpy(), 4, beta),
                                   rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("variant", ["noms", "wms"])
def test_learned_variants_start_as_nms(variant):
    code = load_code(IRREGULAR)
    llr = awgn_llr(16, code.n, 2.0, code.rate, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(MessagePassingDecoder(code, variant, 6)(llr),
                               MessagePassingDecoder(code, "nms", 6)(llr))


def test_first_valid_equals_bp_early_stop():
    code = load_code(REGULAR)
    llr = awgn_llr(256, code.n, 2.5, code.rate, generator=torch.Generator().manual_seed(3)).double()
    dec = MessagePassingDecoder(code, "spa", 20).double()
    ref = SumProductBP(code).double()(llr, 20)
    got = dec.decode(llr)
    # Frames converged at iteration 0 keep the channel LLR in BP; the dense
    # decoder always runs >= 1 iteration, so compare decisions there.
    torch.testing.assert_close((got < 0), (ref.llr < 0))
    conv_late = ref.converged & (ref.iterations > 0)
    torch.testing.assert_close(got[conv_late], ref.llr[conv_late], rtol=1e-7, atol=1e-7)
    # fp32 can shift the first-valid iteration in the saturating regime; the
    # hard decisions must still agree almost everywhere.
    hard32 = dec.float().decode(llr.float()) < 0
    assert (hard32 != (got < 0)).float().mean() < 1e-3


def test_syndrome_ok():
    code = load_code(IRREGULAR)
    lay = TannerLayout(code)
    c = nonzero_codeword(code.H)
    llr = torch.as_tensor(np.stack([np.ones(code.n), 1 - 2.0 * c]), dtype=torch.float32)
    bad = llr[1:].clone()
    bad[0, 0] *= -1
    assert lay.syndrome_ok(llr).tolist() == [True, True]
    assert lay.syndrome_ok(bad).tolist() == [False]


@pytest.mark.parametrize("variant", ["spa", "noms", "wms"])
def test_codeword_equivariance(variant):
    code = load_code(IRREGULAR)
    dec = MessagePassingDecoder(code, variant, 5)
    if variant != "spa":
        with torch.no_grad():
            for p in dec.parameters():
                p.add_(torch.rand(p.shape, generator=torch.Generator().manual_seed(4)) * 0.3)
    c = nonzero_codeword(code.H)
    flip = torch.as_tensor(1.0 - 2.0 * c, dtype=torch.float32)
    llr = awgn_llr(16, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(dec(llr * flip), dec(llr) * flip, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("variant", ["noms", "wms"])
def test_learned_parameters_get_gradients(variant):
    code = load_code(REGULAR)
    dec = MessagePassingDecoder(code, variant, 5)
    llr = awgn_llr(64, code.n, 1.0, code.rate, generator=torch.Generator().manual_seed(6))
    multiloss(dec(llr)).backward()
    for p in dec.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_fp16_minsum_close_to_fp32():
    code = load_code(IRREGULAR)
    dec = MessagePassingDecoder(code, "nms", 10)
    llr = awgn_llr(64, code.n, 2.0, code.rate, generator=torch.Generator().manual_seed(7))
    hard32 = dec.decode(llr) < 0
    hard16 = dec.decode(llr.half()) < 0
    assert (hard32 != hard16).float().mean() < 1e-3
