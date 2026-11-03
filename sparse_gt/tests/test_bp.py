import numpy as np
import pytest
import torch

from sparse_gt.bp import SumProductBP
from sparse_gt.channel import awgn_llr
from sparse_gt.codes import LDPCCode, load_code


def small_code() -> LDPCCode:
    H = np.array(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return LDPCCode("small", H, k=4)


def reference_bp(H: np.ndarray, llr: np.ndarray, iterations: int) -> np.ndarray:
    """Textbook flooding sum-product with the tanh rule, no early stopping.

    Deliberately written independently of SumProductBP: dense matrices,
    explicit loops, float64.
    """
    m, n = H.shape
    c2v = np.zeros((m, n))
    post = llr.copy()
    for _ in range(iterations):
        v2c = np.where(H == 1, post[None, :] - c2v, 0.0)
        new = np.zeros((m, n))
        for i in range(m):
            cols = np.nonzero(H[i])[0]
            for j in cols:
                others = [c for c in cols if c != j]
                prod = np.prod(np.tanh(v2c[i, others] / 2.0))
                prod = np.clip(prod, -1 + 1e-15, 1 - 1e-15)
                new[i, j] = 2.0 * np.arctanh(prod)
        c2v = new
        post = llr + c2v.sum(0)
    return post


@pytest.mark.parametrize("iterations", [1, 2, 5])
def test_matches_textbook_reference(iterations):
    code = small_code()
    bp = SumProductBP(code).double()
    rng = np.random.default_rng(0)
    for _ in range(5):
        llr = rng.normal(1.0, 1.5, size=code.n)
        expected = reference_bp(code.H, llr, iterations)
        got = bp(torch.tensor(llr)[None], iterations, early_stop=False).llr[0].numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_matches_reference_on_real_code_float32():
    code = load_code("MACKAY_N204_K102")
    bp = SumProductBP(code)
    llr = awgn_llr(2, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(3))
    got = bp(llr, 5, early_stop=False).llr
    for b in range(2):
        expected = reference_bp(code.H, llr[b].double().numpy(), 5)
        # Compare where messages are not saturated (float32 caps |c2v| at 30).
        mask = np.abs(expected) < 25
        np.testing.assert_allclose(got[b].numpy()[mask], expected[mask], rtol=1e-3, atol=1e-3)


def test_noiseless_frame_converges_immediately():
    code = load_code("WIMAX_N576_K288")
    res = SumProductBP(code)(torch.full((3, code.n), 4.0), 10)
    assert res.converged.all() and (res.iterations == 0).all()
    assert (res.llr > 0).all()


def test_trace_equals_output_with_that_budget():
    code = load_code("MACKAY_N204_K102")
    bp = SumProductBP(code)
    llr = awgn_llr(64, code.n, 2.0, code.rate, generator=torch.Generator().manual_seed(1))
    full = bp(llr, 8, return_trace=True)
    assert full.trace.shape == (9, 64, code.n)
    torch.testing.assert_close(full.trace[0], llr)
    for t in (1, 3, 8):
        torch.testing.assert_close(full.trace[t], bp(llr, t).llr)
    torch.testing.assert_close(full.trace[-1], full.llr)


def test_early_stop_freezes_output_and_counts_iterations():
    code = load_code("MACKAY_N204_K102")
    bp = SumProductBP(code)
    llr = awgn_llr(256, code.n, 3.0, code.rate, generator=torch.Generator().manual_seed(2))
    res = bp(llr, 30, return_trace=True)
    for b in torch.nonzero(res.converged).flatten()[:20].tolist():
        t = int(res.iterations[b])
        assert bp.syndrome_weight(res.trace[t, b:b + 1]).item() == 0
        if t > 0:
            assert bp.syndrome_weight(res.trace[t - 1, b:b + 1]).item() > 0
        torch.testing.assert_close(res.trace[t:, b], res.llr[b].expand(31 - t, -1))


def test_resume_equals_uninterrupted_run():
    code = load_code("MACKAY_N204_K102")
    bp = SumProductBP(code)
    llr = awgn_llr(16, code.n, 1.0, code.rate, generator=torch.Generator().manual_seed(4))
    first = bp(llr, 4, early_stop=False)
    resumed = bp(llr, 6, early_stop=False, c2v_init=first.c2v)
    torch.testing.assert_close(resumed.llr, bp(llr, 10, early_stop=False).llr)


def test_differentiable():
    code = load_code("MACKAY_N204_K102")
    bp = SumProductBP(code)
    llr = awgn_llr(8, code.n, 1.0, code.rate).requires_grad_(True)
    out = bp(llr, 5).llr
    torch.nn.functional.binary_cross_entropy_with_logits(
        -out, torch.zeros_like(out)).backward()
    assert torch.isfinite(llr.grad).all() and llr.grad.abs().sum() > 0
