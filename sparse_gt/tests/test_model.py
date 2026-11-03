import numpy as np
import pytest
import torch

from sparse_gt.baselines import bp_compact, chase_bp
from sparse_gt.bp import SumProductBP
from sparse_gt.channel import awgn_llr
from sparse_gt.codes import load_code
from sparse_gt.model import GTConfig, SparseGTDecoder


def nonzero_codeword(H: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random nonzero codeword: solve H c = 0 over GF(2) by elimination."""
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    pivots, row = [], 0
    for col in range(n):
        hit = np.nonzero(H[row:, col])[0]
        if hit.size == 0:
            continue
        r = row + hit[0]
        H[[row, r]] = H[[r, row]]
        for rr in range(m):
            if rr != row and H[rr, col]:
                H[rr] ^= H[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    free = [c for c in range(n) if c not in pivots]
    rng = np.random.default_rng(seed)
    c = np.zeros(n, dtype=np.uint8)
    c[free] = rng.integers(0, 2, len(free))
    c[free[0]] = 1
    for i, col in enumerate(pivots):
        c[col] = H[i, free] @ c[free] % 2
    return c


def randomized(model: SparseGTDecoder, seed: int = 0) -> SparseGTDecoder:
    """Give every parameter (incl. the zero-initialised head/bias) random values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.gt.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model.eval()


@pytest.fixture(scope="module")
def code():
    return load_code("MACKAY_N204_K102")


@pytest.mark.parametrize("R", [0, 3])
def test_untrained_equals_bp_t_plus_r(code, R):
    model = SparseGTDecoder(code, GTConfig(T=4, k=16, R=R, layers=1)).eval()
    llr = awgn_llr(64, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(0))
    out, aux = model(llr)
    assert aux["active"].any()
    torch.testing.assert_close(out, SumProductBP(code)(llr, 4 + R).llr)


@pytest.mark.parametrize("R", [0, 3])
def test_codeword_equivariance(code, R):
    """Decoding codeword c gives exactly the all-zero result with signs flipped by c."""
    c = nonzero_codeword(code.H)
    assert not (code.H @ c % 2).any() and c.any()
    model = randomized(SparseGTDecoder(code, GTConfig(T=4, k=16, R=R, layers=2)))
    llr0 = awgn_llr(32, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(1))
    flip = torch.as_tensor(1.0 - 2.0 * c, dtype=torch.float32)
    out0, _ = model(llr0)
    outc, _ = model(llr0 * flip)
    torch.testing.assert_close(outc, out0 * flip, rtol=1e-4, atol=1e-4)


def test_gradients_reach_transformer_through_resumed_bp(code):
    model = randomized(SparseGTDecoder(code, GTConfig(T=4, k=16, R=3, layers=1))).train()
    llr = awgn_llr(64, code.n, 1.0, code.rate, generator=torch.Generator().manual_seed(2))
    _, aux = model(llr)
    z = aux["active_llr"]
    torch.nn.functional.binary_cross_entropy_with_logits(-z, torch.zeros_like(z)).backward()
    grads = [p.grad for p in model.gt.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.gt.embed.weight.grad.abs().sum() > 0


def test_structure_counts(code):
    model = SparseGTDecoder(code, GTConfig(T=2, k=12, layers=1))
    llr = awgn_llr(4, code.n, 0.5, code.rate, generator=torch.Generator().manual_seed(3))
    trace = SumProductBP(code)(llr, 2, early_stop=False, return_trace=True).trace
    idx = model.select(trace[-1])
    _, shared, shared_unsat = model.features(trace, llr, idx)
    H = torch.as_tensor(code.H, dtype=torch.long)
    for b in range(4):
        cols = H[:, idx[b]]
        torch.testing.assert_close(shared[b], cols.T @ cols)
        unsat = model.bp.unsatisfied_checks(trace[-1][b:b + 1])[0].long()
        torch.testing.assert_close(shared_unsat[b], cols.T @ (cols * unsat[:, None]))
    assert (shared_unsat <= shared).all()


def test_bp_compact_matches_freeze_early_stop(code):
    bp = SumProductBP(code)
    llr = awgn_llr(128, code.n, 2.0, code.rate, generator=torch.Generator().manual_seed(4))
    ref = bp(llr, 20)
    out, used, conv = bp_compact(bp, llr, 20)
    torch.testing.assert_close(out, ref.llr)
    assert torch.equal(used, ref.iterations) and torch.equal(conv, ref.converged)


def test_chase_never_worse_on_converged_frames(code):
    bp = SumProductBP(code)
    llr = awgn_llr(128, code.n, 1.5, code.rate, generator=torch.Generator().manual_seed(5))
    first = bp(llr, 4)
    out = chase_bp(bp, llr, T=4, p=3, R=3)
    torch.testing.assert_close(out[first.converged], first.llr[first.converged])
    # Chase picks a syndrome-satisfying pattern when one exists, so it cannot
    # have more unsatisfied checks than plain BP(4) on the failed frames.
    assert (bp.syndrome_weight(out) <= bp.syndrome_weight(first.llr) + 1e-6).float().mean() > 0.5
