"""Monte-Carlo BER / BLER simulation and decoder wrappers."""

import time

import torch

from .bp import bp_decode, random_perturbation, staged_decode
from .osd import osd


def model_label(model):
    if getattr(model, "label", None):
        return model.label
    if hasattr(model, "stages"):
        tag = " detached-BP" if getattr(model, "detach_bp", False) else ""
        return f"BP-GNN{tag} ({model.stages + 1}x{model.bp_iters} BP it)"
    if hasattr(model, "per_edge"):
        return f"NBP {'per-edge' if model.per_edge else 'shared'} ({model.iters} it)"
    return f"GNN ({model.iters} it)"


def base_decoders(g, models=(), bp_iters=(10, 50), perturb=None, nms_iters=()):
    """{name: (fn(llr, early_stop) -> (LLR, iterations), learned?)}.

    learned is None for a model with osd_post=False (no OSD post-processing).
    perturb: (stages, iters, scale) adds the random-perturbation BP baseline."""
    decs = {}
    for m in models:
        decs[model_label(m)] = ((lambda m: lambda llr, es: m.decode(llr, g, early_stop=es))(m),
                                True if getattr(m, "osd_post", True) else None)
    for T in bp_iters:
        decs[f"BP ({T} it)"] = ((lambda T: lambda llr, es: bp_decode(llr, g, T, "spa", early_stop=es))(T), False)
    if perturb:
        S, T, scale = perturb
        ref = random_perturbation(scale)
        decs[f"BP + perturb. ({S + 1}x{T} BP it)"] = (
            lambda llr, es: staged_decode(llr, g, S, T, ref, early_stop=es), False)
    for T in nms_iters:
        decs[f"NMS ({T} it)"] = ((lambda T: lambda llr, es: bp_decode(llr, g, T, "nms", early_stop=es))(T), False)
    return decs


def osd_postprocess(dec, g, rank, order, lam=16):
    """Wrap fn(llr)->(L, used) so that frames failing the syndrome check get OSD."""
    def fn(llr):
        L, used = dec(llr)
        hard = (L < 0).float()
        fail = ~(g.syndrome(hard) == 0).all(1)
        n_osd = int(fail.sum())
        if n_osd:
            hard[fail] = osd(L[fail], g.H, rank, order=order, lam=lam)
        return hard, used, n_osd
    return fn


def make_decoders(code, models=(), bp_iters=(10, 50), perturb=None, osd=True, early_stop=True,
                  osd_learned=True, osd_orders=(0,), osd_learned_orders=None, bp_osd_iters=None):
    """Returns {name: fn(llr) -> (hard bits [B, n], iterations [B], osd_calls)}.

    OSD post-processing (orders ``osd_orders``) is added to BP with ``bp_osd_iters``
    iterations (default: the longest BP), orders ``osd_learned_orders`` to every
    learned decoder."""
    g = code.graph
    if osd_learned_orders is None:
        osd_learned_orders = osd_orders if osd_learned else ()
    bp_osd_iters = bp_osd_iters or max(bp_iters)

    def plain(dec):
        def fn(llr):
            L, used = dec(llr)
            return (L < 0).float(), used, 0
        return fn

    decs = {}
    for name, (fn, learned) in base_decoders(g, models, bp_iters, perturb).items():
        dec = (lambda fn: lambda llr: fn(llr, early_stop))(fn)
        decs[name] = plain(dec)
        if not osd:
            continue
        orders = osd_learned_orders if learned else (osd_orders if name == f"BP ({bp_osd_iters} it)" else ())
        for o in orders:
            decs[f"{name} + OSD-{o}"] = (lambda dec, o: (lambda f: lambda llr: f(llr))(
                osd_postprocess(dec, g, code.rank, o)))(dec, o)
    return decs


@torch.no_grad()
def simulate(fn, code, ebno_list, batch, max_frames, min_frame_errors, max_seconds=None, seed=0):
    torch.manual_seed(seed)
    res = []
    for ebno in ebno_list:
        be = fe = nf = it = n_osd = 0
        t0 = time.time()
        while nf < max_frames and fe < min_frame_errors:
            c, llr = code.sample(batch, ebno)
            hard, used, k_osd = fn(llr)
            err = hard != c
            be += err[:, code.info_pos].sum().item()
            fe += err.any(1).sum().item()
            it += used.sum().item()
            n_osd += k_osd
            nf += batch
            if max_seconds and time.time() - t0 > max_seconds:
                break
        res.append(dict(ebno=float(ebno), ber=be / (nf * code.k), bler=fe / nf, frames=nf,
                        frame_errors=fe, bit_errors=be, avg_iters=it / nf, osd_frac=n_osd / nf,
                        seconds=round(time.time() - t0, 2)))
        # stop the curve once the decoder makes no errors at all
        if fe == 0:
            break
    return res
