"""Diagnose / fix the pure GNN on 5G NR (BLER 1.0, flat loss in the tuning sweep).

Finding (run 1): every check of the 5G graph touches a punctured bit (LLR 0), so
all soft syndromes are 0 and a freshly initialised GNN sits at a saddle with
~1e-5 gradients (MacKay: ~1e-1). This run compares, on BG2 (k=64, n=128), short
CPU trainings of the GNN with
  A  zero readout init (as before)        B  random readout init
  C  A + punctured-bit hint curriculum    D  B + hint
and reports BER / BLER at 3.5 dB (true puncturing) next to BP(10) / BP(50).
Usage: python diag_gnn_5g.py [--minutes 8]
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "5g_ldpc"))
from common.bp import bp_decode  # noqa: E402
from common.gnn_decoder import GNNDecoder  # noqa: E402
from common.train import train_gnn  # noqa: E402
from nr_ldpc import BG_SHAPE, NUM_BG_EDGES, nr_code  # noqa: E402


@torch.no_grad()
def rates(dec, code, ebno, frames=8192, batch=1024):
    torch.manual_seed(1)
    be = fe = 0
    for _ in range(frames // batch):
        c, llr = code.sample(batch, ebno)
        hard = (dec(llr) < 0).float()
        err = hard != c
        be += err[:, code.info_pos].sum().item()
        fe += err.any(1).sum().item()
    return be / (frames * code.k), fe / frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    args = ap.parse_args()
    code = nr_code(64, 128, bg="bg2", device="cpu")
    g = code.graph
    rows, cols = BG_SHAPE["bg2"]
    types = dict(num_vn_types=cols, num_cn_types=rows, num_edge_types=NUM_BG_EDGES["bg2"])
    for T in (10, 50):
        ber, bler = rates(lambda llr: bp_decode(llr, g, T, "spa", early_stop=False)[0], code, 3.5)
        print(f"BP ({T} it): BER {ber:.2e} BLER {bler:.2e}", flush=True)
    base = dict(steps=10 ** 7, batch=128, lr=1e-3, lr_min=1e-5, train_ebno=(0.5, 4.5), lam_syn=0.1,
                eval_every=200, val_ebno=3.5, val_frames=2048, grad_clip=1.0, max_minutes=args.minutes,
                schedule="plateau", patience=6)
    hint = dict(punct_hint=1.0, punct_hint_frac=0.5)
    variants = [("A zero init", "zero", {}), ("B random init", "default", {}),
                ("C zero init + hint", "zero", hint), ("D random init + hint", "default", hint)]
    for name, init, extra in variants:
        torch.manual_seed(0)
        model = GNNDecoder(**types, d=16, d_e=8, hidden=48, iters=10, readout_init=init)
        print(f"\n=== {name} ===", flush=True)
        train_gnn(model, [code], dict(base, **extra), lambda *a: print(*a, flush=True))
        ber, bler = rates(lambda llr: model.decode(llr, g, early_stop=False)[0], code, 3.5)
        print(f"RESULT {name}: BER {ber:.2e} BLER {bler:.2e}", flush=True)


if __name__ == "__main__":
    main()
