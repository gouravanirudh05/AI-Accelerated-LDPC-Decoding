"""Analytic cost of one SGT stage per frame (MACs and elements moved), per component,
for every code of the final comparison. Runs on CPU in seconds; it predicts where the
time should go if the GPU were compute bound, to compare with the measured tables.

  python cost_model.py [--k-frac 0.5] > results/cost_model.md
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lat  # noqa: E402,F401
from run_final import code_specs  # noqa: E402


def stage_cost(n, m, E, k, d=64, heads=4, hidden=128, layers=2, struct=True):
    c = {}
    c["features (E-sized gathers/scatters)"] = (0, 12 * E + 6 * n)
    c["select (top-k over n)"] = (0, n * 4)
    c["tokens (9 features, Linear 9->d)"] = (k * 9 * d, k * (9 + 2 * d))
    if struct:
        c["bias/Hk gather [k, m]"] = (0, k * m)
        c["bias/shared bmm k*k*m"] = (k * k * m, k * k)
        c["bias/shared_unsat mul + bmm k*k*m"] = (k * k * m, k * m + k * k)
        c["bias/embed [k, k, heads]"] = (0, k * k * (2 + 2 * heads))
    c["attention (qkv, proj, QK^T, AV) x layers"] = (layers * (4 * k * d * d + 2 * k * k * d),
                                                     layers * (k * k * heads * 3 + 6 * k * d))
    c["FFN x layers"] = (layers * 2 * k * d * hidden, layers * k * (2 * d + hidden))
    c["head + scatter"] = (k * d, 2 * n)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-frac", type=float, default=0.5)
    a = ap.parse_args()
    print(f"# Analytic cost of one SGT stage per frame (k_frac = {a.k_frac}, d 64, 4 heads, 2 layers, FFN 128)\n")
    print("MMAC = million multiply-accumulates; Melem = million tensor elements read or written (memory traffic proxy).\n")
    print("| code | n | m | k | total MMAC | bias build MMAC (share) | attention MMAC | FFN MMAC | "
          "total Melem | bias Melem (share) | k*k*m bias / BP(10) MAC ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for grp in ("normal", "5g"):
        for code, _, _, _ in code_specs(grp, torch.device("cpu")):
            g = code.graph
            k = min(g.n, max(8, -(-int(a.k_frac * g.n * 1000) // 1000)))
            c = stage_cost(g.n, g.m, g.num_edges, k)
            mac = sum(v[0] for v in c.values())
            el = sum(v[1] for v in c.values())
            bmac = sum(v[0] for kk, v in c.items() if kk.startswith("bias"))
            bel = sum(v[1] for kk, v in c.items() if kk.startswith("bias"))
            att = c["attention (qkv, proj, QK^T, AV) x layers"][0]
            ffn = c["FFN x layers"][0]
            bp10 = 10 * 8 * g.num_edges  # ~8 flops per edge per SPA iteration
            print(f"| {code.name} | {g.n} | {g.m} | {k} | {mac / 1e6:.1f} | {bmac / 1e6:.1f} ({100 * bmac / mac:.0f}%) | "
                  f"{att / 1e6:.1f} | {ffn / 1e6:.1f} | {el / 1e6:.2f} | {bel / 1e6:.2f} ({100 * bel / el:.0f}%) | "
                  f"{bmac / bp10:.0f}x |")


if __name__ == "__main__":
    main()
