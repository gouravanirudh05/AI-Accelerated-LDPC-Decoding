"""CPU smoke test of BP-DAT (seconds to minutes; no training).

Checks on every code of the comparison, for every config in final/:
1. training forward / backward, finite gradients, every parameter connected;
2. an untrained model (zero-initialised head) decodes exactly like BP with
   restarts (same output as staged_decode with the identity refinement);
3. MACs of one stage <= 1.5 x those of one BP-GNN stage (the budget Gourav
   set for BP-PGNN, kept so the hybrids stay comparable in cost);
4. the early-stopping decode runs for B = 1 and for a batch.
It also prints the CPU wall time of one refinement stage next to BP-GNN's.

Usage: python tests/smoke_test.py [--codes a,b] [--configs dat_config.json,...]
"""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from dat import DualAttnBPDecoder  # noqa: E402
from dat.specs import BANDS, dat_spec, load_any_code  # noqa: E402
from common.bp import bp_iterations, staged_decode  # noqa: E402
from common.bp_gnn_decoder import BPGNNDecoder  # noqa: E402
from common.train import gnn_loss  # noqa: E402

CODES = ["mackay_96_48", "ccsds_128_64", "mackay_204_102", "mackay_408_204", "wimax_576_288",
         "5G_BG2_k64_n128", "5G_BG2_k128_n256", "5G_BG2_k256_n512", "5G_BG1_k528_n704", "5G_BG1_k1056_n1408"]
CONFIGS = ["dat_config.json", "dat_nb_config.json", "dat_noself_config.json"]


def stage_ms(model, llr, g, reps=3):
    with torch.no_grad():
        L, v2c, c2v = bp_iterations(llr, g, 3)
        model.refine(llr, llr, L, v2c, c2v, g)
        t0 = time.perf_counter()
        for _ in range(reps):
            model.refine(llr, llr, L, v2c, c2v, g)
    return 1e3 * (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(CODES))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cpu")
    cfgs = []
    for c in args.configs.split(","):
        with open(os.path.join(HERE, "..", "final", c)) as f:
            cfgs.append(json.load(f))
    print(f"{'code':22s} {'model':12s} {'params':>7s} {'MACs/stage':>12s} {'x BP-GNN':>8s} "
          f"{'fwd+bwd s':>9s} {'stage ms':>8s} {'GNN ms':>7s} {'loss':>7s}")
    for name in args.codes.split(","):
        code = load_any_code(name, dev)
        g = code.graph
        band = BANDS[name]
        types = {k: dat_spec(code)["kwargs"][k] for k in ("num_vn_types", "num_cn_types", "num_edge_types")}
        bpgnn = BPGNNDecoder(**types, d=16, d_e=8, hidden=48, stages=4, bp_iters=10, gnn_layers=2).eval()
        ref_macs = bpgnn.macs_per_refinement(g)
        c, llr = code.sample(args.batch, band["val_ebno"] - 1.0)
        gnn_ms = stage_ms(bpgnn, llr, g)
        for cfg in cfgs:
            spec = dat_spec(code, dict(cfg["overrides"], bp_iters=3))
            model = DualAttnBPDecoder(**spec["kwargs"])
            model.train()
            t0 = time.time()
            loss = gnn_loss(model(llr, g), c, g, 0.1)[0]
            loss.backward()
            dt = time.time() - t0
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(x).all() for x in grads), "non-finite gradient"
            # the head is zero at init, so only it gets a gradient at step 0; give it a non-zero
            # weight and check that every other parameter is connected to the loss
            model.zero_grad()
            with torch.no_grad():
                model.head[-1].weight.normal_(0, 0.1)
            gnn_loss(model(llr, g), c, g, 0.1)[0].backward()
            dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
            assert not dead or all("emb" in n for n in dead), f"no gradient: {dead}"
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            macs = model.macs_per_refinement(g)
            ratio = macs / ref_macs
            model.eval()
            ms = stage_ms(model, llr, g)
            print(f"{name:22s} {cfg['name']:12s} {sum(p.numel() for p in model.parameters()):7d} {macs:12,d} "
                  f"{ratio:8.2f} {dt:9.2f} {ms:8.1f} {gnn_ms:7.1f} {loss.item():7.4f}", flush=True)
            assert ratio <= 1.5, f"{name}: {ratio:.2f} x BP-GNN MACs > 1.5"
            fresh = DualAttnBPDecoder(**spec["kwargs"]).eval()
            L1, u1 = fresh.decode(llr, g)
            L2, u2 = staged_decode(llr, g, fresh.stages, fresh.bp_iters, lambda x, p, *a: p)
            assert torch.allclose(L1, L2) and torch.equal(u1, u2), "untrained model != BP with restarts"
            model.decode(llr[:1], g)
            model.decode(llr, g)
    print("smoke test passed")


if __name__ == "__main__":
    main()
