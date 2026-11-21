"""CPU smoke test of BP-PGNN (seconds; no training).

Checks on every code of the comparison:
1. training forward / backward for pool = diff, topk, none;
2. an untrained model (zero-initialised head) decodes exactly like BP with
   restarts (same output as staged_decode with the identity refinement);
3. MACs of one PGNN stage <= 1.5 x those of one BP-GNN stage (Gourav's budget).

Usage: python tests/smoke_test.py [--codes a,b]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pgnn import PooledGNNBPDecoder  # noqa: E402
from pgnn.specs import BANDS, pgnn_spec, load_any_code  # noqa: E402
from common.bp import staged_decode  # noqa: E402
from common.bp_gnn_decoder import BPGNNDecoder  # noqa: E402
from common.train import gnn_loss  # noqa: E402

CODES = ["mackay_96_48", "ccsds_128_64", "mackay_204_102", "mackay_408_204", "wimax_576_288",
         "5G_BG2_k64_n128", "5G_BG2_k128_n256", "5G_BG2_k256_n512", "5G_BG1_k528_n704", "5G_BG1_k1056_n1408"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(CODES))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cpu")
    print(f"{'code':22s} {'pool':5s} {'assign':7s} {'params':>7s} {'MACs/stage':>12s} {'x BP-GNN':>8s} "
          f"{'fwd+bwd s':>9s} {'loss':>7s}")
    for name in args.codes.split(","):
        code = load_any_code(name, dev)
        g = code.graph
        band = BANDS[name]
        ref_spec = pgnn_spec(code)  # resolves types / clusters exactly as the final run
        types = {k: ref_spec["kwargs"][k] for k in ("num_vn_types", "num_cn_types", "num_edge_types")}
        bpgnn = BPGNNDecoder(**types, d=16, d_e=8, hidden=48, stages=4, bp_iters=10, gnn_layers=2)
        ref_macs = bpgnn.macs_per_refinement(g)
        c, llr = code.sample(args.batch, band["val_ebno"] - 1.0)
        for pool in ("diff", "topk", "none"):
            spec = pgnn_spec(code, dict(pool=pool, bp_iters=3))
            model = PooledGNNBPDecoder(**spec["kwargs"])
            model.train()
            t0 = time.time()
            outs = model(llr, g)
            loss = gnn_loss(outs, c, g, 0.1)[0]
            loss.backward()
            dt = time.time() - t0
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(x).all() for x in grads), "non-finite gradient"
            # head weights are zero at init, so only the head gets a gradient through the BP chain at step 0;
            # check the rest of the network is connected by giving the head a non-zero weight
            model.zero_grad()
            with torch.no_grad():
                model.head[-1].weight.normal_(0, 0.1)
            gnn_loss(model(llr, g), c, g, 0.1)[0].backward()
            dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
            assert not dead or all("edge_emb" in n or "emb" in n for n in dead), f"no gradient: {dead}"
            macs = model.macs_per_refinement(g)
            ratio = macs / ref_macs
            print(f"{name:22s} {pool:5s} {model.assign:7s} {sum(p.numel() for p in model.parameters()):7d} "
                  f"{macs:12,d} {ratio:8.2f} {dt:9.2f} {loss.item():7.4f}")
            if pool == "diff":
                assert ratio <= 1.5, f"{name}: {ratio:.2f} x BP-GNN MACs > 1.5"
            # untrained model == BP with restarts
            fresh = PooledGNNBPDecoder(**spec["kwargs"]).eval()
            L1, u1 = fresh.decode(llr, g)
            L2, u2 = staged_decode(llr, g, fresh.stages, fresh.bp_iters, lambda x, p, *a: p)
            assert torch.allclose(L1, L2) and torch.equal(u1, u2), "untrained model != BP with restarts"
            # early-stop decode of a trained-looking model runs (shrinking batch, B = 1 path)
            model.eval()
            model.decode(llr[:1], g)
            model.decode(llr, g)
    print("smoke test passed")


if __name__ == "__main__":
    main()
