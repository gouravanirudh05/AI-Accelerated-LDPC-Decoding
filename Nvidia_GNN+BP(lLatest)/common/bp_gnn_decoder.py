"""BP-GNN hybrid decoder: the method of Gong, Cammerer & Renes (NVIDIA, 2023).

"The proposed algorithm is composed of classical belief propagation (BP)
decoding stages and intermediate graph neural network (GNN) layers. Both
component decoders are defined over the same sparse decoding graph [...] The
core idea is to use the GNN component between consecutive BP runs, so that the
knowledge from the previous BP run, if stuck in a local minimum caused by
trapping sets or short cycles, can be leveraged to better initialize the next
BP run." The whole chain is differentiable and trained end to end.

    llr -> BP(T) -> GNN -> BP(T) -> GNN -> ... -> BP(T)      (S GNN stages)

Adaptation to classical / 5G LDPC codes over AWGN (the paper decodes quantum
codes from a syndrome under depolarizing noise):
* the GNN sees the soft channel LLRs, the prior of the stuck run, its BP
  marginals, the soft and hard syndrome on the CNs and the final BP messages
  (v2c, c2v) as edge features, and outputs the prior of the next BP run as a
  residual correction of the channel LLRs;
* loss = BCE on the marginal of every BP stage after a GNN + soft-syndrome term;
* hard-example mining: training batches are filled with frames on which plain
  BP fails (the only frames on which the GNN ever acts at inference);
* node/edge type embeddings (degrees, or 5G base-graph positions, Z-agnostic);
* at inference, frames whose syndrome is satisfied leave the batch after any BP
  iteration, so the GNN is only evaluated for frames BP could not solve.
"""

import torch
import torch.nn as nn

from .bp import bp_iterations, staged_decode
from .gnn_decoder import EdgeMLP, llr_features, mlp, soft_syndrome
from .graph import scatter_sum


class BPGNNDecoder(nn.Module):
    def __init__(self, num_vn_types, num_cn_types, num_edge_types, d=16, d_e=8, hidden=48,
                 stages=4, bp_iters=10, gnn_layers=2, detach_bp=False):
        super().__init__()
        self.stages, self.bp_iters, self.gnn_layers = stages, bp_iters, gnn_layers
        # detach_bp: "chained" variant, BP runs without autograd and every GNN is
        # trained on its own output (no gradient through the following BP runs)
        self.detach_bp = detach_bp
        self.iters = (stages + 1) * bp_iters  # total BP iteration budget
        self.vn_emb = nn.Embedding(num_vn_types, d)
        self.cn_emb = nn.Embedding(num_cn_types, d)
        self.edge_emb = nn.Embedding(num_edge_types, d_e)
        self.vn_in = nn.Linear(6, d)
        self.cn_in = nn.Linear(2, d)
        self.edge_in = nn.Linear(2, d_e)
        self.msg_vc = nn.ModuleList(EdgeMLP(d, d_e, hidden, d) for _ in range(gnn_layers))
        self.msg_cv = nn.ModuleList(EdgeMLP(d, d_e, hidden, d) for _ in range(gnn_layers))
        self.upd_c = nn.ModuleList(mlp(2 * d + 2, hidden, d) for _ in range(gnn_layers))
        self.upd_v = nn.ModuleList(mlp(2 * d + 6, hidden, d) for _ in range(gnn_layers))
        self.readout = mlp(d, hidden, 1)
        # zero-initialised correction: an untrained hybrid restarts BP from the channel LLRs
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def refine(self, llr, prior, L, v2c, c2v, g):
        """GNN between two BP runs: stuck BP state -> prior of the next run."""
        f_v = torch.cat([llr_features(llr), llr_features(prior), llr_features(L)], -1)
        hard_syn = g.syndrome((L < 0).to(L.dtype))
        f_c = torch.stack([soft_syndrome(L, g), 1.0 - 2.0 * hard_syn], -1)
        e = self.edge_in(torch.stack([torch.tanh((v2c / 2).clamp(-15, 15)),
                                      torch.tanh((c2v / 2).clamp(-15, 15))], -1))
        e = e + self.edge_emb(g.edge_type)
        h_v = self.vn_in(f_v) + self.vn_emb(g.vn_type)
        h_c = self.cn_in(f_c) + self.cn_emb(g.cn_type)
        inv_dc, inv_dv = (1.0 / g.cn_deg)[:, None], (1.0 / g.vn_deg)[:, None]
        for l in range(self.gnn_layers):
            m = self.msg_vc[l](h_v, g.e_v, h_c, g.e_c, e)
            h_c = h_c + self.upd_c[l](torch.cat([h_c, scatter_sum(m, g.e_c, g.m) * inv_dc, f_c], -1))
            m = self.msg_cv[l](h_c, g.e_c, h_v, g.e_v, e)
            h_v = h_v + self.upd_v[l](torch.cat([h_v, scatter_sum(m, g.e_v, g.n) * inv_dv, f_v], -1))
        return (llr + self.readout(h_v).squeeze(-1)).clamp(-30, 30)

    def forward(self, llr, g):
        """Differentiable end-to-end pass. Returns the marginals of every BP run
        that follows a GNN stage (the first BP run has no trainable input)."""
        prior, outs = llr, []
        if self.detach_bp:
            for s in range(self.stages):
                with torch.no_grad():
                    L, v2c, c2v = bp_iterations(prior.detach(), g, self.bp_iters)
                prior = self.refine(llr, prior.detach(), L, v2c, c2v, g)
                outs.append(prior)
            return outs
        for s in range(self.stages + 1):
            L, v2c, c2v = bp_iterations(prior, g, self.bp_iters)
            if s > 0:
                outs.append(L)
            if s < self.stages:
                prior = self.refine(llr, prior, L, v2c, c2v, g)
        return outs

    @torch.no_grad()
    def decode(self, llr, g, iters=None, early_stop=True):
        return staged_decode(llr, g, self.stages, self.bp_iters, self.refine, early_stop)

    def macs_per_refinement(self, g):
        """Multiply-accumulates of one GNN stage per codeword."""
        d = self.vn_in.out_features
        h = self.readout[0].out_features
        d_e = self.edge_emb.embedding_dim
        per_layer = ((g.n + g.m) * 2 * d * h + 2 * g.num_edges * (d_e * h + h * d)
                     + g.m * ((2 * d + 2) * h + h * d) + g.n * ((2 * d + 6) * h + h * d))
        inputs = g.n * 6 * d + g.m * 2 * d + g.num_edges * 2 * d_e
        return int(self.gnn_layers * per_layer + inputs + g.n * (d * h + h))
