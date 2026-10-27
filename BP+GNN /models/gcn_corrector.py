"""GCN residual corrector -- the floor baseline.

Standard symmetric degree-normalized aggregation with *separate* weight
matrices for the V->C and C->V passes.  There is no attention and no edge
conditioning, so this is the weakest of the four correctors on purpose: it
measures how much of any improvement is just "some graph network on top of BP"
rather than the specific architecture.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector
from .graph_ops import MessagePassingBlock, linear_flops_for


class GCNPass(MessagePassingBlock):
    """One directed, degree-normalized graph-convolution pass."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="sum")
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.root = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.root.bias)

    def message(
        self,
        source_features: Tensor,
        target_features: Tensor,
        edge_features: Optional[Tensor],
    ) -> Tensor:
        # edge_features carries the precomputed 1/sqrt(d_src * d_tgt) factor.
        transformed = self.linear(source_features)
        if edge_features is None:
            return transformed
        return transformed * edge_features

    def forward(
        self,
        edge_index: Tensor,
        source_features: Tensor,
        target_features: Tensor,
        norm: Tensor,
    ) -> Tensor:
        # Self-loop / root term is applied to the target's own features, as in
        # standard GCN, using the *target* side's weights.
        aggregated = self.propagate(
            edge_index, source_features, target_features, edge_features=norm
        )
        return aggregated + self.root(target_features)


class GCNCorrector(BaseCorrector):
    """1-2 layer bipartite GCN over the Tanner graph."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        residual_mode: str = "node",
    ) -> None:
        # This corrector carries no edge features, so edge_dim stays 0 and the
        # base class rejects residual_mode='edge' with an explanatory error.
        super().__init__(
            graph=graph,
            vn_dim=vn_dim,
            cn_dim=cn_dim,
            hidden_dim=hidden_dim,
            use_gate=use_gate,
            fixed_gate=fixed_gate,
            correction_scale=correction_scale,
            dropout=dropout,
            residual_mode=residual_mode,
            edge_dim=0,
        )
        if num_layers not in (1, 2):
            raise ValueError("GCNCorrector supports 1 or 2 layers.")
        self.num_layers = int(num_layers)

        self.vn_input = nn.Linear(vn_dim, hidden_dim)
        self.cn_input = nn.Linear(cn_dim, hidden_dim)
        # Separate parameters per direction, and per layer.
        self.to_cn = nn.ModuleList(
            GCNPass(hidden_dim, hidden_dim) for _ in range(self.num_layers)
        )
        self.to_vn = nn.ModuleList(
            GCNPass(hidden_dim, hidden_dim) for _ in range(self.num_layers)
        )
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

        # Symmetric normalization, computed once from the static degrees.
        inv_sqrt_vn = graph.vn_degrees.clamp_min(1.0).rsqrt()
        inv_sqrt_cn = graph.cn_degrees.clamp_min(1.0).rsqrt()
        edge_norm = (
            inv_sqrt_vn[graph.edge_vn] * inv_sqrt_cn[graph.edge_cn]
        ).view(1, -1, 1)
        # The same scalar per physical edge serves both directions.
        self.register_buffer("edge_norm", edge_norm, persistent=False)

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        vn_hidden = self.activation(self.vn_input(vn_features))
        cn_hidden = self.activation(self.cn_input(cn_features))
        stages: List[Tensor] = []

        for layer in range(self.num_layers):
            cn_update = self.to_cn[layer](
                vn_to_cn_edge_index, vn_hidden, cn_hidden, self.edge_norm
            )
            cn_hidden = self.drop(self.activation(cn_update)) + cn_hidden

            vn_update = self.to_vn[layer](
                cn_to_vn_edge_index, cn_hidden, vn_hidden, self.edge_norm
            )
            vn_hidden = self.drop(self.activation(vn_update)) + vn_hidden
            stages.append(vn_hidden)

        return stages

    def estimate_flops_per_frame(self) -> float:
        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        total = linear_flops_for(self.vn_input, n_vn) + linear_flops_for(
            self.cn_input, n_cn
        )
        for layer in range(self.num_layers):
            # Message weights are applied once per edge, root weights once per
            # target node.
            total += linear_flops_for(self.to_cn[layer].linear, n_edges)
            total += linear_flops_for(self.to_cn[layer].root, n_cn)
            total += linear_flops_for(self.to_vn[layer].linear, n_edges)
            total += linear_flops_for(self.to_vn[layer].root, n_vn)
        return total + self.head_flops_per_frame()
