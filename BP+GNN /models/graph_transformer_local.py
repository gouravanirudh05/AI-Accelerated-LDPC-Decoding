"""Local graph-transformer corrector.

A GATv2-style attention block with FFN, residual, and LayerNorm on each
directed pass -- reusing the Phase 0-2 ``DirectedEdgeGATv2``, which already has
exactly that structure.

``hop_count`` controls the receptive field:

``1``
    the bipartite V->C then C->V passes only.
``2``
    plus a variable-to-variable attention pass over pairs that share a check.
    That 2-hop edge index is materialized once in ``__init__`` (at graph-build
    time) and cached as a buffer -- never recomputed inside a forward pass.

``edge_exclusion`` and ``attention_variant`` ("v1" / "v2") select the
aggregation rule and the attention formulation, exactly as in
:mod:`~.gat_corrector`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .._compat import DirectedEdgeGATv2, LegacyTannerGraph
from ..bp_trace import EDGE_FEATURE_DIM, VN_IDX_SOFT_FINAL, build_edge_features
from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector, EncodeResult
from .gat_corrector import ConfigurableEdgeGATPass
from .graph_ops import gather_nodes, linear_flops, linear_flops_for

# 2-hop edge features: how many checks the pair shares, and whether the two
# variables' current soft bits agree. Disagreeing pairs that share checks are
# where residual errors concentrate.
TWO_HOP_EDGE_DIM = 2


class GraphTransformerLocal(BaseCorrector):
    """1- or 2-hop local attention over the Tanner graph."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        edge_dim: int = EDGE_FEATURE_DIM,
        hop_count: int = 1,
        edge_exclusion: bool = False,
        attention_variant: str = "v2",
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        residual_mode: str = "node",
    ) -> None:
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
            edge_dim=int(edge_dim),
        )
        if hop_count not in (1, 2):
            raise ValueError("hop_count must be 1 or 2.")
        self.hop_count = int(hop_count)
        self.edge_exclusion = bool(edge_exclusion)
        if attention_variant not in ("v1", "v2"):
            raise ValueError("attention_variant must be 'v1' or 'v2'.")
        self.attention_variant = attention_variant
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.edge_dim = int(edge_dim)

        self.legacy_graph = LegacyTannerGraph.from_parity_check(graph.to_dense())
        self.vn_encoder = nn.Linear(vn_dim, hidden_dim)
        self.cn_encoder = nn.Linear(cn_dim, hidden_dim)
        nn.init.xavier_uniform_(self.vn_encoder.weight)
        nn.init.zeros_(self.vn_encoder.bias)
        nn.init.xavier_uniform_(self.cn_encoder.weight)
        nn.init.zeros_(self.cn_encoder.bias)

        # Separate parameters per direction. Every variant shares the same
        # parameterization, so toggling a flag isolates that rule alone.
        def block(edge_dim_for_pass: int) -> DirectedEdgeGATv2:
            return ConfigurableEdgeGATPass(
                hidden_dim,
                edge_dim_for_pass,
                self.heads,
                self.head_dim,
                dropout,
                attention_variant=self.attention_variant,
                edge_exclusion=self.edge_exclusion,
            )

        self.v_to_c = block(self.edge_dim)
        self.c_to_v = block(self.edge_dim)

        if self.hop_count == 2:
            self.v_to_v = block(TWO_HOP_EDGE_DIM)
            two_hop = graph.two_hop_index("vn")
            shared = graph.two_hop_shared_counts("vn")
            self.register_buffer("two_hop_source", two_hop[0], persistent=False)
            self.register_buffer("two_hop_target", two_hop[1], persistent=False)
            self.register_buffer(
                "two_hop_shared",
                (shared / shared.max().clamp_min(1.0)).view(1, -1, 1),
                persistent=False,
            )
            # Contiguous per-target slices for the attention softmax, computed
            # once here rather than per forward pass.
            self._two_hop_slices = TannerGraph.contiguous_target_slices(
                two_hop[1], graph.n_variables
            )
        else:
            self.v_to_v = None
            self._two_hop_slices = []

    # -------------------------------------------------------------- plumbing

    def extra_inputs(self, trace: Dict[str, Tensor]) -> Dict[str, Tensor]:
        missing = {"variable_to_check", "check_to_variable"} - set(trace)
        if missing:
            raise KeyError(
                "GraphTransformerLocal needs the BP messages in the trace; "
                f"missing {sorted(missing)}."
            )
        return {
            "edge_features": build_edge_features(
                trace["variable_to_check"], trace["check_to_variable"]
            )
        }

    def _two_hop_edge_features(self, vn_features: Tensor) -> Tensor:
        soft_bits = vn_features[..., VN_IDX_SOFT_FINAL].unsqueeze(-1)
        source = gather_nodes(soft_bits, self.two_hop_source)
        target = gather_nodes(soft_bits, self.two_hop_target)
        shared = self.two_hop_shared.expand(vn_features.shape[0], -1, -1)
        return torch.cat((shared, source * target), dim=-1)

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        **extra: Tensor,
    ) -> List[Tensor]:
        return self.encode_all(
            vn_features,
            cn_features,
            vn_to_cn_edge_index,
            cn_to_vn_edge_index,
            edge_features=edge_features,
            **extra,
        ).stages

    def encode_all(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        **extra: Tensor,
    ) -> EncodeResult:
        if edge_features is None:
            edge_features = vn_features.new_zeros(
                (vn_features.shape[0], self.graph.n_edges, self.edge_dim)
            )
        if edge_features.shape[-1] != self.edge_dim:
            raise ValueError(
                f"Expected {self.edge_dim} edge channels, got {edge_features.shape[-1]}."
            )

        vn_hidden = self.vn_encoder(vn_features)
        cn_hidden = self.cn_encoder(cn_features)
        graph = self.legacy_graph
        stages: List[Tensor] = []

        # Hop 1: variables into checks, then checks back into variables.
        cn_hidden = self.v_to_c(
            target_hidden=cn_hidden,
            source_hidden=vn_hidden,
            edge_features=edge_features,
            source_index=graph.edge_variables,
            target_index=graph.edge_checks,
            target_slices=graph.check_slices,
            n_targets=graph.n_checks,
            edge_order=graph.check_order,
            inverse_edge_order=graph.inverse_check_order,
        )
        vn_hidden = self.c_to_v(
            target_hidden=vn_hidden,
            source_hidden=cn_hidden,
            edge_features=edge_features,
            source_index=graph.edge_checks,
            target_index=graph.edge_variables,
            target_slices=graph.variable_slices,
            n_targets=graph.n_variables,
        )
        stages.append(vn_hidden)

        # Hop 2: variables attend directly to variables sharing a check.
        if self.v_to_v is not None:
            vn_hidden = self.v_to_v(
                target_hidden=vn_hidden,
                source_hidden=vn_hidden,
                edge_features=self._two_hop_edge_features(vn_features),
                source_index=self.two_hop_source,
                target_index=self.two_hop_target,
                target_slices=self._two_hop_slices,
                n_targets=self.graph.n_variables,
            )
            stages.append(vn_hidden)

        return EncodeResult(
            stages=stages,
            cn_hidden=cn_hidden,
            edge_features=edge_features,
        )

    # ------------------------------------------------------------------ cost

    def estimate_flops_per_frame(self) -> float:
        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        projected = self.heads * self.head_dim
        total = linear_flops_for(self.vn_encoder, n_vn) + linear_flops_for(
            self.cn_encoder, n_cn
        )

        def block_flops(n_targets: int, n_sources: int, edges: int, edge_dim: int) -> float:
            cost = linear_flops(self.hidden_dim, projected, n_targets)  # query
            cost += 2 * linear_flops(self.hidden_dim, projected, n_sources)  # key/value
            cost += 2 * linear_flops(edge_dim, projected, edges)
            cost += linear_flops(edge_dim, self.heads, edges)
            cost += linear_flops(projected, self.hidden_dim, n_targets)  # output
            cost += 2 * linear_flops(self.hidden_dim, 2 * self.hidden_dim, n_targets)
            cost += 6.0 * edges * projected  # attention arithmetic
            return cost

        total += block_flops(n_cn, n_vn, n_edges, self.edge_dim)
        total += block_flops(n_vn, n_cn, n_edges, self.edge_dim)
        if self.v_to_v is not None:
            two_hop_edges = int(self.two_hop_target.numel())
            total += block_flops(n_vn, n_vn, two_hop_edges, TWO_HOP_EDGE_DIM)

        return total + self.head_flops_per_frame()
