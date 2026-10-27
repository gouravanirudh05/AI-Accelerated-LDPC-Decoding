"""Correctors built on PyTorch Geometric layers.

These replace the hand-rolled message passing with the library's own
implementations, which matters most for the GATv1-vs-GATv2 comparison: PyG
ships both ``GATConv`` (v1, static attention) and ``GATv2Conv`` (v2, dynamic
attention), so that ablation no longer depends on my own scoring code.

The Tanner graph is bipartite, and PyG supports that natively: node features go
in as a tuple ``(x_src, x_dst)`` and every convolution is built with
``add_self_loops=False``, since a self-loop between a variable node and a check
node is not meaningful.

Layer choices and why:

``SAGEConv(aggr="mean")``
    The bipartite stand-in for GCN.  ``GCNConv``'s symmetric ``D^-1/2 A D^-1/2``
    normalization is defined for a homogeneous graph with self-loops and does
    not apply to a bipartite Tanner graph; ``SAGEConv`` gives exactly what the
    spec asks for -- degree-normalized (mean) neighbour aggregation plus a
    separate root weight, with independent parameters per direction.
``GATConv`` / ``GATv2Conv``
    Multi-head, edge-conditioned via ``edge_dim``, separate parameters per
    direction.
``TransformerConv``
    Attention with edge conditioning; the FFN, residual and LayerNorm around it
    are added here, since PyG provides the convolution rather than a whole
    transformer block.

Batching is handled by :mod:`~ldpc_corrector.pyg_graph`, which flattens the
``[B, nodes, channels]`` decoder layout into PyG's disjoint union and caches the
replicated ``edge_index``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, TransformerConv
from torch_geometric.utils import scatter, softmax

from ..bp_trace import EDGE_FEATURE_DIM, VN_IDX_SOFT_FINAL, build_edge_features
from ..pyg_graph import (
    BipartiteBatcher,
    flatten_edges,
    flatten_nodes,
    two_hop_batcher,
    unflatten_nodes,
)
from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector, EncodeResult

TWO_HOP_EDGE_DIM = 2


def _resolve_head_dim(hidden_dim: int, heads: int) -> int:
    if heads < 1:
        raise ValueError("heads must be positive.")
    if hidden_dim % heads:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})."
        )
    return hidden_dim // heads


class _PyGCorrectorBase(BaseCorrector):
    """Shared plumbing: batching, encoders, and BP edge features."""

    def __init__(self, *args, edge_dim: int = EDGE_FEATURE_DIM, **kwargs) -> None:
        super().__init__(*args, edge_dim=edge_dim, **kwargs)
        self.edge_dim = int(edge_dim)
        self.batcher = BipartiteBatcher(self.graph)
        self.vn_encoder = nn.Linear(self.vn_dim, self.hidden_dim)
        self.cn_encoder = nn.Linear(self.cn_dim, self.hidden_dim)
        for layer in (self.vn_encoder, self.cn_encoder):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def extra_inputs(self, trace: Dict[str, Tensor]) -> Dict[str, Tensor]:
        missing = {"variable_to_check", "check_to_variable"} - set(trace)
        if missing:
            raise KeyError(
                f"{type(self).__name__} needs the BP messages in the trace; "
                f"missing {sorted(missing)}."
            )
        return {
            "edge_features": build_edge_features(
                trace["variable_to_check"], trace["check_to_variable"]
            )
        }

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        """These correctors implement :meth:`encode_all`; this delegates to it."""
        return self.encode_all(
            vn_features,
            cn_features,
            vn_to_cn_edge_index,
            cn_to_vn_edge_index,
            **extra,
        ).stages

    def _default_edge_features(self, vn_features: Tensor) -> Tensor:
        return vn_features.new_zeros(
            (vn_features.shape[0], self.graph.n_edges, self.edge_dim)
        )


class PyGGCNCorrector(_PyGCorrectorBase):
    """Bipartite degree-normalized convolution via ``SAGEConv``."""

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
        # No edge conditioning in this variant, so edge_dim stays 0 and the
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
            raise ValueError("PyGGCNCorrector supports 1 or 2 layers.")
        self.num_layers = int(num_layers)

        # Separate parameters per direction and per layer.
        self.to_cn = nn.ModuleList(
            SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")
            for _ in range(self.num_layers)
        )
        self.to_vn = nn.ModuleList(
            SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")
            for _ in range(self.num_layers)
        )
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def encode_all(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> EncodeResult:
        batch = vn_features.shape[0]
        device = vn_features.device
        v2c = self.batcher.edge_index("vn_to_cn", batch, device)
        c2v = self.batcher.edge_index("cn_to_vn", batch, device)

        vn_hidden = flatten_nodes(self.activation(self.vn_encoder(vn_features)))
        cn_hidden = flatten_nodes(self.activation(self.cn_encoder(cn_features)))
        stages: List[Tensor] = []

        for layer in range(self.num_layers):
            cn_update = self.to_cn[layer]((vn_hidden, cn_hidden), v2c)
            cn_hidden = cn_hidden + self.drop(self.activation(cn_update))

            vn_update = self.to_vn[layer]((cn_hidden, vn_hidden), c2v)
            vn_hidden = vn_hidden + self.drop(self.activation(vn_update))
            stages.append(unflatten_nodes(vn_hidden, batch))

        return EncodeResult(stages=stages)

    def estimate_flops_per_frame(self) -> float:
        from .graph_ops import linear_flops

        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        total = linear_flops(self.vn_dim, self.hidden_dim, n_vn)
        total += linear_flops(self.cn_dim, self.hidden_dim, n_cn)
        for _ in range(self.num_layers):
            # SAGEConv: neighbour weight once per edge, root weight per target.
            total += linear_flops(self.hidden_dim, self.hidden_dim, n_edges)
            total += linear_flops(self.hidden_dim, self.hidden_dim, n_cn)
            total += linear_flops(self.hidden_dim, self.hidden_dim, n_edges)
            total += linear_flops(self.hidden_dim, self.hidden_dim, n_vn)
        return total + self.head_flops_per_frame()


class ExtrinsicGATConv(nn.Module):
    """GATv2-style attention with exclude-self-message aggregation.

    PyG has no extrinsic-aggregation convolution, and this pattern does not fit
    ``MessagePassing.propagate``: every edge has to see its *target's* total
    before subtracting its own contribution, whereas ``propagate`` gathers
    source features and reduces in one step.  It is therefore built directly on
    PyG's functional utilities, :func:`~torch_geometric.utils.softmax` and
    :func:`~torch_geometric.utils.scatter`.

    For each edge ``e`` into target ``t``::

        excl(e) = (sum_j a_j v_j - a_e v_e) / (1 - a_e)

    which is BP's extrinsic-information principle applied to attention.  The
    target's output is the mean of ``excl(e)`` over its incident edges.  A
    degree-1 target has no extrinsic information, so it contributes zero rather
    than dividing by ``1 - a_e = 0``.
    """

    EPSILON = 1e-4

    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        heads: int = 1,
        edge_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout

        src, dst = in_channels
        self.lin_src = nn.Linear(src, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(dst, heads * out_channels, bias=False)
        self.lin_value = nn.Linear(src, heads * out_channels, bias=False)
        self.lin_edge = (
            nn.Linear(edge_dim, heads * out_channels, bias=False) if edge_dim else None
        )
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:  # noqa: D102 - PyG convention
        for layer in (self.lin_src, self.lin_dst, self.lin_value, self.lin_edge):
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x_src, x_dst = x
        heads, channels = self.heads, self.out_channels
        source_index, target_index = edge_index[0], edge_index[1]
        num_targets = x_dst.shape[0]

        query = self.lin_dst(x_dst).view(-1, heads, channels)[target_index]
        key = self.lin_src(x_src).view(-1, heads, channels)[source_index]
        value = self.lin_value(x_src).view(-1, heads, channels)[source_index]

        score = query + key
        if self.lin_edge is not None and edge_attr is not None:
            edge = self.lin_edge(edge_attr).view(-1, heads, channels)
            score = score + edge
            value = value + edge

        alpha = (torch.nn.functional.leaky_relu(score, 0.2) * self.att).sum(dim=-1)
        alpha = softmax(alpha, target_index, num_nodes=num_targets)
        alpha = torch.nn.functional.dropout(
            alpha, p=self.dropout, training=self.training
        )

        weighted = alpha.unsqueeze(-1) * value
        totals = scatter(
            weighted, target_index, dim=0, dim_size=num_targets, reduce="sum"
        )

        remaining = (1.0 - alpha).unsqueeze(-1)
        exclusive = (totals[target_index] - weighted) / remaining.clamp_min(self.EPSILON)
        # No extrinsic information available (degree-1 target, or attention
        # saturated onto this one edge): contribute nothing rather than a
        # numerically meaningless quotient.
        exclusive = torch.where(
            remaining < self.EPSILON, torch.zeros_like(exclusive), exclusive
        )

        summed = scatter(
            exclusive, target_index, dim=0, dim_size=num_targets, reduce="sum"
        )
        counts = scatter(
            torch.ones_like(exclusive),
            target_index,
            dim=0,
            dim_size=num_targets,
            reduce="sum",
        )
        return (summed / counts.clamp_min(1.0)).view(-1, heads * channels)


class PyGGATCorrector(_PyGCorrectorBase):
    """Edge-conditioned multi-head attention via PyG's own GAT layers.

    ``attention_variant="v1"`` uses :class:`~torch_geometric.nn.GATConv`
    (static attention), ``"v2"`` uses :class:`~torch_geometric.nn.GATv2Conv`
    (dynamic attention).  Both come from the library, so the comparison is
    between the published formulations rather than between two of my own.

    ``edge_exclusion=True`` swaps in :class:`ExtrinsicGATConv`, the one layer
    PyG has no equivalent for.
    """

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        edge_dim: int = EDGE_FEATURE_DIM,
        attention_variant: str = "v2",
        edge_exclusion: bool = False,
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
            edge_dim=edge_dim,
        )
        if attention_variant not in ("v1", "v2"):
            raise ValueError("attention_variant must be 'v1' or 'v2'.")
        self.attention_variant = attention_variant
        self.edge_exclusion = bool(edge_exclusion)
        self.heads = int(heads)
        self.head_dim = _resolve_head_dim(hidden_dim, self.heads)

        def make_conv() -> nn.Module:
            common = dict(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=self.head_dim,
                heads=self.heads,
                edge_dim=self.edge_dim,
                dropout=dropout,
            )
            if self.edge_exclusion:
                return ExtrinsicGATConv(**common)
            layer = GATConv if attention_variant == "v1" else GATv2Conv
            # add_self_loops is invalid on a bipartite graph: a variable node
            # and a check node are different node types.
            return layer(**common, concat=True, add_self_loops=False)

        # Separate parameters per direction.
        self.to_cn = make_conv()
        self.to_vn = make_conv()
        self.norm_cn = nn.LayerNorm(hidden_dim)
        self.norm_vn = nn.LayerNorm(hidden_dim)

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
            edge_features = self._default_edge_features(vn_features)
        if edge_features.shape[-1] != self.edge_dim:
            raise ValueError(
                f"Expected {self.edge_dim} edge channels, got {edge_features.shape[-1]}."
            )

        batch = vn_features.shape[0]
        device = vn_features.device
        v2c = self.batcher.edge_index("vn_to_cn", batch, device)
        c2v = self.batcher.edge_index("cn_to_vn", batch, device)
        edge_attr = flatten_edges(edge_features)

        vn_hidden = flatten_nodes(self.vn_encoder(vn_features))
        cn_hidden = flatten_nodes(self.cn_encoder(cn_features))

        cn_hidden = self.norm_cn(
            cn_hidden + self.to_cn((vn_hidden, cn_hidden), v2c, edge_attr)
        )
        vn_hidden = self.norm_vn(
            vn_hidden + self.to_vn((cn_hidden, vn_hidden), c2v, edge_attr)
        )

        return EncodeResult(
            stages=[unflatten_nodes(vn_hidden, batch)],
            cn_hidden=unflatten_nodes(cn_hidden, batch),
            edge_features=edge_features,
        )

    def estimate_flops_per_frame(self) -> float:
        from .graph_ops import linear_flops

        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        projected = self.heads * self.head_dim
        total = linear_flops(self.vn_dim, self.hidden_dim, n_vn)
        total += linear_flops(self.cn_dim, self.hidden_dim, n_cn)
        for n_targets, n_sources in ((n_cn, n_vn), (n_vn, n_cn)):
            # Query per target, key/value per source, edge projection per edge.
            total += linear_flops(self.hidden_dim, projected, n_targets)
            total += 2 * linear_flops(self.hidden_dim, projected, n_sources)
            total += linear_flops(self.edge_dim, projected, n_edges)
            total += 4.0 * n_edges * projected  # scoring + weighted aggregation
        return total + self.head_flops_per_frame()


class PyGGraphTransformerLocal(_PyGCorrectorBase):
    """Local graph transformer built on ``TransformerConv``.

    ``TransformerConv`` supplies the edge-conditioned attention; the FFN,
    residual and LayerNorm that make it a transformer *block* are added here,
    since PyG provides convolutions rather than whole blocks.

    ``hop_count=2`` adds a variable-to-variable pass over pairs sharing a
    check.  That 2-hop index is materialized once here and cached by the
    batcher, never rebuilt inside a forward pass.
    """

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        edge_dim: int = EDGE_FEATURE_DIM,
        hop_count: int = 1,
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
            edge_dim=edge_dim,
        )
        if hop_count not in (1, 2):
            raise ValueError("hop_count must be 1 or 2.")
        self.hop_count = int(hop_count)
        self.heads = int(heads)
        self.head_dim = _resolve_head_dim(hidden_dim, self.heads)

        def block(edge_dim_for_pass: int) -> nn.Module:
            return TransformerConv(
                (hidden_dim, hidden_dim),
                self.head_dim,
                heads=self.heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim_for_pass,
            )

        self.to_cn = block(self.edge_dim)
        self.to_vn = block(self.edge_dim)
        self.norm_cn = nn.LayerNorm(hidden_dim)
        self.norm_vn = nn.LayerNorm(hidden_dim)
        self.ffn_vn = self._make_ffn(hidden_dim, dropout)
        self.ffn_norm_vn = nn.LayerNorm(hidden_dim)

        if self.hop_count == 2:
            self.to_vv = block(TWO_HOP_EDGE_DIM)
            self.norm_vv = nn.LayerNorm(hidden_dim)
            self.ffn_vv = self._make_ffn(hidden_dim, dropout)
            self.ffn_norm_vv = nn.LayerNorm(hidden_dim)
            self.two_hop = two_hop_batcher(graph, "vn")
            shared = graph.two_hop_shared_counts("vn")
            self.register_buffer(
                "two_hop_shared",
                (shared / shared.max().clamp_min(1.0)).view(-1, 1),
                persistent=False,
            )
            index = graph.two_hop_index("vn")
            self.register_buffer("two_hop_source", index[0], persistent=False)
            self.register_buffer("two_hop_target", index[1], persistent=False)
        else:
            self.to_vv = None
            self.two_hop = None

    @staticmethod
    def _make_ffn(hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _two_hop_edge_attr(self, vn_features: Tensor) -> Tensor:
        """``[B * E2, 2]``: shared-check count, and endpoint soft-bit agreement.

        Two variables that share checks but disagree on their current soft bit
        are where residual errors concentrate.
        """
        batch = vn_features.shape[0]
        soft = vn_features[..., VN_IDX_SOFT_FINAL]  # [B, n]
        source = soft[:, self.two_hop_source]
        target = soft[:, self.two_hop_target]
        agreement = (source * target).reshape(-1, 1)
        shared = self.two_hop_shared.expand(-1, 1).repeat(batch, 1)
        return torch.cat((shared, agreement), dim=-1)

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
            edge_features = self._default_edge_features(vn_features)
        if edge_features.shape[-1] != self.edge_dim:
            raise ValueError(
                f"Expected {self.edge_dim} edge channels, got {edge_features.shape[-1]}."
            )

        batch = vn_features.shape[0]
        device = vn_features.device
        v2c = self.batcher.edge_index("vn_to_cn", batch, device)
        c2v = self.batcher.edge_index("cn_to_vn", batch, device)
        edge_attr = flatten_edges(edge_features)

        vn_hidden = flatten_nodes(self.vn_encoder(vn_features))
        cn_hidden = flatten_nodes(self.cn_encoder(cn_features))
        stages: List[Tensor] = []

        # Hop 1: variables into checks, then checks back into variables.
        cn_hidden = self.norm_cn(
            cn_hidden + self.to_cn((vn_hidden, cn_hidden), v2c, edge_attr)
        )
        vn_hidden = self.norm_vn(
            vn_hidden + self.to_vn((cn_hidden, vn_hidden), c2v, edge_attr)
        )
        vn_hidden = self.ffn_norm_vn(vn_hidden + self.ffn_vn(vn_hidden))
        stages.append(unflatten_nodes(vn_hidden, batch))

        # Hop 2: variables attend to variables sharing a check.
        if self.to_vv is not None:
            vv = self.two_hop.edge_index("two_hop", batch, device)
            vv_attr = self._two_hop_edge_attr(vn_features)
            vn_hidden = self.norm_vv(
                vn_hidden + self.to_vv((vn_hidden, vn_hidden), vv, vv_attr)
            )
            vn_hidden = self.ffn_norm_vv(vn_hidden + self.ffn_vv(vn_hidden))
            stages.append(unflatten_nodes(vn_hidden, batch))

        return EncodeResult(
            stages=stages,
            cn_hidden=unflatten_nodes(cn_hidden, batch),
            edge_features=edge_features,
        )

    def estimate_flops_per_frame(self) -> float:
        from .graph_ops import linear_flops

        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        projected = self.heads * self.head_dim
        total = linear_flops(self.vn_dim, self.hidden_dim, n_vn)
        total += linear_flops(self.cn_dim, self.hidden_dim, n_cn)

        def pass_flops(n_targets: int, n_sources: int, edges: int, edge_dim: int) -> float:
            cost = linear_flops(self.hidden_dim, projected, n_targets)
            cost += 2 * linear_flops(self.hidden_dim, projected, n_sources)
            cost += linear_flops(edge_dim, projected, edges)
            cost += 4.0 * edges * projected
            return cost

        total += pass_flops(n_cn, n_vn, n_edges, self.edge_dim)
        total += pass_flops(n_vn, n_cn, n_edges, self.edge_dim)
        total += 2 * linear_flops(self.hidden_dim, 2 * self.hidden_dim, n_vn)
        if self.to_vv is not None:
            two_hop_edges = int(self.two_hop_target.numel())
            total += pass_flops(n_vn, n_vn, two_hop_edges, TWO_HOP_EDGE_DIM)
            total += 2 * linear_flops(self.hidden_dim, 2 * self.hidden_dim, n_vn)
        return total + self.head_flops_per_frame()
