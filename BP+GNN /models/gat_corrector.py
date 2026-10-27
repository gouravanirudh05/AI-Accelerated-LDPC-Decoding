"""GAT residual corrector -- the primary baseline.

This wraps the Phase 0-2 ``BidirectionalEdgeGATv2`` block (multi-head,
edge-conditioned GATv2 with separate parameters for the V->C and C->V passes)
in the common :class:`BaseCorrector` interface, so the Aug-26 architecture and
its results stay directly comparable.

Two config flags from the spec are ablations, not decisions:

``use_gate``
    learned per-bit gate vs a fixed scalar lambda -- handled entirely by
    :class:`BaseCorrector`.
``edge_exclusion``
    standard neighbour aggregation vs *exclude-self-message* aggregation.  The
    exclusion path reuses the identical projections, attention logits, and
    output/FFN parameters as the standard path; only the aggregation step
    differs, so the ablation isolates exactly that.
``attention_variant``
    ``"v2"`` (default) or ``"v1"``.  GATv2 applies the nonlinearity *before*
    the attention vector, ``a^T LeakyReLU(W[h_t || h_s || e])``, which makes
    the attention *dynamic*: the ranking of a node's neighbours depends on the
    query.  GATv1 scores first and applies the nonlinearity after,
    ``LeakyReLU(a_t^T Wh_t + a_s^T Wh_s + a_e^T We)``, which is *static* -- the
    neighbour ranking is the same for every query node.  On a Tanner graph
    that matters: static attention cannot express "this check matters to this
    variable but not to that one", which is exactly the query-dependence BP's
    extrinsic messages rely on.

The two flags are orthogonal and compose, so all four combinations are
reachable from config.  Every combination shares one parameter set (v1 adds
only its two extra attention vectors), so an ablation isolates the rule under
test rather than a whole reimplementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._compat import BidirectionalEdgeGATv2, DirectedEdgeGATv2, LegacyTannerGraph
from ..bp_trace import EDGE_FEATURE_DIM, build_edge_features
from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector, EncodeResult
from .graph_ops import linear_flops, linear_flops_for


class ConfigurableEdgeGATPass(DirectedEdgeGATv2):
    """One directed pass with selectable attention variant and aggregation.

    ``attention_variant``
        ``"v2"`` reuses the inherited dynamic-attention scoring.  ``"v1"``
        scores with per-side attention vectors and applies ``LeakyReLU``
        afterwards, giving static attention.

    ``edge_exclusion``
        For each edge ``e`` into target ``t``, the extrinsic aggregate leaves
        out ``e``'s own message and renormalizes the remaining attention::

            excl(e) = (sum_j a_j v_j - a_e v_e) / (1 - a_e)

        which is BP's extrinsic-information principle applied to attention.
        The target's hidden update is the mean of ``excl(e)`` over its incident
        edges.  A degree-1 target has no extrinsic information at all, so its
        aggregate is zero rather than a division by ``1 - a_e = 0``.

    With ``v2`` and no exclusion this delegates straight to the Phase 0-2
    implementation, so the default path is the legacy code exactly.  All shared
    parameters are inherited, so switching either flag changes the rule under
    test and nothing else.
    """

    EPSILON = 1e-4

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
        attention_variant: str = "v2",
        edge_exclusion: bool = False,
    ) -> None:
        super().__init__(hidden_dim, edge_dim, heads, head_dim, dropout)
        if attention_variant not in ("v1", "v2"):
            raise ValueError("attention_variant must be 'v1' or 'v2'.")
        self.attention_variant = attention_variant
        self.edge_exclusion = bool(edge_exclusion)

        if attention_variant == "v1":
            # GATv1 needs one attention vector per side; the inherited
            # ``attn_vector`` serves as the target-side vector.
            self.attn_vector_source = nn.Parameter(torch.empty(heads, head_dim))
            self.attn_vector_edge = nn.Parameter(torch.empty(heads, head_dim))
            nn.init.xavier_uniform_(self.attn_vector_source.unsqueeze(0))
            nn.init.xavier_uniform_(self.attn_vector_edge.unsqueeze(0))

    @property
    def is_legacy_path(self) -> bool:
        return self.attention_variant == "v2" and not self.edge_exclusion

    def _attention_logits(
        self,
        target_edges: Tensor,
        source_edges: Tensor,
        edge_context: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """Pre-softmax logits ``[B, E, heads]`` for the selected variant."""
        batch_size = target_edges.shape[0]
        scale = self.head_dim**0.5

        if self.attention_variant == "v2":
            # Nonlinearity first, then the shared attention vector: dynamic.
            compatibility = F.leaky_relu(
                target_edges + source_edges + edge_context, negative_slope=0.2
            )
            logits = (compatibility * self.attn_vector).sum(dim=-1) / scale
        else:
            # Score first, nonlinearity after: static attention.
            score = (
                (target_edges * self.attn_vector).sum(dim=-1)
                + (source_edges * self.attn_vector_source).sum(dim=-1)
                + (edge_context * self.attn_vector_edge).sum(dim=-1)
            ) / scale
            logits = F.leaky_relu(score, negative_slope=0.2)

        return logits + self.edge_bias(edge_features).view(batch_size, -1, self.heads)

    def forward(
        self,
        target_hidden: Tensor,
        source_hidden: Tensor,
        edge_features: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        target_slices: List[Tuple[int, int]],
        n_targets: int,
        edge_order: Optional[Tensor] = None,
        inverse_edge_order: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if self.is_legacy_path:
            return super().forward(
                target_hidden=target_hidden,
                source_hidden=source_hidden,
                edge_features=edge_features,
                source_index=source_index,
                target_index=target_index,
                target_slices=target_slices,
                n_targets=n_targets,
                edge_order=edge_order,
                inverse_edge_order=inverse_edge_order,
                return_attention=return_attention,
            )

        batch_size = target_hidden.shape[0]
        device = target_hidden.device
        source_index = source_index.to(device)
        target_index = target_index.to(device)

        if edge_order is not None:
            edge_order = edge_order.to(device)
            source_index = source_index.index_select(0, edge_order)
            target_index = target_index.index_select(0, edge_order)
            edge_features = edge_features.index_select(1, edge_order)

        # --- same projections as the standard block ---
        target_projection = self.query(target_hidden).view(
            batch_size, -1, self.heads, self.head_dim
        )
        source_projection = self.key(source_hidden).view(
            batch_size, -1, self.heads, self.head_dim
        )
        source_values = self.value(source_hidden).view(
            batch_size, -1, self.heads, self.head_dim
        )
        target_edges = target_projection.index_select(1, target_index)
        source_edges = source_projection.index_select(1, source_index)
        values = source_values.index_select(1, source_index)
        edge_context = self.edge_attention(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )
        values = values + self.edge_value(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )
        logits = self._attention_logits(
            target_edges, source_edges, edge_context, edge_features
        )
        attention = torch.cat(
            [F.softmax(logits[:, start:end], dim=1) for start, end in target_slices],
            dim=1,
        )
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # --- aggregation: extrinsic if edge_exclusion, else standard ---
        weighted = attention.unsqueeze(-1) * values
        scatter_index = target_index.view(1, -1, 1, 1).expand(
            batch_size, -1, self.heads, self.head_dim
        )
        totals = torch.zeros(
            batch_size,
            n_targets,
            self.heads,
            self.head_dim,
            device=device,
            dtype=weighted.dtype,
        )
        totals.scatter_add_(1, scatter_index, weighted)

        gathered_totals = totals.index_select(1, target_index)
        remaining = (1.0 - attention).unsqueeze(-1)
        exclusive = (gathered_totals - weighted) / remaining.clamp_min(self.EPSILON)
        # No extrinsic information available (degree-1 target, or attention
        # saturated onto this one edge): contribute nothing rather than a
        # numerically meaningless quotient.
        exclusive = torch.where(
            remaining < self.EPSILON, torch.zeros_like(exclusive), exclusive
        )

        aggregated = torch.zeros_like(totals)
        aggregated.scatter_add_(1, scatter_index, exclusive)
        degrees = torch.zeros(
            batch_size, n_targets, self.heads, self.head_dim, device=device
        )
        degrees.scatter_add_(1, scatter_index, torch.ones_like(exclusive))
        aggregated = aggregated / degrees.clamp_min(1.0)

        message = self.output(aggregated.reshape(batch_size, n_targets, -1))
        updated = self.norm_message(
            target_hidden + F.dropout(message, p=self.dropout, training=self.training)
        )
        updated = self.norm_feed_forward(updated + self.feed_forward(updated))

        if return_attention:
            if inverse_edge_order is not None:
                attention = attention.index_select(1, inverse_edge_order.to(device))
            return updated, attention
        return updated


class GATCorrector(BaseCorrector):
    """BP(k) + one bidirectional edge-aware GATv2 refinement block."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        edge_dim: int = EDGE_FEATURE_DIM,
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        edge_exclusion: bool = False,
        attention_variant: str = "v2",
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
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.edge_dim = int(edge_dim)
        self.edge_exclusion = bool(edge_exclusion)
        if attention_variant not in ("v1", "v2"):
            raise ValueError("attention_variant must be 'v1' or 'v2'.")
        self.attention_variant = attention_variant

        # The legacy block needs the Phase 0-2 graph object; edge ordering is
        # identical (asserted by tests/test_tanner_graph.py), so message and
        # feature tensors index-align between the two representations.
        self.legacy_graph = LegacyTannerGraph.from_parity_check(graph.to_dense())
        self.block = BidirectionalEdgeGATv2(
            self.legacy_graph,
            variable_dim=vn_dim,
            check_dim=cn_dim,
            edge_dim=self.edge_dim,
            hidden_dim=hidden_dim,
            heads=self.heads,
            head_dim=self.head_dim,
            dropout=dropout,
        )
        if self.edge_exclusion or self.attention_variant != "v2":
            # Swap in the configurable pass, carrying over the standard
            # block's weights so only the selected rule differs. v1 adds two
            # attention vectors that the standard state dict does not have,
            # hence strict=False.
            for name in ("v_to_c", "c_to_v"):
                standard = getattr(self.block, name)
                custom = ConfigurableEdgeGATPass(
                    hidden_dim,
                    self.edge_dim,
                    self.heads,
                    self.head_dim,
                    dropout,
                    attention_variant=self.attention_variant,
                    edge_exclusion=self.edge_exclusion,
                )
                custom.load_state_dict(standard.state_dict(), strict=False)
                setattr(self.block, name, custom)

    # -------------------------------------------------------------- plumbing

    def extra_inputs(self, trace: Dict[str, Tensor]) -> Dict[str, Tensor]:
        missing = {"variable_to_check", "check_to_variable"} - set(trace)
        if missing:
            raise KeyError(
                f"GATCorrector needs the BP messages in the trace; missing {sorted(missing)}."
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
        edge_features: Optional[Tensor] = None,
        **extra: Tensor,
    ) -> List[Tensor]:
        if edge_features is None:
            # Standalone / unit-test use: fall back to zero edge conditioning.
            edge_features = vn_features.new_zeros(
                (vn_features.shape[0], self.graph.n_edges, self.edge_dim)
            )
        if edge_features.shape[-1] != self.edge_dim:
            raise ValueError(
                f"Expected {self.edge_dim} edge channels, got {edge_features.shape[-1]}."
            )

        vn_hidden, _, _ = self.block(
            vn_features, cn_features, edge_features, return_attention=False
        )
        return [vn_hidden]

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
        vn_hidden, cn_hidden, _ = self.block(
            vn_features, cn_features, edge_features, return_attention=False
        )
        return EncodeResult(
            stages=[vn_hidden],
            cn_hidden=cn_hidden,
            edge_features=edge_features,
        )

    def attention_maps(
        self, vn_features: Tensor, cn_features: Tensor, edge_features: Tensor
    ) -> Dict[str, Tensor]:
        """Per-direction attention, for inspection rather than training."""
        _, _, attention = self.block(
            vn_features, cn_features, edge_features, return_attention=True
        )
        return attention

    # ------------------------------------------------------------------ cost

    def estimate_flops_per_frame(self) -> float:
        n_vn, n_cn, n_edges = (
            self.graph.n_variables,
            self.graph.n_checks,
            self.graph.n_edges,
        )
        projected = self.heads * self.head_dim

        total = linear_flops_for(self.block.variable_encoder, n_vn)
        total += linear_flops_for(self.block.check_encoder, n_cn)

        for direction, n_targets, n_sources in (
            (self.block.v_to_c, n_cn, n_vn),
            (self.block.c_to_v, n_vn, n_cn),
        ):
            # Node projections once per node; edge projections once per edge.
            total += linear_flops(self.hidden_dim, projected, n_targets)  # query
            total += 2 * linear_flops(self.hidden_dim, projected, n_sources)  # key, value
            total += 2 * linear_flops(self.edge_dim, projected, n_edges)
            total += linear_flops(self.edge_dim, self.heads, n_edges)
            total += linear_flops(projected, self.hidden_dim, n_targets)  # output
            # FFN: hidden -> 2*hidden -> hidden, once per target node.
            total += 2 * linear_flops(self.hidden_dim, 2 * self.hidden_dim, n_targets)
            # Attention arithmetic scales with edges * heads * head_dim.
            total += 6.0 * n_edges * projected

        return total + self.head_flops_per_frame()
