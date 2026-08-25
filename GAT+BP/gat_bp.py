"""Hybrid differentiable BP + edge-aware GAT decoder.

The BP stage is a conventional sum-product decoder on the Tanner graph.  The
GAT stage consumes the channel LLR, BP posterior, and BP edge messages and
learns a residual correction to the BP posterior.  No PyTorch-Geometric
dependency is required; the graph is represented by the non-zero entries of
the parity-check matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TannerGraph:
    """Static edge layout for a binary parity-check matrix."""

    n_variables: int
    n_checks: int
    edge_variables: Tensor
    edge_checks: Tensor
    check_order: Tensor
    inverse_check_order: Tensor
    variable_slices: List[Tuple[int, int]]
    check_slices: List[Tuple[int, int]]
    variable_degrees: Tensor
    check_degrees: Tensor
    variable_positional: Tensor
    check_positional: Tensor

    @classmethod
    def from_parity_check(
        cls,
        parity_check: np.ndarray | Tensor,
        positional_dim: int = 0,
    ) -> "TannerGraph":
        if isinstance(parity_check, Tensor):
            h = parity_check.detach().cpu().numpy()
        else:
            h = np.asarray(parity_check)
        if h.ndim != 2:
            raise ValueError("The parity-check matrix must be two-dimensional.")
        if positional_dim < 0:
            raise ValueError("positional_dim must be non-negative.")
        h = (h != 0).astype(np.uint8)
        n_checks, n_variables = h.shape
        raw_checks, raw_variables = np.nonzero(h)
        if len(raw_checks) == 0:
            raise ValueError("The parity-check matrix has no Tanner-graph edges.")
        if np.any(np.bincount(raw_variables, minlength=n_variables) == 0):
            raise ValueError("Every variable node must participate in a check.")
        if np.any(np.bincount(raw_checks, minlength=n_checks) == 0):
            raise ValueError("Every check node must contain at least one variable.")

        # Keep the canonical edge order grouped by variable for GAT aggregation.
        variable_order = np.lexsort((raw_checks, raw_variables))
        edge_variables = raw_variables[variable_order]
        edge_checks = raw_checks[variable_order]

        # BP check updates use the same edges grouped by check.  These two
        # permutations avoid scattering differentiable messages in-place.
        check_order = np.lexsort((edge_variables, edge_checks))
        inverse_check_order = np.argsort(check_order)

        variable_slices = _contiguous_slices(edge_variables, n_variables)
        check_sorted_checks = edge_checks[check_order]
        check_slices = _contiguous_slices(check_sorted_checks, n_checks)
        variable_degrees = np.bincount(edge_variables, minlength=n_variables).astype(np.float32)
        check_degrees = np.bincount(edge_checks, minlength=n_checks).astype(np.float32)
        variable_positional, check_positional = _spectral_positional_encoding(
            edge_variables,
            edge_checks,
            n_variables,
            n_checks,
            positional_dim,
        )

        return cls(
            n_variables=n_variables,
            n_checks=n_checks,
            edge_variables=torch.as_tensor(edge_variables, dtype=torch.long),
            edge_checks=torch.as_tensor(edge_checks, dtype=torch.long),
            check_order=torch.as_tensor(check_order, dtype=torch.long),
            inverse_check_order=torch.as_tensor(inverse_check_order, dtype=torch.long),
            variable_slices=variable_slices,
            check_slices=check_slices,
            variable_degrees=torch.as_tensor(variable_degrees, dtype=torch.float32),
            check_degrees=torch.as_tensor(check_degrees, dtype=torch.float32),
            variable_positional=torch.as_tensor(variable_positional, dtype=torch.float32),
            check_positional=torch.as_tensor(check_positional, dtype=torch.float32),
        )

    @property
    def n_edges(self) -> int:
        return int(self.edge_variables.numel())

    def mean_by_check(self, edge_values: Tensor) -> Tensor:
        """Average edge values into check-node features."""
        ordered = edge_values.index_select(1, self.check_order.to(edge_values.device))
        chunks = [ordered[:, start:end].mean(dim=1) for start, end in self.check_slices]
        return torch.stack(chunks, dim=1)

    def product_by_check(self, edge_values: Tensor) -> Tensor:
        """Multiply scalar edge values into check-node features."""
        ordered = edge_values.index_select(1, self.check_order.to(edge_values.device))
        chunks = [ordered[:, start:end].prod(dim=1) for start, end in self.check_slices]
        return torch.stack(chunks, dim=1)


def _contiguous_slices(labels: np.ndarray, count: int) -> List[Tuple[int, int]]:
    slices: List[Tuple[int, int]] = []
    start = 0
    for label in range(count):
        positions = np.flatnonzero(labels == label)
        if len(positions) == 0:
            raise ValueError(f"Node {label} has no Tanner-graph edges.")
        end = int(positions[-1]) + 1
        slices.append((start, end))
        start = end
    if start != len(labels):
        raise RuntimeError("Tanner-graph edge layout is not contiguous.")
    return slices


def _spectral_positional_encoding(
    edge_variables: np.ndarray,
    edge_checks: np.ndarray,
    n_variables: int,
    n_checks: int,
    positional_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic Laplacian positional encodings for Tanner nodes.

    The first eigenvector of the normalized Laplacian is the constant mode and
    is discarded. Eigenvector signs are fixed using the largest-magnitude
    entry so that the encoding is reproducible across runs.
    """
    if positional_dim == 0:
        return (
            np.zeros((n_variables, 0), dtype=np.float32),
            np.zeros((n_checks, 0), dtype=np.float32),
        )

    total_nodes = n_variables + n_checks
    adjacency = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    check_nodes = n_variables + edge_checks
    adjacency[edge_variables, check_nodes] = 1.0
    adjacency[check_nodes, edge_variables] = 1.0
    degrees = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degrees, 1.0))
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(total_nodes, dtype=np.float64) - normalized_adjacency
    _, eigenvectors = np.linalg.eigh(laplacian)
    usable_dim = min(positional_dim, max(total_nodes - 1, 0))
    encoding = eigenvectors[:, 1 : 1 + usable_dim].copy()

    for column in range(encoding.shape[1]):
        pivot = int(np.argmax(np.abs(encoding[:, column])))
        if encoding[pivot, column] < 0.0:
            encoding[:, column] *= -1.0

    # Eigenvectors have unit norm over all Tanner nodes. Scaling by sqrt(N)
    # keeps their per-feature magnitude comparable to normalized learned inputs.
    encoding *= np.sqrt(float(total_nodes))
    if usable_dim < positional_dim:
        encoding = np.pad(
            encoding,
            ((0, 0), (0, positional_dim - usable_dim)),
            mode="constant",
        )
    return (
        encoding[:n_variables].astype(np.float32),
        encoding[n_variables:].astype(np.float32),
    )


def _atanh_clamped(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SumProductBP(nn.Module):
    """Differentiable flooding-schedule sum-product BP on a Tanner graph."""

    def __init__(self, graph: TannerGraph, iterations: int = 5, damping: float = 0.0):
        super().__init__()
        if iterations < 0:
            raise ValueError("BP iterations must be non-negative.")
        if not 0.0 <= damping < 1.0:
            raise ValueError("BP damping must be in [0, 1).")
        self.graph = graph
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.register_buffer("edge_variables", graph.edge_variables)
        self.register_buffer("check_order", graph.check_order)
        self.register_buffer("inverse_check_order", graph.inverse_check_order)

    def _check_to_variable(self, variable_to_check: Tensor) -> Tensor:
        # Work in check-grouped order for the check update.
        ordered = variable_to_check.index_select(1, self.check_order)
        tanh_messages = torch.tanh(0.5 * ordered)
        check_messages: List[Tensor] = []

        for start, end in self.graph.check_slices:
            local = tanh_messages[:, start:end]
            degree = end - start
            if degree == 1:
                excluded = torch.ones_like(local)
            else:
                left = torch.cumprod(local, dim=1)
                right = torch.cumprod(local.flip(dims=(1,)), dim=1).flip(dims=(1,))
                first = right[:, 1:2]
                middle = left[:, :-2] * right[:, 2:]
                last = left[:, -2:-1]
                excluded = torch.cat((first, middle, last), dim=1)
            check_messages.append(2.0 * _atanh_clamped(excluded))

        ordered_messages = torch.cat(check_messages, dim=1)
        return ordered_messages.index_select(1, self.inverse_check_order)

    def step(
        self,
        channel_llr: Tensor,
        variable_to_check: Tensor,
        check_to_variable: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run one differentiable flooding-schedule BP iteration."""
        new_check_to_variable = self._check_to_variable(variable_to_check)
        if self.damping:
            new_check_to_variable = (
                self.damping * check_to_variable
                + (1.0 - self.damping) * new_check_to_variable
            )

        incoming = torch.zeros_like(channel_llr)
        incoming.scatter_add_(
            1,
            self.edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
            new_check_to_variable,
        )
        new_variable_to_check = (
            channel_llr.index_select(1, self.edge_variables)
            + incoming.index_select(1, self.edge_variables)
            - new_check_to_variable
        )

        posterior = channel_llr.clone()
        posterior.scatter_add_(
            1,
            self.edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
            new_check_to_variable,
        )
        return posterior, new_variable_to_check, new_check_to_variable

    def forward(
        self, channel_llr: Tensor, return_messages: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        if channel_llr.ndim != 2 or channel_llr.shape[1] != self.graph.n_variables:
            raise ValueError(
                f"channel_llr must have shape [batch, {self.graph.n_variables}]."
            )

        variable_to_check = channel_llr.index_select(1, self.edge_variables)
        check_to_variable = torch.zeros_like(variable_to_check)

        posterior = channel_llr
        for _ in range(self.iterations):
            posterior, variable_to_check, check_to_variable = self.step(
                channel_llr,
                variable_to_check,
                check_to_variable,
            )
        if return_messages:
            return posterior, variable_to_check, check_to_variable
        return posterior


class EdgeAwareGAT(nn.Module):
    """Edge-conditioned multi-head GATv2-style check-to-variable update.

    Attention is normalized over each variable node's incident checks. The
    edge embedding influences both the attention compatibility score and the
    value message, so BP reliability is used rather than only node states.
    """

    def __init__(
        self,
        graph: TannerGraph,
        variable_dim: int = 4,
        check_dim: int = 5,
        edge_dim: int = 4,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if heads < 1 or head_dim < 1:
            raise ValueError("heads and head_dim must be positive.")
        self.graph = graph
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = float(dropout)
        projected_dim = heads * head_dim

        # Encode node features before splitting into attention heads. This
        # gives the residual block a common hidden space for variable/check
        # states while retaining the bipartite graph direction.
        self.variable_encoder = nn.Linear(variable_dim, hidden_dim)
        self.check_encoder = nn.Linear(check_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.key = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.value = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.edge_attention = nn.Linear(edge_dim, projected_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, projected_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.attn_vector = nn.Parameter(torch.empty(heads, head_dim))
        self.output = nn.Linear(projected_dim, hidden_dim)
        self.norm_message = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_feed_forward = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.variable_encoder.weight)
        nn.init.zeros_(self.variable_encoder.bias)
        nn.init.xavier_uniform_(self.check_encoder.weight)
        nn.init.zeros_(self.check_encoder.bias)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.edge_attention.weight)
        nn.init.xavier_uniform_(self.edge_value.weight)
        nn.init.xavier_uniform_(self.edge_bias.weight)
        nn.init.xavier_uniform_(self.attn_vector.unsqueeze(0))
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        variable_features: Tensor,
        check_features: Tensor,
        edge_features: Tensor,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        batch_size = variable_features.shape[0]
        n_checks = check_features.shape[1]
        variable_hidden = self.variable_encoder(variable_features)
        check_hidden = self.check_encoder(check_features)
        projected_variables = self.query(variable_hidden).view(
            batch_size, -1, self.heads, self.head_dim
        )
        projected_checks = self.key(check_hidden).view(
            batch_size, n_checks, self.heads, self.head_dim
        )
        values = self.value(check_hidden).view(
            batch_size, n_checks, self.heads, self.head_dim
        )
        edge_context = self.edge_attention(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )
        edge_values = self.edge_value(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )

        edge_variables = self.graph.edge_variables.to(variable_features.device)
        edge_checks = self.graph.edge_checks.to(variable_features.device)
        query_edges = projected_variables.index_select(1, edge_variables)
        key_edges = projected_checks.index_select(1, edge_checks)
        value_edges = values.index_select(1, edge_checks) + edge_values
        compatibility = F.leaky_relu(
            query_edges + key_edges + edge_context,
            negative_slope=0.2,
        )
        logits = (
            compatibility * self.attn_vector
        ).sum(dim=-1) / (self.head_dim**0.5)
        logits = logits + self.edge_bias(edge_features).view(
            batch_size, -1, self.heads
        )

        # Each variable node normalizes over its incident check nodes.
        attention = torch.cat(
            [F.softmax(logits[:, start:end], dim=1) for start, end in self.graph.variable_slices],
            dim=1,
        )
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        aggregated = torch.stack(
            [
                (attention[:, start:end].unsqueeze(-1) * value_edges[:, start:end]).sum(dim=1)
                for start, end in self.graph.variable_slices
            ],
            dim=1,
        ).reshape(batch_size, self.graph.n_variables, -1)
        message = self.output(aggregated)
        updated = self.norm_message(variable_hidden + F.dropout(
            message, p=self.dropout, training=self.training
        ))
        updated = self.norm_feed_forward(updated + self.feed_forward(updated))
        if return_attention:
            return updated, attention
        return updated


class DirectedEdgeGATv2(nn.Module):
    """One sparse, edge-conditioned GATv2 message-passing direction."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if heads < 1 or head_dim < 1:
            raise ValueError("heads and head_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = float(dropout)
        projected_dim = heads * head_dim

        self.query = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.key = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.value = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.edge_attention = nn.Linear(edge_dim, projected_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, projected_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.attn_vector = nn.Parameter(torch.empty(heads, head_dim))
        self.output = nn.Linear(projected_dim, hidden_dim)
        self.norm_message = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_feed_forward = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.query,
            self.key,
            self.value,
            self.edge_attention,
            self.edge_value,
            self.edge_bias,
            self.output,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.attn_vector.unsqueeze(0))

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
        batch_size = target_hidden.shape[0]
        device = target_hidden.device
        source_index = source_index.to(device)
        target_index = target_index.to(device)

        if edge_order is not None:
            edge_order = edge_order.to(device)
            source_index = source_index.index_select(0, edge_order)
            target_index = target_index.index_select(0, edge_order)
            edge_features = edge_features.index_select(1, edge_order)

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

        compatibility = F.leaky_relu(
            target_edges + source_edges + edge_context,
            negative_slope=0.2,
        )
        logits = (
            compatibility * self.attn_vector
        ).sum(dim=-1) / (self.head_dim**0.5)
        logits = logits + self.edge_bias(edge_features).view(
            batch_size, -1, self.heads
        )
        attention = torch.cat(
            [F.softmax(logits[:, start:end], dim=1) for start, end in target_slices],
            dim=1,
        )
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        weighted_values = attention.unsqueeze(-1) * values
        scatter_index = target_index.view(1, -1, 1, 1).expand(
            batch_size, -1, self.heads, self.head_dim
        )
        aggregated = torch.zeros(
            batch_size,
            n_targets,
            self.heads,
            self.head_dim,
            device=device,
            dtype=weighted_values.dtype,
        )
        aggregated.scatter_add_(1, scatter_index, weighted_values)
        message = self.output(aggregated.reshape(batch_size, n_targets, -1))
        updated = self.norm_message(
            target_hidden
            + F.dropout(message, p=self.dropout, training=self.training)
        )
        updated = self.norm_feed_forward(updated + self.feed_forward(updated))

        if return_attention:
            if inverse_edge_order is not None:
                attention = attention.index_select(1, inverse_edge_order.to(device))
            return updated, attention
        return updated


class BidirectionalEdgeGATv2(nn.Module):
    """One efficient V→C→V edge-aware Tanner-graph refinement block."""

    def __init__(
        self,
        graph: TannerGraph,
        variable_dim: int,
        check_dim: int,
        edge_dim: int = 6,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.graph = graph
        self.variable_encoder = nn.Linear(variable_dim, hidden_dim)
        self.check_encoder = nn.Linear(check_dim, hidden_dim)
        self.v_to_c = DirectedEdgeGATv2(
            hidden_dim, edge_dim, heads, head_dim, dropout
        )
        self.c_to_v = DirectedEdgeGATv2(
            hidden_dim, edge_dim, heads, head_dim, dropout
        )
        nn.init.xavier_uniform_(self.variable_encoder.weight)
        nn.init.zeros_(self.variable_encoder.bias)
        nn.init.xavier_uniform_(self.check_encoder.weight)
        nn.init.zeros_(self.check_encoder.bias)

    def forward(
        self,
        variable_features: Tensor,
        check_features: Tensor,
        edge_features: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        variable_hidden = self.variable_encoder(variable_features)
        check_hidden = self.check_encoder(check_features)
        attention: Dict[str, Tensor] = {}
        if return_attention:
            refined_check, v_to_c_attention = self.v_to_c(
                target_hidden=check_hidden,
                source_hidden=variable_hidden,
                edge_features=edge_features,
                source_index=self.graph.edge_variables,
                target_index=self.graph.edge_checks,
                target_slices=self.graph.check_slices,
                n_targets=self.graph.n_checks,
                edge_order=self.graph.check_order,
                inverse_edge_order=self.graph.inverse_check_order,
                return_attention=True,
            )
            refined_variable, c_to_v_attention = self.c_to_v(
                target_hidden=variable_hidden,
                source_hidden=refined_check,
                edge_features=edge_features,
                source_index=self.graph.edge_checks,
                target_index=self.graph.edge_variables,
                target_slices=self.graph.variable_slices,
                n_targets=self.graph.n_variables,
                return_attention=True,
            )
            attention = {
                "v_to_c": v_to_c_attention,
                "c_to_v": c_to_v_attention,
            }
        else:
            refined_check = self.v_to_c(
                target_hidden=check_hidden,
                source_hidden=variable_hidden,
                edge_features=edge_features,
                source_index=self.graph.edge_variables,
                target_index=self.graph.edge_checks,
                target_slices=self.graph.check_slices,
                n_targets=self.graph.n_checks,
                edge_order=self.graph.check_order,
                inverse_edge_order=self.graph.inverse_check_order,
                return_attention=False,
            )
            refined_variable = self.c_to_v(
                target_hidden=variable_hidden,
                source_hidden=refined_check,
                edge_features=edge_features,
                source_index=self.graph.edge_checks,
                target_index=self.graph.edge_variables,
                target_slices=self.graph.variable_slices,
                n_targets=self.graph.n_variables,
                return_attention=False,
            )
        return refined_variable, refined_check, attention


class HybridBPGAT(nn.Module):
    """BP followed by one bidirectional edge-aware GATv2 refinement block."""

    def __init__(
        self,
        parity_check: np.ndarray | Tensor,
        bp_iterations: int = 5,
        bp_damping: float = 0.0,
        positional_dim: int = 0,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
        correction_scale: float = 2.0,
        gat_every: int = 0,
    ):
        super().__init__()
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive.")
        if gat_every < 0:
            raise ValueError("gat_every must be non-negative.")
        if gat_every > bp_iterations:
            raise ValueError("gat_every cannot exceed bp_iterations.")
        graph = TannerGraph.from_parity_check(
            parity_check,
            positional_dim=positional_dim,
        )
        self.graph = graph
        self.positional_dim = positional_dim
        self.correction_scale = float(correction_scale)
        # gat_every=0 preserves the original one-shot postprocessor.  A
        # positive value injects a GAT message correction every N BP rounds,
        # allowing the learned graph update to influence later BP messages.
        self.gat_every = int(gat_every)
        self.bp = SumProductBP(graph, iterations=bp_iterations, damping=bp_damping)
        self.gat = BidirectionalEdgeGATv2(
            graph,
            variable_dim=5 + positional_dim,
            check_dim=6 + positional_dim,
            edge_dim=6,
            hidden_dim=hidden_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        edge_input_dim = 2 * hidden_dim + 6
        self.correction = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # These heads are used only by the iterative mode.  They are
        # initialized to zero so the learned decoder starts exactly at BP.
        self.v2c_message_correction = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.c2v_message_correction = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start exactly at the BP decoder.  The refinement is learned as a
        # residual and cannot damage the initial BP solution at initialization.
        for head in (
            self.correction,
            self.v2c_message_correction,
            self.c2v_message_correction,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _features(
        self,
        channel_llr: Tensor,
        bp_llr: Tensor,
        variable_to_check: Tensor,
        check_to_variable: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        variable_features = torch.stack(
            (
                channel_llr,
                bp_llr,
                bp_llr - channel_llr,
                bp_llr.abs(),
            ),
            dim=-1,
        )
        variable_degree = self.graph.variable_degrees.to(channel_llr.device)
        variable_degree = (
            variable_degree / variable_degree.max().clamp_min(1.0)
        ).view(1, -1, 1).expand(channel_llr.shape[0], -1, -1)
        variable_features = torch.cat(
            (
                variable_features,
                variable_degree,
                self.graph.variable_positional.to(channel_llr.device)
                .unsqueeze(0)
                .expand(channel_llr.shape[0], -1, -1),
            ),
            dim=-1,
        )
        check_v2c = self.graph.mean_by_check(variable_to_check)
        check_c2v = self.graph.mean_by_check(check_to_variable)
        soft_check_product = self.graph.product_by_check(
            torch.tanh(0.5 * variable_to_check)
        )
        check_features = torch.cat(
            (
                check_v2c.unsqueeze(-1),
                check_c2v.unsqueeze(-1),
                check_v2c.abs().unsqueeze(-1),
                check_c2v.abs().unsqueeze(-1),
                (
                    self.graph.check_degrees.to(channel_llr.device)
                    / self.graph.check_degrees.max().clamp_min(1.0).to(channel_llr.device)
                )
                .to(channel_llr.device)
                .view(1, -1, 1)
                .expand(channel_llr.shape[0], -1, -1),
                (1.0 - soft_check_product).unsqueeze(-1),
                self.graph.check_positional.to(channel_llr.device)
                .unsqueeze(0)
                .expand(channel_llr.shape[0], -1, -1),
            ),
            dim=-1,
        )
        edge_features = torch.stack(
            (
                variable_to_check,
                check_to_variable,
                variable_to_check.abs(),
                check_to_variable.abs(),
                torch.tanh(0.5 * variable_to_check),
                torch.tanh(0.5 * check_to_variable),
            ),
            dim=-1,
        )
        return variable_features, check_features, edge_features

    def _posterior_from_check_messages(
        self, channel_llr: Tensor, check_to_variable: Tensor
    ) -> Tensor:
        edge_variables = self.graph.edge_variables.to(channel_llr.device)
        posterior = channel_llr.clone()
        posterior.scatter_add_(
            1,
            edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
            check_to_variable,
        )
        return posterior

    def _edge_inputs(
        self,
        variable_hidden: Tensor,
        check_hidden: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        edge_variables = self.graph.edge_variables.to(variable_hidden.device)
        edge_checks = self.graph.edge_checks.to(variable_hidden.device)
        return torch.cat(
            (
                variable_hidden.index_select(1, edge_variables),
                check_hidden.index_select(1, edge_checks),
                edge_features,
            ),
            dim=-1,
        )

    def forward(
        self, channel_llr: Tensor, return_aux: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        # Backward-compatible efficient mode: run BP once, then apply one
        # residual GAT correction to the final posterior.
        if self.gat_every == 0:
            bp_llr, variable_to_check, check_to_variable = self.bp(
                channel_llr, return_messages=True
            )
            variable_features, check_features, edge_features = self._features(
                channel_llr, bp_llr, variable_to_check, check_to_variable
            )
            gat_variable, gat_check, attention = self.gat(
                variable_features,
                check_features,
                edge_features,
                return_attention=return_aux,
            )
            edge_input = self._edge_inputs(
                gat_variable, gat_check, edge_features
            )
            raw_edge_correction = self.correction(edge_input).squeeze(-1)
            edge_correction = self.correction_scale * torch.tanh(raw_edge_correction)
            edge_variables = self.graph.edge_variables.to(channel_llr.device)
            correction = torch.zeros_like(bp_llr)
            correction.scatter_add_(
                1,
                edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
                edge_correction,
            )
            final_llr = bp_llr + correction
            if return_aux:
                return final_llr, {
                    "bp_llr": bp_llr,
                    "correction": correction,
                    "edge_correction": edge_correction,
                    "edge_corrections": [edge_correction],
                    "stage_llrs": [final_llr],
                    "attention": attention["c_to_v"],
                    "v_to_c_attention": attention["v_to_c"],
                    "c_to_v_attention": attention["c_to_v"],
                    "variable_to_check": variable_to_check,
                    "check_to_variable": check_to_variable,
                }
            return final_llr

        # Iterative mode: the GAT predicts residual V->C and C->V message
        # updates.  The corrected messages are fed into the next BP round,
        # which lets the graph network escape BP fixed points.
        variable_to_check = channel_llr.index_select(1, self.bp.edge_variables)
        check_to_variable = torch.zeros_like(variable_to_check)
        final_llr = channel_llr
        last_plain_bp_llr = channel_llr
        stage_llrs = []
        edge_corrections = []
        attention: Dict[str, Tensor] = {}

        for iteration in range(1, self.bp.iterations + 1):
            plain_bp_llr, variable_to_check, check_to_variable = self.bp.step(
                channel_llr,
                variable_to_check,
                check_to_variable,
            )
            last_plain_bp_llr = plain_bp_llr
            if iteration % self.gat_every != 0:
                continue

            variable_features, check_features, edge_features = self._features(
                channel_llr,
                plain_bp_llr,
                variable_to_check,
                check_to_variable,
            )
            gat_variable, gat_check, current_attention = self.gat(
                variable_features,
                check_features,
                edge_features,
                return_attention=return_aux,
            )
            if return_aux:
                attention = current_attention
            edge_input = self._edge_inputs(
                gat_variable, gat_check, edge_features
            )
            raw_v2c = self.v2c_message_correction(edge_input).squeeze(-1)
            raw_c2v = self.c2v_message_correction(edge_input).squeeze(-1)
            v2c_delta = self.correction_scale * torch.tanh(raw_v2c)
            c2v_delta = self.correction_scale * torch.tanh(raw_c2v)
            variable_to_check = variable_to_check + v2c_delta
            check_to_variable = check_to_variable + c2v_delta
            final_llr = self._posterior_from_check_messages(
                channel_llr, check_to_variable
            )
            stage_llrs.append(final_llr)
            edge_corrections.append(torch.cat((v2c_delta, c2v_delta), dim=1))

        # If the schedule does not land exactly on the last BP iteration, the
        # final posterior still comes from the last corrected C->V messages.
        if not stage_llrs:
            final_llr = last_plain_bp_llr
        correction = final_llr - last_plain_bp_llr
        if return_aux:
            last_edge_correction = (
                edge_corrections[-1]
                if edge_corrections
                else torch.zeros_like(variable_to_check)
            )
            return final_llr, {
                "bp_llr": last_plain_bp_llr,
                "correction": correction,
                "edge_correction": last_edge_correction,
                "edge_corrections": edge_corrections,
                "stage_llrs": stage_llrs,
                "attention": attention.get("c_to_v"),
                "v_to_c_attention": attention.get("v_to_c"),
                "c_to_v_attention": attention.get("c_to_v"),
                "variable_to_check": variable_to_check,
                "check_to_variable": check_to_variable,
            }
        return final_llr


def hard_decode(llr: Tensor) -> Tensor:
    """Map LLRs (positive means bit 0) to binary hard decisions."""
    return (llr < 0).to(torch.long)


def bce_from_llr(llr: Tensor, target_bits: Tensor) -> Tensor:
    """Binary cross entropy where target bit 1 corresponds to negative LLR."""
    return F.binary_cross_entropy_with_logits(-llr, target_bits.float())
