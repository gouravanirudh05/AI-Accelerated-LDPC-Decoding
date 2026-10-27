"""Batched scatter/gather primitives for a single fixed Tanner graph.

The code's ``H`` never changes, so signals are batched over one static graph:
node features are ``[B, N, F]`` and edge features ``[B, E, F]``.  This avoids
replicating ``edge_index`` per batch element, which a disjoint-union layout
would require.

The naming follows the PyTorch-Geometric ``MessagePassing`` convention --
``message`` / ``aggregate`` / ``update`` -- so every graph layer in this
package reads the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


def gather_nodes(node_features: Tensor, index: Tensor) -> Tensor:
    """Broadcast ``[B, N, F]`` node features onto ``[B, E, F]`` edges."""
    return node_features.index_select(1, index)


def scatter_sum(edge_features: Tensor, index: Tensor, size: int) -> Tensor:
    """Sum ``[B, E, F]`` edge features into ``[B, size, F]`` node features."""
    batch, _, channels = edge_features.shape
    out = edge_features.new_zeros((batch, size, channels))
    expanded = index.view(1, -1, 1).expand(batch, index.numel(), channels)
    return out.scatter_add(1, expanded, edge_features)


def scatter_mean(
    edge_features: Tensor, index: Tensor, size: int, degrees: Optional[Tensor] = None
) -> Tensor:
    """Degree-normalized sum. ``degrees`` avoids recomputing node degrees."""
    total = scatter_sum(edge_features, index, size)
    if degrees is None:
        ones = edge_features.new_ones((1, index.numel(), 1))
        degrees = scatter_sum(ones, index, size).squeeze(0).squeeze(-1)
    return total / degrees.clamp_min(1.0).view(1, -1, 1).to(total.dtype)


def scatter_max(edge_features: Tensor, index: Tensor, size: int) -> Tensor:
    """Max-reduce ``[B, E, F]`` onto ``[B, size, F]``; empty nodes give ``-inf``."""
    batch, _, channels = edge_features.shape
    out = edge_features.new_full((batch, size, channels), float("-inf"))
    expanded = index.view(1, -1, 1).expand(batch, index.numel(), channels)
    return out.scatter_reduce(1, expanded, edge_features, reduce="amax")


def segment_softmax(logits: Tensor, index: Tensor, size: int) -> Tensor:
    """Softmax over the edges sharing a target node.

    ``logits`` is ``[B, E, H]`` (H attention heads) and ``index`` maps each
    edge to its target node, so the result sums to one over each target's
    incoming edges -- per head, per batch element.
    """
    maximum = scatter_max(logits, index, size)
    # A node with no incoming edges would give -inf; the graph builder rejects
    # those, but guard anyway so the arithmetic below stays finite.
    maximum = torch.nan_to_num(maximum, neginf=0.0)
    shifted = (logits - gather_nodes(maximum, index)).exp()
    denominator = gather_nodes(scatter_sum(shifted, index, size), index)
    return shifted / denominator.clamp_min(1e-16)


def linear_flops(in_features: int, out_features: int, count: int) -> float:
    """Multiply-accumulate FLOPs for ``count`` applications of a linear map."""
    return 2.0 * float(in_features) * float(out_features) * float(count)


def linear_flops_for(layer: nn.Linear, count: int) -> float:
    """``linear_flops`` read straight off an ``nn.Linear``."""
    return linear_flops(layer.in_features, layer.out_features, count)


class MessagePassingBlock(nn.Module, ABC):
    """Minimal ``MessagePassing``-style base for one directed pass.

    Subclasses implement :meth:`message` and, if they need more than a plain
    sum, :meth:`aggregate`.  :meth:`propagate` wires them together over one
    directed edge index.
    """

    def __init__(self, aggr: str = "sum") -> None:
        super().__init__()
        if aggr not in ("sum", "mean"):
            raise ValueError("aggr must be 'sum' or 'mean'.")
        self.aggr = aggr

    @abstractmethod
    def message(
        self,
        source_features: Tensor,
        target_features: Tensor,
        edge_features: Optional[Tensor],
    ) -> Tensor:
        """Compute the ``[B, E, F]`` message flowing along each directed edge."""

    def aggregate(
        self, messages: Tensor, target_index: Tensor, target_size: int
    ) -> Tensor:
        if self.aggr == "mean":
            return scatter_mean(messages, target_index, target_size)
        return scatter_sum(messages, target_index, target_size)

    def update(self, aggregated: Tensor, target_features: Tensor) -> Tensor:
        return aggregated

    def propagate(
        self,
        edge_index: Tensor,
        source_features: Tensor,
        target_features: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one directed pass over ``edge_index`` (row 0 source, row 1 target)."""
        source_index, target_index = edge_index[0], edge_index[1]
        target_size = target_features.shape[1]
        gathered_source = gather_nodes(source_features, source_index)
        gathered_target = gather_nodes(target_features, target_index)
        messages = self.message(gathered_source, gathered_target, edge_features)
        aggregated = self.aggregate(messages, target_index, target_size)
        return self.update(aggregated, target_features)
