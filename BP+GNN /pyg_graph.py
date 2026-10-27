"""PyTorch Geometric adapters for a fixed Tanner graph.

The decoder pipeline works in a batched ``[B, nodes, channels]`` layout (one
static ``H``, many received frames).  PyG layers want the flat disjoint-union
layout, ``[B * nodes, channels]`` with ``edge_index`` replicated and offset per
batch element.  This module converts between the two and caches the replicated
indices, so the offsets are built once per batch size rather than per forward
pass.

The Tanner graph is bipartite, which PyG supports natively: pass node features
as a tuple ``(x_src, x_dst)`` and give ``size=(num_src, num_dst)``.  Every
convolution must be constructed with ``add_self_loops=False`` -- a self-loop is
not meaningful between a variable node and a check node.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from .tanner_graph import TannerGraph


class BipartiteBatcher:
    """Replicates a bipartite ``edge_index`` across a batch, with caching."""

    # A decoder sees few distinct batch sizes (train batch, eval batch, and a
    # ragged final batch), so a small cache covers everything in practice.
    MAX_CACHED = 8

    def __init__(self, graph: TannerGraph) -> None:
        self.n_variables = graph.n_variables
        self.n_checks = graph.n_checks
        self.n_edges = graph.n_edges
        self._vn_to_cn = graph.vn_to_cn
        self._cn_to_vn = graph.cn_to_vn
        self._cache: Dict[Tuple[int, torch.device, str], Tensor] = {}

    def _replicate(self, edge_index: Tensor, batch: int, src_size: int, dst_size: int) -> Tensor:
        """``[2, E]`` -> ``[2, B * E]`` with per-element node offsets."""
        offsets = torch.arange(batch, device=edge_index.device).view(1, batch, 1)
        source = edge_index[0].view(1, 1, -1) + offsets * src_size
        target = edge_index[1].view(1, 1, -1) + offsets * dst_size
        return torch.cat((source.reshape(1, -1), target.reshape(1, -1)), dim=0)

    def edge_index(self, direction: str, batch: int, device: torch.device) -> Tensor:
        """Cached replicated index for ``"vn_to_cn"`` or ``"cn_to_vn"``."""
        if direction not in ("vn_to_cn", "cn_to_vn"):
            raise ValueError("direction must be 'vn_to_cn' or 'cn_to_vn'.")

        key = (batch, device, direction)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if direction == "vn_to_cn":
            base = self._vn_to_cn.to(device)
            replicated = self._replicate(base, batch, self.n_variables, self.n_checks)
        else:
            base = self._cn_to_vn.to(device)
            replicated = self._replicate(base, batch, self.n_checks, self.n_variables)

        if len(self._cache) >= self.MAX_CACHED:
            self._cache.clear()
        self._cache[key] = replicated
        return replicated

    def size(self, direction: str, batch: int) -> Tuple[int, int]:
        """``(num_src, num_dst)`` for a replicated bipartite pass."""
        if direction == "vn_to_cn":
            return batch * self.n_variables, batch * self.n_checks
        return batch * self.n_checks, batch * self.n_variables


def flatten_nodes(node_features: Tensor) -> Tensor:
    """``[B, N, F]`` -> ``[B * N, F]``."""
    if node_features.ndim != 3:
        raise ValueError("Expected [batch, nodes, channels].")
    return node_features.reshape(-1, node_features.shape[-1])


def unflatten_nodes(flat: Tensor, batch: int) -> Tensor:
    """``[B * N, F]`` -> ``[B, N, F]``."""
    return flat.reshape(batch, -1, flat.shape[-1])


def flatten_edges(edge_features: Tensor) -> Tensor:
    """``[B, E, F]`` -> ``[B * E, F]``, aligned with a replicated edge_index."""
    if edge_features.ndim != 3:
        raise ValueError("Expected [batch, edges, channels].")
    return edge_features.reshape(-1, edge_features.shape[-1])


def two_hop_batcher(graph: TannerGraph, side: str = "vn"):
    """A batcher for the cached 2-hop (variable-to-variable) index."""

    class _TwoHop(BipartiteBatcher):
        """Homogeneous index, so both endpoints share one node count."""

        def __init__(self) -> None:
            super().__init__(graph)
            index = graph.two_hop_index(side)
            self._vn_to_cn = index
            self._cn_to_vn = index
            self.node_count = (
                graph.n_variables if side == "vn" else graph.n_checks
            )
            self.n_edges = int(index.shape[1])

        def edge_index(self, direction: str, batch: int, device: torch.device) -> Tensor:
            key = (batch, device, "two_hop")
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            replicated = self._replicate(
                self._vn_to_cn.to(device), batch, self.node_count, self.node_count
            )
            if len(self._cache) >= self.MAX_CACHED:
                self._cache.clear()
            self._cache[key] = replicated
            return replicated

        def size(self, direction: str, batch: int) -> Tuple[int, int]:
            return batch * self.node_count, batch * self.node_count

    return _TwoHop()
