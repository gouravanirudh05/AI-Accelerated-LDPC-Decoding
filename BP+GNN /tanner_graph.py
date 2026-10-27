"""Directed Tanner graph for LDPC parity-check matrices.

The graph exposes two *directed* edge-index tensors, ``vn_to_cn`` and
``cn_to_vn``, so the two message directions can carry separate learned
weights.  They are deliberately not collapsed into one undirected edge set.

Internally the graph is always a sparse COO triple, so a dense ``numpy``
array (small test codes) and a ``scipy.sparse`` matrix (5G scale) both work
through the same call site.  ``H`` is never densified unless
:meth:`TannerGraph.to_dense` is called explicitly.

Edge ordering is canonical and grouped by variable node, matching the layout
of ``GAT+BP/gat_bp.py``.  A ``[B, E]`` message tensor produced by the Phase
0-2 ``SumProductBP`` therefore index-aligns with ``edge_vn`` / ``edge_cn``
without any permutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

try:  # scipy is optional: dense codes never need it.
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    sp = None  # type: ignore[assignment]
    _HAVE_SCIPY = False


def _to_coo(parity_check: Any) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Normalize any supported ``H`` to ``(rows, cols, n_checks, n_vars)``.

    Accepts a dense ``numpy`` array, a dense ``torch`` tensor, a sparse
    ``torch`` tensor, or any ``scipy.sparse`` matrix.  Only the nonzero
    pattern matters: ``H`` is binary, so values are discarded.
    """
    if _HAVE_SCIPY and sp is not None and sp.issparse(parity_check):
        coo = parity_check.tocoo()
        n_checks, n_variables = coo.shape
        mask = coo.data != 0
        return (
            coo.row[mask].astype(np.int64),
            coo.col[mask].astype(np.int64),
            int(n_checks),
            int(n_variables),
        )

    if isinstance(parity_check, Tensor):
        if parity_check.is_sparse:
            coalesced = parity_check.detach().coalesce()
            indices = coalesced.indices().cpu().numpy()
            values = coalesced.values().cpu().numpy()
            mask = values != 0
            n_checks, n_variables = coalesced.shape
            return (
                indices[0][mask].astype(np.int64),
                indices[1][mask].astype(np.int64),
                int(n_checks),
                int(n_variables),
            )
        dense = parity_check.detach().cpu().numpy()
    else:
        dense = np.asarray(parity_check)

    if dense.ndim != 2:
        raise ValueError("The parity-check matrix must be two-dimensional.")
    rows, cols = np.nonzero(dense != 0)
    return (
        rows.astype(np.int64),
        cols.astype(np.int64),
        int(dense.shape[0]),
        int(dense.shape[1]),
    )


@dataclass
class TannerGraph:
    """Static directed edge layout for a binary parity-check matrix."""

    n_variables: int
    n_checks: int

    # One entry per nonzero of H, canonically ordered (grouped by variable).
    edge_vn: Tensor  # [E] variable endpoint of each edge
    edge_cn: Tensor  # [E] check endpoint of each edge

    # Directed PyG-style indices: row 0 is the source, row 1 the target.
    vn_to_cn: Tensor  # [2, E] variable -> check
    cn_to_vn: Tensor  # [2, E] check -> variable

    vn_degrees: Tensor  # [n] column sums of H
    cn_degrees: Tensor  # [m] row sums of H

    # Permutation regrouping the same physical edges by check node, kept for
    # interoperability with the Phase 0-2 BP implementation.
    check_order: Tensor
    inverse_check_order: Tensor

    _two_hop_vn: Optional[Tensor] = field(default=None, repr=False)
    _two_hop_cn: Optional[Tensor] = field(default=None, repr=False)
    _two_hop_vn_shared: Optional[Tensor] = field(default=None, repr=False)
    _two_hop_cn_shared: Optional[Tensor] = field(default=None, repr=False)

    # ---------------------------------------------------------------- build

    @classmethod
    def from_parity_check(
        cls,
        parity_check: Any,
        validate: bool = True,
    ) -> "TannerGraph":
        rows, cols, n_checks, n_variables = _to_coo(parity_check)

        if rows.size == 0:
            raise ValueError("The parity-check matrix has no Tanner-graph edges.")

        vn_degrees = np.bincount(cols, minlength=n_variables)
        cn_degrees = np.bincount(rows, minlength=n_checks)
        if validate:
            if np.any(vn_degrees == 0):
                raise ValueError("Every variable node must participate in a check.")
            if np.any(cn_degrees == 0):
                raise ValueError("Every check node must contain at least one variable.")

        # Canonical order: group by variable, then by check within a variable.
        order = np.lexsort((rows, cols))
        edge_vn = cols[order]
        edge_cn = rows[order]

        check_order = np.lexsort((edge_vn, edge_cn))
        inverse_check_order = np.argsort(check_order)

        as_long = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
        edge_vn_t = as_long(edge_vn)
        edge_cn_t = as_long(edge_cn)

        return cls(
            n_variables=n_variables,
            n_checks=n_checks,
            edge_vn=edge_vn_t,
            edge_cn=edge_cn_t,
            vn_to_cn=torch.stack((edge_vn_t, edge_cn_t), dim=0),
            cn_to_vn=torch.stack((edge_cn_t, edge_vn_t), dim=0),
            vn_degrees=torch.as_tensor(vn_degrees, dtype=torch.float32),
            cn_degrees=torch.as_tensor(cn_degrees, dtype=torch.float32),
            check_order=as_long(check_order),
            inverse_check_order=as_long(inverse_check_order),
        )

    # ---------------------------------------------------------- properties

    @property
    def n_edges(self) -> int:
        return int(self.edge_vn.numel())

    @property
    def device(self) -> torch.device:
        return self.edge_vn.device

    def to(self, device: torch.device | str) -> "TannerGraph":
        """Move every index/degree buffer to ``device``."""
        moved = {}
        for name in (
            "edge_vn",
            "edge_cn",
            "vn_to_cn",
            "cn_to_vn",
            "vn_degrees",
            "cn_degrees",
            "check_order",
            "inverse_check_order",
            "_two_hop_vn",
            "_two_hop_cn",
            "_two_hop_vn_shared",
            "_two_hop_cn_shared",
        ):
            value = getattr(self, name)
            moved[name] = value.to(device) if isinstance(value, Tensor) else value
        return TannerGraph(
            n_variables=self.n_variables,
            n_checks=self.n_checks,
            **moved,
        )

    def to_dense(self) -> Tensor:
        """Materialize ``H`` as a dense float tensor.

        Only for small codes and for utilities that genuinely need dense ``H``
        (syndrome checks, the legacy BP constructor).
        """
        dense = torch.zeros(
            (self.n_checks, self.n_variables),
            dtype=torch.float32,
            device=self.edge_vn.device,
        )
        dense[self.edge_cn, self.edge_vn] = 1.0
        return dense

    # --------------------------------------------------------- aggregation

    def _scatter(
        self,
        edge_values: Tensor,
        index: Tensor,
        size: int,
        reduce: str,
        degrees: Tensor,
    ) -> Tensor:
        """Scatter ``[..., E]`` edge values onto ``[..., size]`` nodes.

        ``degrees`` is passed in rather than inferred from ``size``: inferring
        it would be ambiguous for a square ``H``, where ``n == m``.
        """
        if edge_values.shape[-1] != self.n_edges:
            raise ValueError(
                f"Expected the last dimension to be E={self.n_edges}, "
                f"got {edge_values.shape[-1]}."
            )
        shape = edge_values.shape[:-1] + (size,)
        out = edge_values.new_zeros(shape)
        expanded = index.expand(shape[:-1] + (self.n_edges,))
        out = out.scatter_add(-1, expanded, edge_values)
        if reduce == "sum":
            return out
        if reduce == "mean":
            return out / degrees.clamp_min(1.0).to(out.dtype)
        raise ValueError(f"Unsupported reduction {reduce!r}.")

    def aggregate_to_vn(self, edge_values: Tensor, reduce: str = "sum") -> Tensor:
        """Aggregate edge values into variable-node features."""
        return self._scatter(
            edge_values, self.edge_vn, self.n_variables, reduce, self.vn_degrees
        )

    def aggregate_to_cn(self, edge_values: Tensor, reduce: str = "sum") -> Tensor:
        """Aggregate edge values into check-node features."""
        return self._scatter(
            edge_values, self.edge_cn, self.n_checks, reduce, self.cn_degrees
        )

    def product_to_cn(self, edge_values: Tensor, eps: float = 1e-12) -> Tensor:
        """Multiply ``[..., E]`` edge values into ``[..., m]`` check features.

        Computed in the log domain so it stays vectorized (and therefore usable
        at 5G scale) instead of looping over checks: magnitudes add as logs, the
        sign is the parity of the negative count, and any exact zero forces the
        whole product to zero.
        """
        negatives = (edge_values < 0).to(edge_values.dtype)
        zeros = (edge_values == 0).to(edge_values.dtype)
        log_magnitude = edge_values.abs().clamp_min(eps).log()

        summed_log = self.aggregate_to_cn(log_magnitude)
        negative_count = self.aggregate_to_cn(negatives)
        zero_count = self.aggregate_to_cn(zeros)

        sign = 1.0 - 2.0 * torch.remainder(negative_count, 2.0)
        product = sign * summed_log.exp()
        return torch.where(zero_count > 0, torch.zeros_like(product), product)

    def gather_vn(self, node_values: Tensor) -> Tensor:
        """Broadcast ``[..., n]`` variable features onto their ``[..., E]`` edges."""
        return node_values.index_select(-1, self.edge_vn)

    def gather_cn(self, node_values: Tensor) -> Tensor:
        """Broadcast ``[..., m]`` check features onto their ``[..., E]`` edges."""
        return node_values.index_select(-1, self.edge_cn)

    # ------------------------------------------------------------- two-hop

    def two_hop_index(self, side: str = "vn") -> Tensor:
        """Return (and cache) the 2-hop edge index for one node side.

        Two variable nodes are 2-hop neighbours when they share a check; two
        check nodes when they share a variable.  Materialized once and cached,
        never recomputed inside a forward pass.

        Edges are sorted by *target* (row 1), so a consumer can slice each
        target's incoming edges contiguously -- which is what the attention
        softmax needs.
        """
        if side not in ("vn", "cn"):
            raise ValueError("side must be 'vn' or 'cn'.")

        cached = self._two_hop_vn if side == "vn" else self._two_hop_cn
        if cached is not None:
            return cached

        src = self.edge_vn.cpu().numpy() if side == "vn" else self.edge_cn.cpu().numpy()
        via = self.edge_cn.cpu().numpy() if side == "vn" else self.edge_vn.cpu().numpy()
        size = self.n_variables if side == "vn" else self.n_checks
        n_via = self.n_checks if side == "vn" else self.n_variables

        if _HAVE_SCIPY and sp is not None:
            incidence = sp.coo_matrix(
                (np.ones(src.size, dtype=np.int8), (src, via)),
                shape=(size, n_via),
            ).tocsr()
            adjacency = (incidence @ incidence.T).tocoo()
            keep = adjacency.row != adjacency.col
            rows = adjacency.row[keep]
            cols = adjacency.col[keep]
            shared = adjacency.data[keep]
        else:  # dense fallback for small codes without scipy
            incidence = np.zeros((size, n_via), dtype=np.int64)
            incidence[src, via] = 1
            adjacency = incidence @ incidence.T
            np.fill_diagonal(adjacency, 0)
            rows, cols = np.nonzero(adjacency)
            shared = adjacency[rows, cols]

        # Treat ``cols`` as the source and ``rows`` as the target, then sort by
        # target so each target's incoming edges form one contiguous block.
        order = np.lexsort((cols, rows))
        source = cols[order]
        target = rows[order]
        shared = np.asarray(shared, dtype=np.float32)[order]

        index = torch.stack(
            (
                torch.as_tensor(source, dtype=torch.long),
                torch.as_tensor(target, dtype=torch.long),
            ),
            dim=0,
        ).to(self.edge_vn.device)
        shared_counts = torch.as_tensor(shared, dtype=torch.float32).to(
            self.edge_vn.device
        )

        if side == "vn":
            self._two_hop_vn = index
            self._two_hop_vn_shared = shared_counts
        else:
            self._two_hop_cn = index
            self._two_hop_cn_shared = shared_counts
        return index

    def two_hop_shared_counts(self, side: str = "vn") -> Tensor:
        """How many nodes each 2-hop pair shares, aligned with the index."""
        self.two_hop_index(side)
        counts = self._two_hop_vn_shared if side == "vn" else self._two_hop_cn_shared
        assert counts is not None  # set by two_hop_index
        return counts

    @staticmethod
    def contiguous_target_slices(
        target_index: Tensor, size: int
    ) -> List[Tuple[int, int]]:
        """``[start, end)`` per target for a target-sorted edge index.

        Targets with no incoming edges get an empty slice, which the attention
        softmax must then skip.
        """
        counts = torch.bincount(target_index, minlength=size).tolist()
        slices: List[Tuple[int, int]] = []
        start = 0
        for count in counts:
            slices.append((start, start + count))
            start += count
        if start != int(target_index.numel()):
            raise ValueError("The edge index is not sorted by target.")
        return slices


def build_tanner_graph(parity_check: Any, validate: bool = True) -> TannerGraph:
    """Build a directed Tanner graph from dense or sparse ``H``."""
    return TannerGraph.from_parity_check(parity_check, validate=validate)
