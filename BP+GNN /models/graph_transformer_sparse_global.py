"""Sparse-global graph-transformer corrector.

After BP(k), pick the subset ``S`` of variable nodes BP left unresolved, then
run *full* (structurally unmasked) self-attention among the members of ``S``
only -- a standard transformer block, no Tanner-graph edges involved.  The
premise: residual errors after truncated BP are few and coupled, so the bits
that matter should be able to see each other directly rather than through
several hops of the graph.

Selection is config-driven:

``mode="threshold"``
    ``S = { v : |LLR(v)| < threshold }`` after BP(k).  ``threshold`` is in raw
    LLR units, so ``threshold: 2.0`` means exactly ``|LLR| < 2``.  ``S`` then
    varies per frame, so the batch is padded to the largest ``|S|`` and the
    padding is excluded with a key-padding mask.
``mode="top_k"``
    ``S`` = the ``top_k`` least-confident bits per frame; fixed size.

Variable nodes outside ``S`` get ``delta_llr = 0``, enforced by gather/scatter
on the *output* -- they are never fed zeroed inputs, so nothing is misled into
treating a confident bit as an erased one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..bp_trace import LLR_SCALE, VN_IDX_CONFIDENCE
from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector, EncodeResult
from .graph_ops import linear_flops, linear_flops_for


class GraphTransformerSparseGlobal(BaseCorrector):
    """Full attention over the least-confident variable nodes only."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        heads: int = 2,
        num_layers: int = 1,
        mode: str = "threshold",
        threshold: float = 2.0,
        top_k: int = 8,
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        residual_mode: str = "node",
    ) -> None:
        # Attention here is over a selected node subset, not over edges, so
        # there is no per-edge context for an edge-wise residual.
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
        if mode not in ("threshold", "top_k"):
            raise ValueError("mode must be 'threshold' or 'top_k'.")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if mode == "top_k" and not 1 <= top_k <= graph.n_variables:
            raise ValueError(f"top_k must be in [1, {graph.n_variables}].")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")

        self.mode = mode
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.heads = int(heads)
        self.num_layers = int(num_layers)

        self.vn_encoder = nn.Linear(vn_dim, hidden_dim)
        nn.init.xavier_uniform_(self.vn_encoder.weight)
        nn.init.zeros_(self.vn_encoder.bias)

        self.attention = nn.ModuleList(
            nn.MultiheadAttention(
                hidden_dim, self.heads, dropout=dropout, batch_first=True
            )
            for _ in range(self.num_layers)
        )
        self.attention_norm = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
        )
        self.feed_forward = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_layers)
        )
        self.feed_forward_norm = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
        )

    # ------------------------------------------------------------- selection

    def select(self, confidence: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Choose ``S`` per frame.

        Returns ``(indices, valid, mask)`` where ``indices`` is ``[B, C]`` into
        the variable axis, ``valid`` is ``[B, C]`` marking real (non-padding)
        slots, and ``mask`` is the ``[B, n]`` membership mask of ``S``.
        """
        batch, n_variables = confidence.shape

        if self.mode == "top_k":
            capacity = min(self.top_k, n_variables)
            # Least confident first.
            indices = confidence.topk(capacity, dim=1, largest=False).indices
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            selected = confidence < self.threshold
            counts = selected.sum(dim=1)
            capacity = int(counts.max().item())
            if capacity == 0:
                empty_index = confidence.new_zeros((batch, 0), dtype=torch.long)
                empty_valid = confidence.new_zeros((batch, 0), dtype=torch.bool)
                return empty_index, empty_valid, torch.zeros_like(confidence)
            # Stable sort brings the selected nodes to the front in index
            # order; the tail is padding.
            order = torch.argsort(
                (~selected).to(torch.uint8), dim=1, stable=True
            )
            indices = order[:, :capacity]
            positions = torch.arange(capacity, device=confidence.device)
            valid = positions.unsqueeze(0) < counts.unsqueeze(1)

        mask = torch.zeros_like(confidence)
        mask.scatter_(1, indices, valid.to(confidence.dtype))
        return indices, valid, mask

    # --------------------------------------------------------------- forward

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        return self.encode_all(
            vn_features, cn_features, vn_to_cn_edge_index, cn_to_vn_edge_index, **extra
        ).stages

    def encode_all(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        confidence: Optional[Tensor] = None,
        **extra: Tensor,
    ) -> EncodeResult:
        if confidence is None:
            # The VN feature vector carries |LLR| / LLR_SCALE; undo the scaling
            # so ``threshold`` is interpreted in raw LLR units.
            confidence = vn_features[..., VN_IDX_CONFIDENCE] * LLR_SCALE

        indices, valid, mask = self.select(confidence)
        hidden = self.vn_encoder(vn_features)

        if indices.shape[1] == 0:
            # Empty S: nothing to attend over. The mask pins every residual to
            # zero, so return the encoded state untouched rather than erroring.
            return EncodeResult(stages=[hidden], mask=mask)

        gather_index = indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        subset = hidden.gather(1, gather_index)
        # Padding slots must not be attended to, and a fully padded row would
        # make softmax undefined, so keep at least one key alive per row.
        key_padding_mask = ~valid
        key_padding_mask = torch.where(
            key_padding_mask.all(dim=1, keepdim=True),
            torch.zeros_like(key_padding_mask),
            key_padding_mask,
        )

        for layer in range(self.num_layers):
            attended, _ = self.attention[layer](
                subset, subset, subset, key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            subset = self.attention_norm[layer](subset + attended)
            subset = self.feed_forward_norm[layer](
                subset + self.feed_forward[layer](subset)
            )

        # Scatter the refined states back. Padded slots point at arbitrary
        # nodes, so restore their original state rather than overwriting it;
        # untouched nodes keep their encoded state, and the output mask is what
        # actually zeroes their residual.
        original = hidden.gather(1, gather_index)
        subset = torch.where(valid.unsqueeze(-1), subset, original)
        refined = hidden.scatter(1, gather_index, subset)
        return EncodeResult(stages=[refined], mask=mask)

    def extra_inputs(self, trace: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Selection uses the same scaled confidence the features carry, so
        # nothing extra is needed; the trace is available if that changes.
        return {}

    # ------------------------------------------------------------------ cost

    def estimate_flops_per_frame(self) -> float:
        """Cost at the configured selection size.

        ``top_k`` gives an exact count.  In ``threshold`` mode ``|S|`` is
        data-dependent, so this reports the worst case, ``|S| = n``.
        """
        selection = self.top_k if self.mode == "top_k" else self.graph.n_variables
        total = linear_flops_for(self.vn_encoder, self.graph.n_variables)
        for layer in range(self.num_layers):
            # QKV projections plus the output projection, once per member.
            total += 4 * linear_flops(self.hidden_dim, self.hidden_dim, selection)
            # Full attention among |S| members: scores and the value mixing.
            total += 4.0 * selection * selection * self.hidden_dim
            total += 2 * linear_flops(
                self.hidden_dim, 2 * self.hidden_dim, selection
            )
        return total + self.head_flops_per_frame()
