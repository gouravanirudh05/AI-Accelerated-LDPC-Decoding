"""Common interface for every graph-based residual corrector.

The contract from the spec:

    forward(vn_features, cn_features, vn_to_cn_edge_index, cn_to_vn_edge_index)
        -> delta_llr  [B, n]

Combination happens *outside* the model::

    L_final = L_BP + gate * delta_llr

``gate`` is either a fixed scalar from config, or a learned per-bit gate.  When
learned, the model additionally exposes ``gate_logits`` ``[B, n]`` -- a linear
head plus sigmoid on the final VN hidden state, deliberately per-bit rather
than one global scalar.

Subclasses implement :meth:`encode_stages`, returning one VN hidden state per
unrolled layer.  The last element is the final hidden state; the earlier ones
feed the ``deep_supervision`` loss term through the same shared residual head.  The residual head and the optional gate head live here so all
four architectures share identical output behaviour, in particular the
zero-initialized residual: at initialization ``delta_llr == 0``, so an untrained
corrector reproduces BP(k) exactly and cannot degrade the baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..tanner_graph import TannerGraph
from .graph_ops import gather_nodes


@dataclass
class EncodeResult:
    """What :meth:`BaseCorrector.encode_all` returns.

    ``cn_hidden`` and ``edge_features`` are only needed by the edge-wise
    residual; node-wise architectures leave them ``None``.
    """

    stages: List[Tensor]  # one [B, n, hidden] VN state per unrolled layer
    mask: Optional[Tensor] = None  # [B, n] bits this corrector may touch
    cn_hidden: Optional[Tensor] = None  # [B, m, hidden]
    edge_features: Optional[Tensor] = None  # [B, E, edge_dim]


@dataclass
class CorrectorOutput:
    """What a corrector returns when asked for more than ``delta_llr``."""

    delta_llr: Tensor  # [B, n]
    gate_logits: Optional[Tensor] = None  # [B, n], None when the gate is fixed
    # One delta per unrolled layer, last == ``delta_llr``. Used by the
    # deep_supervision loss term.
    stage_deltas: Optional[List[Tensor]] = None
    # [B, n] 0/1 mask of the bits this corrector was allowed to touch. ``None``
    # means every bit. Sparse architectures use it to pin the rest to zero.
    output_mask: Optional[Tensor] = None


def combine(
    bp_llr: Tensor,
    delta_llr: Tensor,
    gate_logits: Optional[Tensor] = None,
    fixed_gate: float = 1.0,
) -> Tensor:
    """``L_final = L_BP + gate * delta_llr``, done outside the model."""
    gate = torch.sigmoid(gate_logits) if gate_logits is not None else fixed_gate
    return bp_llr + gate * delta_llr


class BaseCorrector(nn.Module, ABC):
    """Abstract residual corrector over a fixed Tanner graph."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        residual_mode: str = "node",
        edge_dim: int = 0,
    ) -> None:
        super().__init__()
        if correction_scale <= 0.0:
            raise ValueError("correction_scale must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if residual_mode not in ("node", "edge"):
            raise ValueError("residual_mode must be 'node' or 'edge'.")

        self.graph = graph
        self.vn_dim = int(vn_dim)
        self.cn_dim = int(cn_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_gate = bool(use_gate)
        self.fixed_gate = float(fixed_gate)
        self.correction_scale = float(correction_scale)
        self.dropout = float(dropout)
        self.residual_mode = residual_mode

        # Index buffers travel with the module across devices.
        self.register_buffer("vn_to_cn", graph.vn_to_cn, persistent=False)
        self.register_buffer("cn_to_vn", graph.cn_to_vn, persistent=False)
        self.register_buffer("edge_vn", graph.edge_vn, persistent=False)
        self.register_buffer("edge_cn", graph.edge_cn, persistent=False)
        self.register_buffer("vn_degrees", graph.vn_degrees, persistent=False)
        self.register_buffer("cn_degrees", graph.cn_degrees, persistent=False)

        if residual_mode == "node":
            # One correction per variable node.
            self.delta_head = nn.Linear(self.hidden_dim, 1)
            self.edge_delta_head = None
        else:
            # One correction per *edge*, summed onto its variable node. This is
            # the Phase 0-2 formulation: each incident edge contributes a
            # correction conditioned on its own BP messages, which is far more
            # expressive than a single per-node scalar. Measured on this code,
            # the node-wise residual loses to BP(5) on FER while the legacy
            # edge-wise one beats it.
            if edge_dim <= 0:
                raise ValueError(
                    "residual_mode='edge' needs a positive edge_dim; this "
                    "architecture exposes no per-edge context to condition on."
                )
            # No node-wise head in edge mode: it would never be used, and dead
            # parameters silently receive no gradient.
            self.delta_head = None
            self.edge_delta_head = nn.Sequential(
                nn.Linear(2 * self.hidden_dim + edge_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.edge_delta_head[-1].weight)
            nn.init.zeros_(self.edge_delta_head[-1].bias)

        # Start exactly at BP: the correction is a residual that is identically
        # zero until training moves it.
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

        if self.use_gate:
            self.gate_head = nn.Linear(self.hidden_dim, 1)
            nn.init.zeros_(self.gate_head.weight)
            # sigmoid(2.0) ~= 0.88: the gate starts mostly open, so the learned
            # residual is not throttled before it has learned anything.
            nn.init.constant_(self.gate_head.bias, 2.0)
        else:
            self.gate_head = None

    # ------------------------------------------------------------- subclass

    @abstractmethod
    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        """Return one ``[B, n, hidden_dim]`` VN hidden state per unrolled layer.

        Must be non-empty; the last element is the final hidden state.
        ``extra`` carries whatever :meth:`extra_inputs` supplied.
        """

    def encode(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> Tensor:
        """The final VN hidden state ``[B, n, hidden_dim]``."""
        return self.encode_stages(
            vn_features, cn_features, vn_to_cn_edge_index, cn_to_vn_edge_index, **extra
        )[-1]

    def encode_all(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> EncodeResult:
        """Hidden states plus whatever the residual head needs.

        Architectures that correct only a subset of bits override this and set
        ``mask``; the residual is then forced to zero everywhere else, on the
        *output* side.  Architectures supporting ``residual_mode='edge'`` also
        set ``cn_hidden`` and ``edge_features``.
        """
        stages = self.encode_stages(
            vn_features, cn_features, vn_to_cn_edge_index, cn_to_vn_edge_index, **extra
        )
        return EncodeResult(stages=stages)

    def _delta_from_hidden(self, hidden: Tensor) -> Tensor:
        """Bounded per-node residual.

        tanh keeps a badly-behaved corrector from blowing up a confident BP
        posterior.  ``correction_scale`` has to be large enough to flip a
        confidently-wrong bit -- see ``configs/*.yaml`` for the measurement
        behind the default of 8.
        """
        if self.delta_head is None:
            raise RuntimeError(
                "No node-wise residual head: this corrector was built with "
                "residual_mode='edge'."
            )
        return self.correction_scale * torch.tanh(self.delta_head(hidden).squeeze(-1))

    def _delta_from_edges(
        self, vn_hidden: Tensor, cn_hidden: Tensor, edge_features: Tensor
    ) -> Tensor:
        """Per-edge residuals, each bounded, summed onto their variable node.

        Every incident edge contributes a correction conditioned on its own BP
        messages -- the Phase 0-2 formulation.

        Note the bound: because the per-edge terms are *summed*, a degree-d bit
        can be moved by up to ``d * correction_scale``, not ``correction_scale``.
        So ``correction_scale`` means different things in the two modes, and an
        edge-mode config needs roughly ``node_scale / mean_degree``.  Getting
        this wrong is not subtle: at scale 8 with mean degree 4 the effective
        range is +/-32, and a corrector measured here applied a median
        correction of 13 LLR units to every bit, breaking 3336 bits BP had
        right while fixing 634.  Use :attr:`effective_correction_range` to see
        the real bound.
        """
        edge_inputs = torch.cat(
            (
                gather_nodes(vn_hidden, self.edge_vn),
                gather_nodes(cn_hidden, self.edge_cn),
                edge_features,
            ),
            dim=-1,
        )
        edge_delta = self.correction_scale * torch.tanh(
            self.edge_delta_head(edge_inputs).squeeze(-1)
        )
        return self.graph.aggregate_to_vn(edge_delta)

    def _delta(self, hidden: Tensor, encoded: EncodeResult) -> Tensor:
        """Residual from whichever head ``residual_mode`` selected."""
        if self.residual_mode == "node":
            return self._delta_from_hidden(hidden)
        if encoded.cn_hidden is None or encoded.edge_features is None:
            raise RuntimeError(
                f"{type(self).__name__} does not support residual_mode='edge': "
                "encode_all() must supply cn_hidden and edge_features."
            )
        return self._delta_from_edges(
            hidden, encoded.cn_hidden, encoded.edge_features
        )

    # -------------------------------------------------------------- forward

    def extra_inputs(self, trace: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Architecture-specific extra forward kwargs drawn from the BP trace.

        Plain node-feature architectures need nothing; edge-conditioned ones
        override this to pull the directed BP messages out of the trace.
        """
        return {}

    def _check_inputs(self, vn_features: Tensor, cn_features: Tensor) -> None:
        if vn_features.ndim != 3 or cn_features.ndim != 3:
            raise ValueError("Node features must be [batch, nodes, channels].")
        if vn_features.shape[1] != self.graph.n_variables:
            raise ValueError(
                f"Expected {self.graph.n_variables} variable nodes, "
                f"got {vn_features.shape[1]}."
            )
        if cn_features.shape[1] != self.graph.n_checks:
            raise ValueError(
                f"Expected {self.graph.n_checks} check nodes, got {cn_features.shape[1]}."
            )
        if vn_features.shape[-1] != self.vn_dim:
            raise ValueError(
                f"Expected {self.vn_dim} variable channels, got {vn_features.shape[-1]}."
            )
        if cn_features.shape[-1] != self.cn_dim:
            raise ValueError(
                f"Expected {self.cn_dim} check channels, got {cn_features.shape[-1]}."
            )

    def forward(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Optional[Tensor] = None,
        cn_to_vn_edge_index: Optional[Tensor] = None,
        return_gate: bool = False,
        return_stages: bool = False,
        **extra: Tensor,
    ) -> Tensor | CorrectorOutput:
        """Return ``delta_llr`` ``[B, n]``, or a :class:`CorrectorOutput`."""
        self._check_inputs(vn_features, cn_features)
        if vn_to_cn_edge_index is None:
            vn_to_cn_edge_index = self.vn_to_cn
        if cn_to_vn_edge_index is None:
            cn_to_vn_edge_index = self.cn_to_vn

        encoded = self.encode_all(
            vn_features, cn_features, vn_to_cn_edge_index, cn_to_vn_edge_index, **extra
        )
        stages, output_mask = encoded.stages, encoded.mask
        if not stages:
            raise RuntimeError("encode_stages() must return at least one hidden state.")
        hidden = stages[-1]
        if hidden.shape[:2] != vn_features.shape[:2]:
            raise RuntimeError(
                "encode_stages() must return one hidden vector per variable node."
            )

        def masked_delta(state: Tensor) -> Tensor:
            delta = self._delta(state, encoded)
            return delta if output_mask is None else delta * output_mask

        delta_llr = masked_delta(hidden)
        gate_logits = (
            self.gate_head(hidden).squeeze(-1) if self.gate_head is not None else None
        )
        stage_deltas = (
            [masked_delta(stage) for stage in stages[:-1]] + [delta_llr]
            if return_stages
            else None
        )

        if return_gate or return_stages:
            return CorrectorOutput(
                delta_llr=delta_llr,
                gate_logits=gate_logits,
                stage_deltas=stage_deltas,
                output_mask=output_mask,
            )
        return delta_llr

    def correct(self, bp_llr: Tensor, *forward_args, **forward_kwargs) -> Tensor:
        """Convenience: forward, then combine with the configured gate."""
        output = self.forward(*forward_args, return_gate=True, **forward_kwargs)
        return combine(
            bp_llr, output.delta_llr, output.gate_logits, fixed_gate=self.fixed_gate
        )

    @property
    def effective_correction_range(self) -> float:
        """Largest per-bit residual this corrector can actually produce.

        Node mode is bounded by ``correction_scale``; edge mode sums one
        bounded term per incident edge, so the bound scales with the maximum
        variable degree.
        """
        if self.residual_mode == "node":
            return self.correction_scale
        return self.correction_scale * float(self.graph.vn_degrees.max())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def head_flops_per_frame(self) -> float:
        """FLOPs for the residual head(s) and the optional gate head."""
        from .graph_ops import linear_flops_for

        if self.edge_delta_head is not None:
            # Applied once per edge rather than once per node.
            total = sum(
                linear_flops_for(layer, self.graph.n_edges)
                for layer in self.edge_delta_head
                if isinstance(layer, nn.Linear)
            )
        else:
            total = linear_flops_for(self.delta_head, self.graph.n_variables)
        if self.gate_head is not None:
            total += linear_flops_for(self.gate_head, self.graph.n_variables)
        return total

    def estimate_flops_per_frame(self) -> float:
        """Analytic FLOPs for one frame through this corrector.

        Each architecture accounts for its own layers explicitly; a generic
        parameter count cannot know how many times a weight is applied (once
        per node, or once per edge).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement estimate_flops_per_frame()."
        )
