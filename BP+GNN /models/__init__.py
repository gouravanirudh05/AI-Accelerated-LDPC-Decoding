"""Graph-based residual correctors applied on top of a truncated BP(k)."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base_corrector import BaseCorrector, CorrectorOutput, combine
from .cnn_corrector import ResNet1DCorrector, ResNet2DCorrector
from .gat_corrector import GATCorrector
from .gcn_corrector import GCNCorrector
from .graph_transformer_local import GraphTransformerLocal
from .graph_transformer_sparse_global import GraphTransformerSparseGlobal
from .pyg_correctors import (
    ExtrinsicGATConv,
    PyGGATCorrector,
    PyGGCNCorrector,
    PyGGraphTransformerLocal,
)

CORRECTOR_REGISTRY: Dict[str, Type[BaseCorrector]] = {
    "gcn": GCNCorrector,
    "gat": GATCorrector,
    "local_gt": GraphTransformerLocal,
    "sparse_global_gt": GraphTransformerSparseGlobal,
    # PyTorch Geometric implementations. These are the primary ones: the GAT
    # variants use the library's own GATConv / GATv2Conv, so the v1-vs-v2
    # comparison does not depend on my scoring code. The hand-rolled versions
    # above are kept because they carry the earlier trained results.
    "pyg_gcn": PyGGCNCorrector,
    "pyg_gat": PyGGATCorrector,
    "pyg_local_gt": PyGGraphTransformerLocal,
    # Selection plus full attention over a node subset has no graph structure
    # to exploit, so this stays on torch.nn.MultiheadAttention -- the standard
    # implementation for dense attention. Aliased for a symmetric registry.
    "pyg_sparse_global_gt": GraphTransformerSparseGlobal,
    # Convolutional comparison: no message passing. resnet1d sees only the
    # bit sequence; resnet2d sees H rendered as an image.
    "resnet1d": ResNet1DCorrector,
    "resnet2d": ResNet2DCorrector,
}

__all__ = [
    "BaseCorrector",
    "CorrectorOutput",
    "combine",
    "GCNCorrector",
    "GATCorrector",
    "GraphTransformerLocal",
    "GraphTransformerSparseGlobal",
    "PyGGCNCorrector",
    "PyGGATCorrector",
    "PyGGraphTransformerLocal",
    "ExtrinsicGATConv",
    "ResNet1DCorrector",
    "ResNet2DCorrector",
    "CORRECTOR_REGISTRY",
    "build_corrector",
]


def build_corrector(name: str, **kwargs: Any) -> BaseCorrector:
    """Instantiate a corrector by registry name."""
    key = name.lower()
    if key not in CORRECTOR_REGISTRY:
        raise KeyError(
            f"Unknown corrector {name!r}. Available: {sorted(CORRECTOR_REGISTRY)}"
        )
    return CORRECTOR_REGISTRY[key](**kwargs)
