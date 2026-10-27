"""Convolutional (ResNet) residual correctors -- the non-graph comparison.

Same contract as every graph corrector: read BP(k)'s trace features, emit one
bounded ``delta_llr`` per bit through the shared zero-initialized head, so an
untrained CNN is exactly BP(k).  Only the encoder differs.

Two ways to put a convolution on an LDPC code:

``resnet1d``
    A 1-D ResNet over the bit sequence ``[B, vn_dim, n]``.  This is the plain
    "CNN instead of a GNN" baseline: the network sees the per-bit features
    every corrector gets (which already include the mean violation of each
    bit's checks), but *no* Tanner-graph connectivity.  Convolution assumes
    neighbouring indices are related; on an LDPC code bit ``i`` and bit
    ``i+1`` usually share no check, so locality is an assumption the code
    does not satisfy.  Dilations grow the receptive field past ``n`` so every
    bit can still see every other one.

``resnet2d``
    A 2-D ResNet over the parity-check matrix as an image ``[B, C, m, n]``.
    Cell ``(j, i)`` holds an ``H`` mask channel plus bit ``i``'s and check
    ``j``'s features where ``H[j, i] = 1`` and zeros elsewhere, so the CNN
    *does* see which bits share a check.  Rows and columns of ``H`` are still
    in arbitrary order, so 3x3 locality is again only partly meaningful.  Per
    bit, the column is pooled (masked mean over its checks, plus plain mean)
    back to one hidden vector.

Both use circular padding: bit and check indices have no natural boundary, so
every position gets the same amount of context instead of zero-padded edges.
Both read features only, never the edge-index tensors.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn

from ..tanner_graph import TannerGraph
from .base_corrector import BaseCorrector
from .graph_ops import linear_flops

DEFAULT_DILATIONS = (1, 2, 4, 8)


def _clamp_dilations(
    dilations: Sequence[int], half_kernel: int, length: int
) -> List[int]:
    """Cap each dilation so circular padding never wraps more than once.

    Circular padding of ``d * half_kernel`` needs ``<= length`` elements to
    copy from; beyond that the receptive field already covers the axis.
    """
    largest = max(1, length // max(half_kernel, 1))
    return [max(1, min(int(d), largest)) for d in dilations]


class ResidualBlock1D(nn.Module):
    """conv -> BN -> SiLU -> conv -> BN, plus identity skip, then SiLU."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        conv = dict(
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="circular",
            bias=False,
        )
        self.conv1 = nn.Conv1d(channels, channels, **conv)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, **conv)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop(self.activation(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        return self.activation(out + x)


class ResidualBlock2D(nn.Module):
    """2-D counterpart of :class:`ResidualBlock1D` with 3x3 kernels."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        conv = dict(
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
            bias=False,
        )
        self.conv1 = nn.Conv2d(channels, channels, **conv)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, **conv)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop(self.activation(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        return self.activation(out + x)


class ResNet1DCorrector(BaseCorrector):
    """1-D ResNet over the bit sequence; no Tanner-graph connectivity."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        use_gate: bool = False,
        fixed_gate: float = 1.0,
        correction_scale: float = 8.0,
        dropout: float = 0.0,
        residual_mode: str = "node",
    ) -> None:
        # No per-edge context, so edge_dim stays 0 and the base class rejects
        # residual_mode='edge' with an explanatory error.
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
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3.")
        if not dilations:
            raise ValueError("dilations must be non-empty.")

        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        cycle = _clamp_dilations(
            dilations, (self.kernel_size - 1) // 2, graph.n_variables
        )
        # Dilations repeat if there are more blocks than listed values.
        self.block_dilations = [cycle[i % len(cycle)] for i in range(self.num_blocks)]

        self.stem = nn.Conv1d(vn_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualBlock1D(hidden_dim, self.kernel_size, dilation, dropout)
            for dilation in self.block_dilations
        )

    @property
    def receptive_field(self) -> int:
        """Bits each output position can see (before circular wrap-around)."""
        return 1 + sum(2 * d * (self.kernel_size - 1) for d in self.block_dilations)

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        x = self.stem(vn_features.transpose(1, 2))  # [B, hidden, n]
        stages: List[Tensor] = []
        for block in self.blocks:
            x = block(x)
            stages.append(x.transpose(1, 2))  # [B, n, hidden]
        return stages

    def estimate_flops_per_frame(self) -> float:
        n_vn = self.graph.n_variables
        width = self.hidden_dim
        total = linear_flops(self.vn_dim, width, n_vn)  # 1x1 stem
        # Two k-tap convolutions per block. BatchNorm and SiLU are O(C * n)
        # and omitted, as norms are elsewhere in this package.
        total += self.num_blocks * 2 * linear_flops(width * self.kernel_size, width, n_vn)
        return total + self.head_flops_per_frame()


class ResNet2DCorrector(BaseCorrector):
    """2-D ResNet over the parity-check matrix rendered as an image."""

    def __init__(
        self,
        graph: TannerGraph,
        vn_dim: int,
        cn_dim: int,
        hidden_dim: int = 16,
        num_blocks: int = 4,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
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
            edge_dim=0,
        )
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        if not dilations:
            raise ValueError("dilations must be non-empty.")

        self.num_blocks = int(num_blocks)
        shortest_axis = min(graph.n_checks, graph.n_variables)
        cycle = _clamp_dilations(dilations, 1, shortest_axis)
        self.block_dilations = [cycle[i % len(cycle)] for i in range(self.num_blocks)]

        # [m, n] 0/1 image of H. Dense is fine: this corrector is for short
        # codes; a 5G-scale H would make the image itself the bottleneck.
        mask = graph.to_dense().to(torch.float32)
        self.register_buffer("h_mask", mask, persistent=False)
        self.register_buffer(
            "checks_per_bit", mask.sum(dim=0).clamp_min(1.0), persistent=False
        )

        self.in_channels = 1 + vn_dim + cn_dim
        self.stem = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualBlock2D(hidden_dim, dilation, dropout)
            for dilation in self.block_dilations
        )
        # Column pooling gives [masked mean over the bit's checks, plain mean]
        # per channel; one shared readout maps that to the bit's hidden state.
        self.readout = nn.Linear(2 * hidden_dim, hidden_dim)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(4 * d for d in self.block_dilations)

    def build_image(self, vn_features: Tensor, cn_features: Tensor) -> Tensor:
        """``[B, 1 + vn_dim + cn_dim, m, n]``: H mask, then edge-masked features."""
        batch = vn_features.shape[0]
        mask = self.h_mask.view(1, 1, *self.h_mask.shape)
        vn_plane = vn_features.transpose(1, 2).unsqueeze(2) * mask  # [B, Fv, m, n]
        cn_plane = cn_features.transpose(1, 2).unsqueeze(3) * mask  # [B, Fc, m, n]
        return torch.cat((mask.expand(batch, -1, -1, -1), vn_plane, cn_plane), dim=1)

    def _pool_columns(self, x: Tensor) -> Tensor:
        """``[B, C, m, n]`` -> ``[B, n, hidden]`` per-bit hidden state."""
        masked = (x * self.h_mask).sum(dim=2) / self.checks_per_bit
        pooled = torch.cat((masked, x.mean(dim=2)), dim=1)  # [B, 2C, n]
        return self.readout(pooled.transpose(1, 2))

    def encode_stages(
        self,
        vn_features: Tensor,
        cn_features: Tensor,
        vn_to_cn_edge_index: Tensor,
        cn_to_vn_edge_index: Tensor,
        **extra: Tensor,
    ) -> List[Tensor]:
        x = self.stem(self.build_image(vn_features, cn_features))
        stages: List[Tensor] = []
        for block in self.blocks:
            x = block(x)
            stages.append(self._pool_columns(x))
        return stages

    def estimate_flops_per_frame(self) -> float:
        cells = self.graph.n_checks * self.graph.n_variables
        width = self.hidden_dim
        total = linear_flops(self.in_channels, width, cells)  # 1x1 stem
        total += self.num_blocks * 2 * linear_flops(width * 9, width, cells)
        # Pooling is two reductions over the check axis per channel.
        total += 3.0 * width * cells
        total += linear_flops(2 * width, width, self.graph.n_variables)
        return total + self.head_flops_per_frame()
