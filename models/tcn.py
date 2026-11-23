"""Dilated Causal Temporal Convolutional Network (TCN) branch encoder."""

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    """1D convolution padded strictly on the left so output at `t` sees only `<= t`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal conv to `[B, C, T]` tensor."""
        if self.pad_left > 0:
            x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)


class CausalTCNBlock(nn.Module):
    """Residual dilated causal conv block with frame-wise LayerNorm and GELU."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for `[B, C, T]` tensor preserving strict causality."""
        h = self.conv1(x)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.act(h))
        h = self.conv2(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.act(h))
        return x + h


class CausalTCNBranch(nn.Module):
    """Per-cognitive-element dilated causal TCN encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=int(d),
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Encode `[B, T, in_channels]` -> `([B, T, hidden_dim], None)`."""
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        h = torch.relu(self.in_norm(self.in_proj(x)))
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        h_ct = h.transpose(1, 2)  # [B, D, T]
        for block in self.blocks:
            h_ct = block(h_ct)
        out = h_ct.transpose(1, 2)  # [B, T, D]
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out, None
