"""Feature-Gated GRU cognitive branch encoder."""

import torch
from torch import nn


class FeatureGatedGRUBranch(nn.Module):
    """Per-timestep feature attention gate followed by a multi-layer GRU."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gate_proj = nn.Linear(in_channels, in_channels)
        self.in_norm = nn.LayerNorm(in_channels)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_proj = (
            nn.Linear(out_dim, hidden_dim) if bidirectional else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `[B, T, in_channels]` -> `([B, T, hidden_dim], feature_gates)`."""
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        gates = torch.sigmoid(self.gate_proj(x))
        x_gated = self.in_norm(x * gates)
        if mask is not None:
            x_gated = x_gated.masked_fill(~mask.unsqueeze(-1), 0.0)
            gates = gates.masked_fill(~mask.unsqueeze(-1), 0.0)

        out, _ = self.gru(x_gated)
        out = self.out_proj(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out, gates
