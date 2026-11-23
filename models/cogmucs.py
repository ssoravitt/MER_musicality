"""CogMucs: Cognitive Multi-Branch Temporal Network for Music Emotion Recognition."""

from dataclasses import dataclass

import torch
from torch import nn

from models.gru import FeatureGatedGRUBranch
from models.tcn import CausalTCNBranch


@dataclass
class CogMucsOutput:
    """Structured forward output with predictions and interpretability weights."""

    pred: torch.Tensor  # [B, 4] or [B, 2] (static) or [B, T_out, 2] (dynamic)
    branch_weights: torch.Tensor  # [B, n_branches] or [B, T, n_branches]
    feature_gates: dict[str, torch.Tensor]


class _MaskedAttentionPool(nn.Module):
    """Frame-level masked attention pooling over `[B, T, D]` -> `[B, D]`."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(h).squeeze(-1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # [B, T, 1]
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).to(weights.dtype)
        return torch.sum(h * weights, dim=1)


class CogMucs(nn.Module):
    """Multi-branch temporal model decomposing music into cognitive elements.

    Replaces Demucs instrument stems from EmoMucs (de Berardinis et al., 2020)
    with biologically grounded musical elements (Melody, Rhythm, Timbre, Harmony),
    supporting Early (`early`), Mid (`mid`), and Late (`late`) fusion as well as
    both static (4Q / V-A) and causal dynamic (`2 Hz` V-A) readouts.
    """

    def __init__(
        self,
        branch_indices: dict[str, list[int]],
        branch_type: str = "tcn",
        fusion: str = "mid",
        task: str = "static_4q",
        hidden_dim: int = 32,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
        kernel_size: int = 3,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.2,
        dynamic_stride: int = 5,
    ) -> None:
        super().__init__()
        if not branch_indices:
            raise ValueError("branch_indices must contain at least one branch.")
        if branch_type not in ("tcn", "gru"):
            raise ValueError(f"Unsupported branch_type={branch_type!r}.")
        if fusion not in ("early", "mid", "late"):
            raise ValueError(f"Unsupported fusion={fusion!r}.")
        if task not in ("static_4q", "static_va", "dynamic_va"):
            raise ValueError(f"Unsupported task={task!r}.")

        self.branch_names = list(branch_indices.keys())
        self.branch_indices = {k: list(v) for k, v in branch_indices.items()}
        self.branch_type = branch_type
        self.fusion = fusion
        self.task = task
        self.hidden_dim = hidden_dim
        self.dynamic_stride = dynamic_stride
        self.out_dim = 4 if task == "static_4q" else 2

        self.branches = nn.ModuleDict()
        for name in self.branch_names:
            in_ch = len(self.branch_indices[name])
            if branch_type == "tcn":
                self.branches[name] = CausalTCNBranch(
                    in_channels=in_ch,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
            else:
                self.branches[name] = FeatureGatedGRUBranch(
                    in_channels=in_ch,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )

        n_branches = len(self.branch_names)
        if fusion == "early":
            self.early_proj = nn.Sequential(
                nn.Linear(n_branches * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.early_gate = nn.Linear(n_branches * hidden_dim, n_branches)
        elif fusion == "mid":
            self.mid_projs = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                    )
                    for name in self.branch_names
                }
            )
            self.mid_gates = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, 1) for name in self.branch_names}
            )
        else:  # late
            self.late_heads = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, self.out_dim) for name in self.branch_names}
            )
            self.late_gates = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, 1) for name in self.branch_names}
            )

        self.is_static = task in ("static_4q", "static_va")
        if self.is_static:
            if fusion == "late":
                self.branch_pools = nn.ModuleDict(
                    {name: _MaskedAttentionPool(hidden_dim) for name in self.branch_names}
                )
            else:
                self.pool = _MaskedAttentionPool(hidden_dim)

        if fusion != "late":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.out_dim),
            )

    def _causal_subsample(
        self, seq: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select strictly causal endpoint of each `dynamic_stride` block (`10 Hz -> 2 Hz`)."""
        t_len = seq.shape[1]
        if self.dynamic_stride <= 1:
            return seq
        idx = torch.arange(self.dynamic_stride - 1, t_len, self.dynamic_stride, device=seq.device)
        if len(idx) == 0:
            idx = torch.tensor([t_len - 1], device=seq.device)
        return seq.index_select(1, idx)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> CogMucsOutput:
        """Run multi-branch forward pass over `[B, T, C]` cognitive trajectories."""
        if mask is not None:
            mask = mask.bool()
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        branch_seq: dict[str, torch.Tensor] = {}
        feature_gates: dict[str, torch.Tensor] = {}
        for name in self.branch_names:
            idx = self.branch_indices[name]
            x_b = x[..., idx]
            h_b, g_b = self.branches[name](x_b, mask=mask)
            branch_seq[name] = h_b
            if g_b is not None:
                feature_gates[name] = g_b

        if self.fusion == "early":
            cat = torch.cat([branch_seq[n] for n in self.branch_names], dim=-1)
            branch_weights = torch.softmax(self.early_gate(cat), dim=-1)
            fused = self.early_proj(cat)
            if mask is not None:
                fused = fused.masked_fill(~mask.unsqueeze(-1), 0.0)
            if self.is_static:
                pred = self.head(self.pool(fused, mask=mask))
            else:
                pred = self.head(self._causal_subsample(fused, mask=mask))
            return CogMucsOutput(pred=pred, branch_weights=branch_weights, feature_gates=feature_gates)

        if self.fusion == "mid":
            projs = torch.stack([self.mid_projs[n](branch_seq[n]) for n in self.branch_names], dim=2)
            gate_logits = torch.cat(
                [self.mid_gates[n](branch_seq[n]) for n in self.branch_names], dim=-1
            )  # [B, T, n_branches]
            branch_weights = torch.softmax(gate_logits, dim=-1)
            fused = torch.sum(projs * branch_weights.unsqueeze(-1), dim=2)  # [B, T, D]
            if mask is not None:
                fused = fused.masked_fill(~mask.unsqueeze(-1), 0.0)
            if self.is_static:
                pred = self.head(self.pool(fused, mask=mask))
                if mask is not None:
                    valid_f = mask.unsqueeze(-1).to(branch_weights.dtype)
                    mean_bw = (branch_weights * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
                else:
                    mean_bw = branch_weights.mean(dim=1)
                branch_weights = mean_bw
            else:
                pred = self.head(self._causal_subsample(fused, mask=mask))
            return CogMucsOutput(pred=pred, branch_weights=branch_weights, feature_gates=feature_gates)

        # Late (decision-level) fusion
        if self.is_static:
            pooled = {n: self.branch_pools[n](branch_seq[n], mask=mask) for n in self.branch_names}
            preds = torch.stack([self.late_heads[n](pooled[n]) for n in self.branch_names], dim=1)
            gate_logits = torch.cat(
                [self.late_gates[n](pooled[n]) for n in self.branch_names], dim=-1
            )
            branch_weights = torch.softmax(gate_logits, dim=-1)  # [B, n_branches]
            pred = torch.sum(preds * branch_weights.unsqueeze(-1), dim=1)
            return CogMucsOutput(pred=pred, branch_weights=branch_weights, feature_gates=feature_gates)

        sub_seq = {n: self._causal_subsample(branch_seq[n], mask=mask) for n in self.branch_names}
        preds = torch.stack([self.late_heads[n](sub_seq[n]) for n in self.branch_names], dim=2)
        gate_logits = torch.cat(
            [self.late_gates[n](sub_seq[n]) for n in self.branch_names], dim=-1
        )
        branch_weights = torch.softmax(gate_logits, dim=-1)  # [B, T_sub, n_branches]
        pred = torch.sum(preds * branch_weights.unsqueeze(-1), dim=2)
        return CogMucsOutput(pred=pred, branch_weights=branch_weights, feature_gates=feature_gates)
