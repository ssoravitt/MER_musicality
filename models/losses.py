"""Loss functions for static 4Q classification and masked dynamic V-A regression."""

import torch
import torch.nn.functional as F
from torch import nn


def static_4q_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard 4-quadrant Cross-Entropy loss."""
    return F.cross_entropy(logits, labels.long())


def masked_ccc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute `1 - Lin's CCC` averaged across output dimensions over valid masked frames."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(-1)
        target = target.unsqueeze(-1)

    if mask is not None:
        valid = mask.bool()
        p_valid = pred[valid]  # [N_valid, D]
        t_valid = target[valid]
    else:
        p_valid = pred.reshape(-1, pred.shape[-1])
        t_valid = target.reshape(-1, target.shape[-1])

    if p_valid.shape[0] < 2:
        return F.mse_loss(p_valid, t_valid)

    mu_p = p_valid.mean(dim=0)
    mu_t = t_valid.mean(dim=0)
    p_c = p_valid - mu_p
    t_c = t_valid - mu_t

    var_p = torch.mean(p_c * p_c, dim=0)
    var_t = torch.mean(t_c * t_c, dim=0)
    cov = torch.mean(p_c * t_c, dim=0)

    ccc = (2.0 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + eps)
    return torch.mean(1.0 - ccc)


class CompositeDynamicVALoss(nn.Module):
    """Masked `(1 - CCC) + mse_weight * MSE` loss for 2 Hz continuous Valence-Arousal."""

    def __init__(self, mse_weight: float = 0.25) -> None:
        super().__init__()
        self.mse_weight = mse_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ccc_l = masked_ccc_loss(pred, target, mask=mask)
        if mask is not None:
            valid = mask.bool()
            mse_l = F.mse_loss(pred[valid], target[valid])
        else:
            mse_l = F.mse_loss(pred, target)
        return ccc_l + self.mse_weight * mse_l
