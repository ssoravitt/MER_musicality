"""Tests for CogMucs multi-branch forward/backward pass, causality, masking, and losses."""

import math

import pytest
import torch

from features.tiers import get_branch_indices, get_tier_names
from models.cogmucs import CogMucs
from models.losses import CompositeDynamicVALoss, masked_ccc_loss, static_4q_loss


@pytest.mark.parametrize("tier", [1, 2, 3])
@pytest.mark.parametrize("branch_type", ["tcn", "gru"])
@pytest.mark.parametrize("fusion", ["early", "mid", "late"])
@pytest.mark.parametrize("task", ["static_4q", "dynamic_va"])
def test_cogmucs_forward_backward_all_variants(
    tier: int, branch_type: str, fusion: str, task: str
) -> None:
    torch.manual_seed(0)
    branch_indices = get_branch_indices(tier)
    n_ch = len(get_tier_names(tier))
    batch, t_len = 4, 50  # 5.0 s at 10 Hz

    x = torch.randn(batch, t_len, n_ch)
    mask = torch.ones(batch, t_len, dtype=torch.bool)
    mask[0, 35:] = False  # simulate variable-length padded track

    model = CogMucs(
        branch_indices=branch_indices,
        branch_type=branch_type,
        fusion=fusion,
        task=task,
        hidden_dim=16,
        dilations=(1, 2, 4),
        dynamic_stride=5,
    )
    out = model(x, mask=mask)
    assert torch.all(torch.isfinite(out.pred))

    if task == "static_4q":
        assert out.pred.shape == (batch, 4)
        labels = torch.randint(0, 4, (batch,))
        loss = static_4q_loss(out.pred, labels)
    else:
        assert out.pred.shape == (batch, 10, 2)  # 50 / 5 = 10 steps at 2 Hz
        target = torch.randn(batch, 10, 2)
        t_mask = torch.ones(batch, 10, dtype=torch.bool)
        t_mask[:, :3] = False  # warmup mask
        loss = CompositeDynamicVALoss()(out.pred, target, mask=t_mask)

    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert sum(grad_norms) > 0.0


@pytest.mark.parametrize("branch_type", ["tcn", "gru"])
@pytest.mark.parametrize("fusion", ["early", "mid", "late"])
def test_cogmucs_strict_dynamic_causality(branch_type: str, fusion: str) -> None:
    """Perturbing features at t >= 20 (t > 2.0s) must not alter 2 Hz predictions for t <= 2.0s."""
    torch.manual_seed(42)
    branch_indices = get_branch_indices(3)
    n_ch = len(get_tier_names(3))
    model = CogMucs(
        branch_indices=branch_indices,
        branch_type=branch_type,
        fusion=fusion,
        task="dynamic_va",
        hidden_dim=16,
        dilations=(1, 2, 4),
        dynamic_stride=5,
    )
    model.eval()

    x_a = torch.randn(2, 40, n_ch)
    x_b = x_a.clone()
    x_b[:, 20:, :] += 50.0  # perturb strictly after index 19 (t=2.0s endpoint)

    with torch.no_grad():
        pred_a = model(x_a).pred  # [2, 8, 2] -> indices 0..3 cover steps 0..19 (<=2.0s)
        pred_b = model(x_b).pred

    torch.testing.assert_close(pred_a[:, :4, :], pred_b[:, :4, :], atol=1e-5, rtol=1e-5)
    assert torch.max(torch.abs(pred_a[:, 4:, :] - pred_b[:, 4:, :])).item() > 1e-3


def test_cogmucs_padding_mask_invariance() -> None:
    """Garbage values in masked padding frames (`mask=False`) must have zero effect on logits."""
    torch.manual_seed(7)
    branch_indices = get_branch_indices(3)
    n_ch = len(get_tier_names(3))
    model = CogMucs(
        branch_indices=branch_indices,
        branch_type="tcn",
        fusion="mid",
        task="static_4q",
        hidden_dim=16,
        dilations=(1, 2, 4),
    )
    model.eval()

    x_clean = torch.randn(2, 40, n_ch)
    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[:, 25:] = False

    x_garbage = x_clean.clone()
    x_garbage[:, 25:, :] = 1e6

    with torch.no_grad():
        out_clean = model(x_clean, mask=mask).pred
        out_garbage = model(x_garbage, mask=mask).pred

    torch.testing.assert_close(out_clean, out_garbage, atol=1e-5, rtol=1e-5)


def test_initial_loss_calibration_and_ccc_sanity() -> None:
    """Normalized inputs yield CE loss near ln(4) ≈ 1.386 (fixing June 2024 >150 loss bug)."""
    torch.manual_seed(11)
    branch_indices = get_branch_indices(3)
    n_ch = len(get_tier_names(3))
    model = CogMucs(branch_indices=branch_indices, branch_type="tcn", fusion="mid", task="static_4q")
    x = torch.randn(16, 40, n_ch)
    labels = torch.randint(0, 4, (16,))
    loss = static_4q_loss(model(x).pred, labels).item()
    assert 1.0 < loss < 2.0  # centered around math.log(4) = 1.3863
    assert abs(loss - math.log(4)) < 0.6

    # CCC loss sanity: perfect match -> 0, inverted -> > 0.9
    target = torch.linspace(-1.0, 1.0, 30).view(1, 15, 2)
    t_mask = torch.ones(1, 15, dtype=torch.bool)
    assert masked_ccc_loss(target, target, mask=t_mask).item() < 1e-5
    assert masked_ccc_loss(-target, target, mask=t_mask).item() > 0.9
