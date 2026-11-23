"""Nested 3-tier feature subset selection and branch index mapping."""

import numpy as np

from features.schema import BRANCH_NAMES, FEATURE_SCHEMA, FeatureSpec

VALID_TIERS: tuple[int, ...] = (1, 2, 3)


def get_tier_specs(tier: int) -> tuple[FeatureSpec, ...]:
    """Return the ordered tuple of FeatureSpec items included in `tier` (1, 2, or 3)."""
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier={tier}; expected one of {VALID_TIERS}.")
    return tuple(spec for spec in FEATURE_SCHEMA if spec.min_tier <= tier)


def get_tier_names(tier: int) -> list[str]:
    """Return ordered feature channel names for `tier`."""
    return [spec.name for spec in get_tier_specs(tier)]


def get_tier_indices_in_full(target_tier: int) -> list[int]:
    """Return column indices inside the 23-channel Tier-3 tensor for `target_tier`."""
    if target_tier not in VALID_TIERS:
        raise ValueError(f"Invalid target_tier={target_tier}; expected one of {VALID_TIERS}.")
    return [idx for idx, spec in enumerate(FEATURE_SCHEMA) if spec.min_tier <= target_tier]


def slice_tier_from_full(tier3_features: np.ndarray, target_tier: int) -> np.ndarray:
    """Slice a cached Tier-3 `[..., 23]` array to match `target_tier` (1, 2, or 3)."""
    if tier3_features.shape[-1] != len(FEATURE_SCHEMA):
        raise ValueError(
            f"Expected last dim {len(FEATURE_SCHEMA)} for Tier-3 features, got {tier3_features.shape[-1]}."
        )
    if target_tier == 3:
        return tier3_features
    idxs = get_tier_indices_in_full(target_tier)
    return tier3_features[..., idxs]


def get_branch_indices(tier: int) -> dict[str, list[int]]:
    """Map each active cognitive branch to channel indices within the tier tensor."""
    specs = get_tier_specs(tier)
    mapping: dict[str, list[int]] = {}
    for branch in BRANCH_NAMES:
        idxs = [i for i, spec in enumerate(specs) if spec.branch == branch]
        if idxs:
            mapping[branch] = idxs
    return mapping
