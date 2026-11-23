"""Nested 3-tier feature subset selection and branch index mapping."""

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


def get_branch_indices(tier: int) -> dict[str, list[int]]:
    """Map each active cognitive branch to channel indices within the tier tensor."""
    specs = get_tier_specs(tier)
    mapping: dict[str, list[int]] = {}
    for branch in BRANCH_NAMES:
        idxs = [i for i, spec in enumerate(specs) if spec.branch == branch]
        if idxs:
            mapping[branch] = idxs
    return mapping
