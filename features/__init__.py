"""Cognition-first, strictly causal musicality feature extraction package."""

from features.pipeline import FeatureOutput, extract_features, extract_file
from features.schema import BRANCH_NAMES, FEATURE_SCHEMA, FeatureSpec
from features.tiers import get_branch_indices, get_tier_names, get_tier_specs

__all__ = [
    "BRANCH_NAMES",
    "FEATURE_SCHEMA",
    "FeatureOutput",
    "FeatureSpec",
    "extract_features",
    "extract_file",
    "get_branch_indices",
    "get_tier_names",
    "get_tier_specs",
]
