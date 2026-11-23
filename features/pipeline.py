"""End-to-end cognitive feature extraction pipeline and NPZ caching."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from features.backends.essentia_backend import extract_all_channels_essentia
from features.backends.librosa_backend import extract_all_channels_librosa
from features.clock import BASE_HZ, SR, causal_downsample
from features.tiers import get_branch_indices, get_tier_names


@dataclass
class FeatureOutput:
    """Container for extracted time-series cognitive features."""

    features: np.ndarray  # [T_out, C]
    feature_names: list[str]
    branch_indices: dict[str, list[int]]
    times_s: np.ndarray  # [T_out]
    tier: int
    target_hz: float


def extract_features(
    y: np.ndarray,
    sr: int = SR,
    tier: int = 3,
    target_hz: float = 10.0,
    backend: str = "librosa",
) -> FeatureOutput:
    """Extract strictly causal tier-selected musicality trajectories from waveform `y`."""
    names = get_tier_names(tier)
    branch_indices = get_branch_indices(tier)

    if backend == "librosa":
        channel_map = extract_all_channels_librosa(y, sr=sr, tier=tier)
    elif backend == "essentia":
        channel_map = extract_all_channels_essentia(y, sr=sr, tier=tier)
    else:
        raise ValueError(f"Unsupported backend={backend!r}; expected 'librosa' or 'essentia'.")

    base_matrix = np.stack([np.asarray(channel_map[name], dtype=np.float32) for name in names], axis=-1)
    base_matrix = np.nan_to_num(base_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    features, times_s = causal_downsample(base_matrix, base_hz=BASE_HZ, target_hz=target_hz)
    return FeatureOutput(
        features=features,
        feature_names=names,
        branch_indices=branch_indices,
        times_s=times_s,
        tier=tier,
        target_hz=target_hz,
    )


def extract_file(
    audio_path: str | Path,
    tier: int = 3,
    target_hz: float = 10.0,
    backend: str = "librosa",
    cache_path: str | Path | None = None,
) -> FeatureOutput:
    """Load an audio file at 22050 Hz mono, extract causal features, and optionally cache `.npz`."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    out = extract_features(y, sr=sr, tier=tier, target_hz=target_hz, backend=backend)

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            features=out.features,
            times_s=out.times_s,
            feature_names=np.array(out.feature_names),
            tier=np.array(out.tier, dtype=np.int32),
            target_hz=np.array(out.target_hz, dtype=np.float32),
        )
    return out
