"""Branch 3: Timbre, Brightness & Loudness Dynamics causal extractors (5 channels)."""

import librosa
import numpy as np

from features.clock import (
    BASE_HZ,
    SharedCausalSpectra,
    causal_rolling_slope,
    hz_to_bark,
)


def extract_timbre_features(spectra: SharedCausalSpectra) -> dict[str, np.ndarray]:
    """Extract all 5 Branch-3 Timbre & Loudness channels at base 50 Hz rate."""
    mag = spectra.mag_2048
    freqs = spectra.freqs_2048

    # 1. A-weighted perceptual frame loudness (dB with fixed causal ref=1.0)
    a_db = librosa.A_weighting(np.maximum(freqs, 1.0), min_db=-80.0)
    a_lin = (10.0 ** (a_db / 10.0)).astype(np.float64)
    weighted_power = np.sum(a_lin[:, None] * (mag.astype(np.float64) ** 2), axis=0)
    loudness_perceptual_db = (10.0 * np.log10(weighted_power + 1e-10)).astype(np.float32)

    # 2. Causal 0.50s loudness slope (crescendo / decrescendo)
    loudness_delta = causal_rolling_slope(
        loudness_perceptual_db, win=round(0.50 * BASE_HZ), dt=1.0 / BASE_HZ
    )

    # 3. Spectral centroid mapped to Bark perceptual scale
    mag_sum = mag.sum(axis=0)
    centroid_hz = np.where(
        mag_sum > 1e-8,
        (freqs[:, None] * mag).sum(axis=0) / (mag_sum + 1e-10),
        0.0,
    )
    spectral_centroid_bark = hz_to_bark(centroid_hz)

    # 4. 85% Spectral rolloff mapped to Bark scale (frame-local S=mag -> strictly causal)
    rolloff_hz = librosa.feature.spectral_rolloff(
        S=mag, sr=spectra.sr, roll_percent=0.85
    ).reshape(-1)
    spectral_rolloff_bark = hz_to_bark(rolloff_hz)

    # 5. Wiener spectral flatness (frame-local S=mag -> strictly causal)
    spectral_flatness = librosa.feature.spectral_flatness(S=mag).reshape(-1).astype(np.float32)

    return {
        "loudness_perceptual_db": loudness_perceptual_db,
        "loudness_delta": loudness_delta,
        "spectral_centroid_bark": spectral_centroid_bark,
        "spectral_rolloff_bark": spectral_rolloff_bark,
        "spectral_flatness": spectral_flatness,
    }
