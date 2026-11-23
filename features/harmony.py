"""Branch 4: Harmony, Tonality & Consonance causal extractors (6 channels)."""

import numpy as np

from features.clock import (
    BASE_HZ,
    SharedCausalSpectra,
    causal_ema,
    causal_rolling_mean,
)

# Krumhansl-Schmuckler (1990) major and minor key probe-tone profiles
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def _build_normalized_key_templates(profile: np.ndarray) -> np.ndarray:
    """Create 12 rotated zero-mean unit-norm key correlation templates `[12, 12]`."""
    templates = np.stack([np.roll(profile, shift) for shift in range(12)], axis=0)
    templates -= templates.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12
    return templates / norms


_MAJOR_TEMPLATES = _build_normalized_key_templates(_KRUMHANSL_MAJOR)
_MINOR_TEMPLATES = _build_normalized_key_templates(_KRUMHANSL_MINOR)


def _build_tonnetz_matrix() -> np.ndarray:
    """Harte et al. (2006) 6x12 harmonic Tonnetz projection matrix."""
    c = np.arange(12, dtype=np.float64)
    r1, r2, r3 = 1.0, 1.0, 0.5
    return np.stack(
        [
            r1 * np.sin((7.0 * np.pi / 6.0) * c),
            r1 * np.cos((7.0 * np.pi / 6.0) * c),
            r2 * np.sin((3.0 * np.pi / 2.0) * c),
            r2 * np.cos((3.0 * np.pi / 2.0) * c),
            r3 * np.sin((2.0 * np.pi / 3.0) * c),
            r3 * np.cos((2.0 * np.pi / 3.0) * c),
        ],
        axis=0,
    )


_TONNETZ_MATRIX = _build_tonnetz_matrix()


def _plomp_levelt_dissonance(
    mag_4096: np.ndarray,
    freqs_4096: np.ndarray,
    max_freq: float = 5000.0,
    top_k: int = 12,
) -> np.ndarray:
    """Compute Plomp-Levelt / Sethares pairwise critical-band spectral roughness per frame."""
    band = freqs_4096 <= max_freq
    mag = mag_4096[band, :]
    freqs = freqs_4096[band]
    n_bins, n_frames = mag.shape
    out = np.zeros(n_frames, dtype=np.float32)
    if n_bins < 3:
        return out

    # Vectorized peak mask along frequency bins (frame-local -> strictly causal)
    is_peak = np.zeros_like(mag, dtype=bool)
    is_peak[1:-1, :] = (mag[1:-1, :] > mag[:-2, :]) & (mag[1:-1, :] >= mag[2:, :])
    frame_max = mag.max(axis=0, keepdims=True)
    is_peak &= mag >= (0.05 * frame_max)

    for t in range(n_frames):
        idx = np.flatnonzero(is_peak[:, t])
        if len(idx) < 2:
            continue
        if len(idx) > top_k:
            amps_cand = mag[idx, t]
            top_order = np.argpartition(amps_cand, -top_k)[-top_k:]
            idx = idx[top_order]

        f = freqs[idx].astype(np.float64)
        a = mag[idx, t].astype(np.float64)
        denom = float(np.sum(a ** 2)) + 1e-10

        f_i = f[:, None]
        f_j = f[None, :]
        a_prod = a[:, None] * a[None, :]
        f_min = np.minimum(f_i, f_j)
        df = np.abs(f_i - f_j)
        s = 0.24 / (0.021 * f_min + 19.0)
        d_mat = a_prod * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
        tri_u = np.triu(d_mat, k=1)
        out[t] = np.float32(np.sum(tri_u) / denom)

    return out


def extract_harmony_features(spectra: SharedCausalSpectra) -> dict[str, np.ndarray]:
    """Extract all 6 Branch-4 Harmony, Tonality & Consonance channels at base 50 Hz rate."""
    n_frames = spectra.n_frames

    # 1. Plomp-Levelt / Sethares sensory dissonance
    sensory_dissonance_plomp = _plomp_levelt_dissonance(spectra.mag_4096, spectra.freqs_4096)

    # 2. Log harmonic-to-percussive energy ratio from causal STFT HPSS
    e_harm = np.sum(spectra.harm_mag_2048.astype(np.float64) ** 2, axis=0)
    e_perc = np.sum(spectra.perc_mag_2048.astype(np.float64) ** 2, axis=0)
    harmonic_to_percussive_ratio = np.log((e_harm + 1e-8) / (e_perc + 1e-8)).astype(np.float32)

    # 3. Tonnetz velocity (harmonic change rate)
    tonnetz = _TONNETZ_MATRIX @ spectra.chroma_12.astype(np.float64)
    tonal_centroid_flux = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        tonal_centroid_flux[1:] = np.linalg.norm(
            tonnetz[:, 1:] - tonnetz[:, :-1], axis=0
        ).astype(np.float32)

    # 4-5. Causal 3.0s Krumhansl-Schmuckler major/minor mode score & key clarity
    chroma_3s = causal_rolling_mean(spectra.chroma_12, win=round(3.0 * BASE_HZ)).astype(
        np.float64
    )
    chroma_c = chroma_3s - chroma_3s.mean(axis=0, keepdims=True)
    chroma_norm = np.linalg.norm(chroma_c, axis=0, keepdims=True)
    chroma_unit = np.where(chroma_norm > 1e-8, chroma_c / (chroma_norm + 1e-12), 0.0)

    maj_corr = (_MAJOR_TEMPLATES @ chroma_unit).max(axis=0)
    min_corr = (_MINOR_TEMPLATES @ chroma_unit).max(axis=0)
    mode_major_minor_score = (maj_corr - min_corr).astype(np.float32)
    key_clarity = np.maximum(maj_corr, min_corr).astype(np.float32)

    # 6. Causal harmonic expectancy surprise (ITPRA tonal violation against τ=3.0s EMA)
    tau_frames = 3.0 * BASE_HZ
    ema_chroma = causal_ema(spectra.chroma_12, tau_frames=tau_frames).astype(np.float64)
    cur_c = spectra.chroma_12.astype(np.float64)
    harmonic_expectancy_surprise = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        prev_mem = ema_chroma[:, :-1]
        now_c = cur_c[:, 1:]
        dot = np.sum(now_c * prev_mem, axis=0)
        denom = (np.linalg.norm(now_c, axis=0) * np.linalg.norm(prev_mem, axis=0)) + 1e-10
        cos_sim = np.clip(dot / denom, -1.0, 1.0)
        harmonic_expectancy_surprise[1:] = (1.0 - cos_sim).astype(np.float32)

    return {
        "sensory_dissonance_plomp": sensory_dissonance_plomp,
        "harmonic_to_percussive_ratio": harmonic_to_percussive_ratio,
        "tonal_centroid_flux": tonal_centroid_flux,
        "mode_major_minor_score": mode_major_minor_score,
        "key_clarity": key_clarity,
        "harmonic_expectancy_surprise": harmonic_expectancy_surprise,
    }
