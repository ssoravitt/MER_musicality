"""Synthetic DSP grounding, tier consistency, and strict causality tests."""

import numpy as np
import pytest

from features.clock import SR
from features.pipeline import extract_features
from features.tiers import get_branch_indices, get_tier_names


def _sine(freq_hz: float, dur_s: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_s * sr), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _chord(freqs: list[float], dur_s: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_s * sr), dtype=np.float32) / sr
    sig = np.zeros_like(t)
    for f in freqs:
        # Include 2nd & 3rd partials for realistic harmonic spectrum
        sig += (
            0.6 * np.sin(2.0 * np.pi * f * t)
            + 0.25 * np.sin(2.0 * np.pi * 2.0 * f * t)
            + 0.15 * np.sin(2.0 * np.pi * 3.0 * f * t)
        )
    sig /= max(np.max(np.abs(sig)), 1e-6)
    return (0.5 * sig).astype(np.float32)


def _click_track(bpm: float, dur_s: float, sr: int = SR) -> np.ndarray:
    n_samples = int(dur_s * sr)
    sig = np.zeros(n_samples, dtype=np.float32)
    step = round((60.0 / bpm) * sr)
    click_len = int(0.025 * sr)
    t_click = np.arange(click_len, dtype=np.float32) / sr
    click = np.exp(-150.0 * t_click) * np.sin(2.0 * np.pi * 1000.0 * t_click)
    for pos in range(0, n_samples - click_len, step):
        sig[pos : pos + click_len] += 0.8 * click
    return sig.astype(np.float32)


@pytest.mark.parametrize("tier,expected_ch", [(1, 4), (2, 6), (3, 23)])
def test_tier_shapes_and_no_nans(tier: int, expected_ch: int) -> None:
    y = _sine(440.0, dur_s=3.0)
    out = extract_features(y, sr=SR, tier=tier, target_hz=10.0)
    assert out.features.shape == (30, expected_ch)
    assert len(out.feature_names) == expected_ch
    assert out.times_s.shape == (30,)
    assert np.all(np.isfinite(out.features))
    branch_map = get_branch_indices(tier)
    flat_idx = sorted(i for idxs in branch_map.values() for i in idxs)
    assert flat_idx == list(range(expected_ch))


def test_strict_dsp_causality() -> None:
    """Perturbing audio at t > 4.0 s must NOT alter any feature at t <= 4.0 s."""
    rng = np.random.default_rng(7)
    dur_s = 8.0
    y_clean = _chord([261.63, 329.63, 392.00], dur_s=dur_s) + 0.02 * rng.standard_normal(
        int(dur_s * SR)
    ).astype(np.float32)

    cutoff_sample = int(4.0 * SR)  # exact 4.0s = step 40 at 10 Hz (indices 0..39)
    y_future_perturbed = y_clean.copy()
    y_future_perturbed[cutoff_sample:] += 5.0 * rng.standard_normal(
        len(y_clean) - cutoff_sample
    ).astype(np.float32)

    out_clean = extract_features(y_clean, sr=SR, tier=3, target_hz=10.0)
    out_pert = extract_features(y_future_perturbed, sr=SR, tier=3, target_hz=10.0)

    n_causal = 40  # t_end <= 4.0s
    np.testing.assert_allclose(
        out_clean.features[:n_causal],
        out_pert.features[:n_causal],
        atol=1e-5,
        err_msg="Future audio perturbation leaked into past causal DSP frames!",
    )
    # Sanity check that future frames actually changed
    assert np.max(np.abs(out_clean.features[n_causal:] - out_pert.features[n_causal:])) > 1.0


def test_pure_sine_pitch_and_voicing() -> None:
    y = _sine(440.0, dur_s=4.0)
    out = extract_features(y, sr=SR, tier=3, target_hz=10.0)
    names = get_tier_names(3)
    f0_midi = out.features[:, names.index("pitch_f0_midi")]
    voicing = out.features[:, names.index("pitch_voicing_prob")]
    diss = out.features[:, names.index("sensory_dissonance_plomp")]

    assert abs(float(np.median(f0_midi[5:])) - 69.0) < 0.5
    assert float(np.median(voicing[5:])) > 0.75
    assert float(np.median(diss[5:])) < 0.05


def test_major_vs_minor_triad_mode_score() -> None:
    y_maj = _chord([261.63, 329.63, 392.00], dur_s=5.0)  # C major (C-E-G)
    y_min = _chord([261.63, 311.13, 392.00], dur_s=5.0)  # C minor (C-Eb-G)
    names = get_tier_names(3)
    idx_mode = names.index("mode_major_minor_score")

    s_maj = float(np.mean(extract_features(y_maj, sr=SR, tier=3).features[20:, idx_mode]))
    s_min = float(np.mean(extract_features(y_min, sr=SR, tier=3).features[20:, idx_mode]))
    assert s_maj > s_min
    assert s_maj > 0.0
    assert s_min < 0.0


def test_plomp_dissonance_cluster_vs_fifth() -> None:
    y_cons = _chord([261.63, 392.00], dur_s=3.0)  # C4 + G4 (perfect fifth)
    y_diss = _chord([261.63, 277.18, 369.99], dur_s=3.0)  # C4 + C#4 + F#4 cluster
    names = get_tier_names(3)
    idx_diss = names.index("sensory_dissonance_plomp")

    d_cons = float(np.mean(extract_features(y_cons, sr=SR, tier=3).features[5:, idx_diss]))
    d_diss = float(np.mean(extract_features(y_diss, sr=SR, tier=3).features[5:, idx_diss]))
    assert d_diss > d_cons


def test_120_bpm_click_tempo_and_pulse_clarity() -> None:
    rng = np.random.default_rng(123)
    y_click = _click_track(bpm=120.0, dur_s=8.0)
    y_noise = (0.2 * rng.standard_normal(int(8.0 * SR))).astype(np.float32)

    names = get_tier_names(3)
    idx_bpm = names.index("tempo_local_bpm")
    idx_pulse = names.index("pulse_clarity")

    f_click = extract_features(y_click, sr=SR, tier=3).features
    f_noise = extract_features(y_noise, sr=SR, tier=3).features

    bpm_late = float(np.median(f_click[40:, idx_bpm]))
    assert abs(bpm_late - 120.0) <= 6.0
    assert float(np.mean(f_click[40:, idx_pulse])) > float(np.mean(f_noise[40:, idx_pulse]))
