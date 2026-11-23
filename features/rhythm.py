"""Branch 2: Rhythm, Onsets & Pulse Entrainment causal extractors (6 channels)."""

import numpy as np

from features.clock import (
    BASE_HZ,
    SharedCausalSpectra,
    causal_rolling_mean,
    causal_rolling_std,
    causal_rolling_sum,
)


def extract_rhythm_features(spectra: SharedCausalSpectra) -> dict[str, np.ndarray]:
    """Extract all 6 Branch-2 Rhythm & Pulse channels at base 50 Hz rate."""
    n_frames = spectra.n_frames

    # 1. Backward log-Mel spectral flux (100% causal, no global max normalization)
    log_mel = np.log1p(10.0 * np.maximum(spectra.mel_spec, 0.0))
    onset_strength_flux = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        pos_diff = np.maximum(0.0, log_mel[:, 1:] - log_mel[:, :-1])
        onset_strength_flux[1:] = pos_diff.mean(axis=0).astype(np.float32)

    # 2. Causal 2.0s local adaptive onset density (events/sec)
    thresh_mean = causal_rolling_mean(onset_strength_flux, win=25)
    thresh_std = causal_rolling_std(onset_strength_flux, win=25)
    is_peak = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 2:
        local_max = (onset_strength_flux[1:-1] >= onset_strength_flux[:-2]) & (
            onset_strength_flux[1:-1] > onset_strength_flux[2:]
        )
        above_thresh = onset_strength_flux[1:-1] > np.maximum(
            0.05, thresh_mean[1:-1] + 0.5 * thresh_std[1:-1]
        )
        # Note: peak at t-1 is verified only once frame t arrives; shift by +1 frame for strict causality
        is_peak[2:] = (local_max & above_thresh).astype(np.float32)

    density_win = round(2.0 * BASE_HZ)  # 100 frames
    counts = np.minimum(np.arange(1, n_frames + 1, dtype=np.float32), float(density_win))
    onset_density_local = (causal_rolling_sum(is_peak, density_win) * (BASE_HZ / counts)).astype(
        np.float32
    )

    # 3-5. Causal 4.0s sliding autocorrelation & modulation spectrum (40-200 BPM)
    tempogram_win = round(4.0 * BASE_HZ)  # 200 frames
    min_hist = round(1.0 * BASE_HZ)  # wait at least 1.0s before estimating periodicity
    lags = np.arange(15, 76, dtype=np.int32)  # 60*50/75=40 BPM .. 60*50/15=200 BPM
    bpms = (60.0 * BASE_HZ) / lags.astype(np.float64)
    # Gentle perceptual prior centered at 120 BPM prevents octave-halving (60 vs 120 BPM)
    tempo_prior = np.exp(-0.5 * ((np.log2(bpms / 120.0)) / 0.85) ** 2)

    tempo_local_bpm = np.full(n_frames, 120.0, dtype=np.float32)
    pulse_clarity = np.zeros(n_frames, dtype=np.float32)
    rhythmic_syncopation_proxy = np.zeros(n_frames, dtype=np.float32)

    flux_64 = onset_strength_flux.astype(np.float64)
    fft_freqs = np.fft.rfftfreq(tempogram_win, d=1.0 / BASE_HZ)
    metrical_band = (fft_freqs >= 0.5) & (fft_freqs <= 8.0)

    for idx in range(min_hist - 1, n_frames):
        start = max(0, idx - tempogram_win + 1)
        seg = flux_64[start : idx + 1]
        seg_c = seg - seg.mean()
        energy0 = float(np.dot(seg_c, seg_c))
        if energy0 <= 1e-10:
            continue

        acf_vals = np.empty(len(lags), dtype=np.float64)
        seg_len = len(seg_c)
        for li, lag in enumerate(lags):
            if lag >= seg_len:
                acf_vals[li] = 0.0
            else:
                acf_vals[li] = float(np.dot(seg_c[lag:], seg_c[:-lag])) / energy0

        acf_pos = np.clip(acf_vals, 0.0, 1.0)
        pulse_clarity[idx] = np.float32(np.max(acf_pos))

        weighted_acf = acf_pos * tempo_prior
        best_idx = int(np.argmax(weighted_acf))
        if weighted_acf[best_idx] > 1e-6:
            tempo_local_bpm[idx] = np.float32(bpms[best_idx])

        # Normalized spectral entropy of causal onset modulation spectrum
        if seg_len < tempogram_win:
            seg_fft = np.pad(seg_c, (tempogram_win - seg_len, 0), mode="constant")
        else:
            seg_fft = seg_c
        mod_power = np.abs(np.fft.rfft(seg_fft)) ** 2
        band_power = mod_power[metrical_band]
        bp_sum = float(band_power.sum())
        if bp_sum > 1e-12:
            p = band_power / bp_sum
            ent = -float(np.sum(p * np.log(p + 1e-12)))
            rhythmic_syncopation_proxy[idx] = np.float32(ent / np.log(len(band_power)))

    # 6. Sub-bass / kick (20-150 Hz) causal half-wave rectified flux
    low_band = (spectra.freqs_2048 >= 20.0) & (spectra.freqs_2048 <= 150.0)
    low_log = np.log1p(10.0 * spectra.mag_2048[low_band, :])
    low_band_pulse_energy = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        low_diff = np.maximum(0.0, low_log[:, 1:] - low_log[:, :-1])
        low_band_pulse_energy[1:] = low_diff.mean(axis=0).astype(np.float32)

    return {
        "onset_strength_flux": onset_strength_flux,
        "onset_density_local": onset_density_local,
        "tempo_local_bpm": tempo_local_bpm,
        "pulse_clarity": pulse_clarity,
        "rhythmic_syncopation_proxy": rhythmic_syncopation_proxy,
        "low_band_pulse_energy": low_band_pulse_energy,
    }
