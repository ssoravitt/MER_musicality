"""Branch 1: Melody & Pitch Contour causal cognitive extractors (6 channels)."""

import librosa
import numpy as np

from features.clock import (
    BASE_HZ,
    N_FFT_SHORT,
    SharedCausalSpectra,
    causal_rolling_mean,
    causal_rolling_slope,
    left_pad_audio,
)


def _causal_voicing_prob(
    y_pad: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Compute strictly causal periodicity confidence `[0, 1]` from frame autocorrelation."""
    frames = librosa.util.frame(y_pad, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-10)
    energy_gate = np.clip((rms_db + 60.0) / 15.0, 0.0, 1.0)

    centered = frames - frames.mean(axis=0, keepdims=True)
    n_fft_acf = 2 * frame_length
    spec = np.fft.rfft(centered, n=n_fft_acf, axis=0)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft_acf, axis=0)
    acf_0 = acf[0] + 1e-12

    lag_min = max(1, int(np.floor(sr / fmax)))
    lag_max = min(frame_length - 1, int(np.ceil(sr / fmin)))
    peak_acf = np.max(acf[lag_min : lag_max + 1], axis=0) / acf_0
    voicing = np.clip(peak_acf, 0.0, 1.0) * energy_gate
    return voicing.astype(np.float32)


def extract_melody_features(spectra: SharedCausalSpectra) -> dict[str, np.ndarray]:
    """Extract all 6 Branch-1 Melody & Pitch channels at base 50 Hz rate."""
    fmin, fmax = 65.41, 2093.0  # C2 to C7
    y_pad, n_frames = left_pad_audio(spectra.y, N_FFT_SHORT, spectra.hop_length)

    f0_hz = librosa.yin(
        y_pad,
        fmin=fmin,
        fmax=fmax,
        sr=spectra.sr,
        frame_length=N_FFT_SHORT,
        hop_length=spectra.hop_length,
        center=False,
    )[:n_frames]

    voicing_prob = _causal_voicing_prob(
        y_pad, spectra.sr, N_FFT_SHORT, spectra.hop_length, fmin, fmax
    )[:n_frames]

    voiced_mask = (voicing_prob >= 0.45) & np.isfinite(f0_hz) & (f0_hz > 0.0)
    midi_raw = 69.0 + 12.0 * np.log2(np.clip(f0_hz, fmin, fmax) / 440.0)

    pitch_f0_midi = np.empty(n_frames, dtype=np.float32)
    last_midi = np.float32(60.0)  # Default C4 before first voiced frame
    for idx in range(n_frames):
        if voiced_mask[idx]:
            last_midi = np.float32(midi_raw[idx])
        pitch_f0_midi[idx] = last_midi

    # Zero derivative slope / interval jump when unvoiced so steady silence has 0 motion
    raw_slope = causal_rolling_slope(pitch_f0_midi, win=round(0.30 * BASE_HZ), dt=1.0 / BASE_HZ)
    pitch_contour_slope = np.where(voiced_mask, raw_slope, 0.0).astype(np.float32)

    pitch_interval_abs = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        diffs = np.abs(np.diff(pitch_f0_midi))
        both_voiced = voiced_mask[1:] & voiced_mask[:-1]
        pitch_interval_abs[1:] = np.where(both_voiced, diffs, 0.0)

    # Polyphony-robust CQT-like log-frequency centroid on causal STFT_4096 (65.41..4186 Hz)
    freqs = spectra.freqs_4096
    band = (freqs >= 65.41) & (freqs <= 4186.0)
    band_freqs = freqs[band]
    band_midi = (69.0 + 12.0 * np.log2(band_freqs / 440.0))[:, None]
    band_mag = spectra.mag_4096[band, :]
    mag_sum = band_mag.sum(axis=0)
    pitch_register_centroid = np.where(
        mag_sum > 1e-8,
        (band_mag * band_midi).sum(axis=0) / (mag_sum + 1e-10),
        60.0,
    ).astype(np.float32)

    # Causal 2.0s pitch-class Shannon entropy
    chroma_avg = causal_rolling_mean(spectra.chroma_12, win=round(2.0 * BASE_HZ))
    chroma_p = chroma_avg / (chroma_avg.sum(axis=0, keepdims=True) + 1e-10)
    pitch_entropy_local = (-np.sum(chroma_p * np.log(chroma_p + 1e-10), axis=0)).astype(np.float32)

    return {
        "pitch_f0_midi": pitch_f0_midi,
        "pitch_voicing_prob": voicing_prob,
        "pitch_contour_slope": pitch_contour_slope,
        "pitch_interval_abs": pitch_interval_abs,
        "pitch_register_centroid": pitch_register_centroid,
        "pitch_entropy_local": pitch_entropy_local,
    }
