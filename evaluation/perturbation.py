"""Experiment 2: Causal acoustic perturbation suite (Grimaud & Eerola, 2022 test)."""

import librosa
import numpy as np


def perturb_pitch(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by `n_steps` semitones while preserving tempo and duration."""
    return librosa.effects.pitch_shift(
        np.asarray(y, dtype=np.float32), sr=sr, n_steps=n_steps
    ).astype(np.float32)


def perturb_tempo(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch audio by `rate` (>1.0 faster, <1.0 slower) while preserving pitch."""
    return librosa.effects.time_stretch(
        np.asarray(y, dtype=np.float32), rate=rate
    ).astype(np.float32)


def perturb_brightness_tilt(y: np.ndarray, sr: int, tilt_db_per_octave: float) -> np.ndarray:
    """Apply smooth frequency-domain spectral tilt to increase or decrease brightness."""
    arr = np.asarray(y, dtype=np.float32)
    spec = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / sr)
    octaves = np.log2(np.maximum(freqs, 100.0) / 1000.0)
    gain = 10.0 ** ((tilt_db_per_octave * octaves) / 20.0)
    return np.fft.irfft(spec * gain, n=len(arr)).astype(np.float32)


def perturb_loudness(y: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale perceptual waveform loudness by `gain_db`."""
    scale = 10.0 ** (gain_db / 20.0)
    return (np.asarray(y, dtype=np.float32) * scale).astype(np.float32)
