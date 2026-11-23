"""Unified 50 Hz base clock, left-only causal padding, and shared spectra utilities."""

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import lfilter, lfilter_zi

SR: int = 22050
HOP_LENGTH: int = 441
BASE_HZ: float = SR / HOP_LENGTH  # Exact 50.0 Hz (20.0 ms per frame)
N_FFT_SHORT: int = 2048
N_FFT_LONG: int = 4096


def left_pad_audio(
    y: np.ndarray,
    win_length: int,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, int]:
    """Trim to whole hop frames and prepend `win_length - hop_length` zeros on the left.

    With `center=False`, frame `k` spans samples ending strictly at `(k + 1) * hop_length`.
    """
    y_1d = np.asarray(y, dtype=np.float32).reshape(-1)
    n_frames = len(y_1d) // hop_length
    if n_frames <= 0:
        raise ValueError(f"Audio length {len(y_1d)} is shorter than 1 frame ({hop_length} samples).")
    y_trim = y_1d[: n_frames * hop_length]
    pad_left = max(0, win_length - hop_length)
    y_pad = np.pad(y_trim, (pad_left, 0), mode="constant")
    return y_pad, n_frames


def compute_causal_stft_mag(
    y: np.ndarray,
    n_fft: int,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute causal magnitude STFT `[n_fft // 2 + 1, T]` with left-only padding."""
    y_pad, n_frames = left_pad_audio(y, n_fft, hop_length)
    stft = librosa.stft(
        y_pad,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=False,
    )
    if stft.shape[1] != n_frames:
        stft = stft[:, :n_frames]
    return np.abs(stft).astype(np.float32)


def causal_hpss(
    mag_spec: np.ndarray,
    freq_kernel: int = 17,
    time_kernel: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal Harmonic-Percussive Source Separation on a magnitude spectrogram.

    Harmonic median operates strictly along trailing `[t - time_kernel + 1 .. t]` frames.
    Percussive median operates instantaneously along frequency within each frame.
    """
    mag = np.asarray(mag_spec, dtype=np.float32)
    perc_est = median_filter(mag, size=(freq_kernel, 1), mode="nearest")

    padded = np.pad(mag, ((0, 0), (time_kernel - 1, 0)), mode="edge")
    windows = sliding_window_view(padded, window_shape=time_kernel, axis=1)
    harm_est = np.median(windows, axis=-1).astype(np.float32)

    denom = harm_est ** 2 + perc_est ** 2 + 1e-10
    harm_mag = mag * ((harm_est ** 2) / denom)
    perc_mag = mag * ((perc_est ** 2) / denom)
    return harm_mag.astype(np.float32), perc_mag.astype(np.float32)


def causal_rolling_sum(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing causal sum over up to `win` frames along the last axis."""
    arr = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return arr.astype(np.float32)
    cs = np.cumsum(arr, axis=-1)
    out = cs.copy()
    if arr.shape[-1] > win:
        out[..., win:] = cs[..., win:] - cs[..., :-win]
    return out.astype(np.float32)


def causal_rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing causal mean with expanding startup window `[0..t]` for `t < win`."""
    arr = np.asarray(x, dtype=np.float64)
    n_t = arr.shape[-1]
    counts = np.minimum(np.arange(1, n_t + 1, dtype=np.float64), float(win))
    return (causal_rolling_sum(arr, win).astype(np.float64) / counts).astype(np.float32)


def causal_rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing causal standard deviation along the last axis."""
    mean_x = causal_rolling_mean(x, win).astype(np.float64)
    mean_x2 = causal_rolling_mean(np.square(x, dtype=np.float64), win).astype(np.float64)
    var = np.maximum(0.0, mean_x2 - mean_x ** 2)
    return np.sqrt(var).astype(np.float32)


def causal_rolling_slope(x: np.ndarray, win: int, dt: float = 1.0 / BASE_HZ) -> np.ndarray:
    """Trailing causal OLS linear slope (in signal units per second) along 1D array."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n_t = len(arr)
    out = np.zeros(n_t, dtype=np.float64)
    if n_t < 2 or win < 2:
        return out.astype(np.float32)

    for idx in range(1, min(win - 1, n_t)):
        seg = arr[: idx + 1]
        k = np.arange(idx + 1, dtype=np.float64)
        k_c = k - k.mean()
        out[idx] = np.dot(k_c, seg - seg.mean()) / (np.dot(k_c, k_c) * dt)

    if n_t >= win:
        k = np.arange(win, dtype=np.float64)
        k_c = k - k.mean()
        weights = k_c / (np.dot(k_c, k_c) * dt)
        slopes = np.convolve(arr, weights[::-1], mode="valid")
        out[win - 1 :] = slopes

    return out.astype(np.float32)


def causal_ema(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """Causal 1st-order exponential moving average initialized at frame 0."""
    arr = np.asarray(x, dtype=np.float64)
    alpha = 1.0 - float(np.exp(-1.0 / max(tau_frames, 1.0)))
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi_base = lfilter_zi(b, a)

    if arr.ndim == 1:
        out, _ = lfilter(b, a, arr, zi=zi_base * arr[0])
        return out.astype(np.float32)

    out = np.empty_like(arr, dtype=np.float64)
    for row in range(arr.shape[0]):
        out[row], _ = lfilter(b, a, arr[row], zi=zi_base * arr[row, 0])
    return out.astype(np.float32)


def causal_downsample(
    feats: np.ndarray,
    base_hz: float = BASE_HZ,
    target_hz: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample `[T, C]` base features causally via non-overlapping block averages."""
    arr = np.asarray(feats, dtype=np.float32)
    n_t, n_c = arr.shape
    if np.isclose(target_hz, base_hz):
        times_s = (np.arange(1, n_t + 1, dtype=np.float32) / np.float32(base_hz))
        return arr, times_s

    ratio = base_hz / target_hz
    stride = round(ratio)
    if not np.isclose(ratio, stride) or stride < 1:
        raise ValueError(f"target_hz={target_hz} must evenly divide base_hz={base_hz}.")

    n_blocks = n_t // stride
    if n_blocks <= 0:
        raise ValueError(f"Sequence length {n_t} too short to downsample by stride={stride}.")
    trimmed = arr[: n_blocks * stride].reshape(n_blocks, stride, n_c)
    down = trimmed.mean(axis=1).astype(np.float32)
    times_s = (np.arange(1, n_blocks + 1, dtype=np.float32) * np.float32(stride / base_hz))
    return down, times_s


def hz_to_bark(hz: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to Zwicker psychoacoustic Bark scale."""
    f = np.maximum(np.asarray(hz, dtype=np.float64), 0.0)
    bark = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
    return bark.astype(np.float32)


@dataclass
class SharedCausalSpectra:
    """Precomputed causal spectrograms shared across all 4 cognitive feature branches."""

    y: np.ndarray
    sr: int
    hop_length: int
    n_frames: int
    mag_2048: np.ndarray
    freqs_2048: np.ndarray
    mag_4096: np.ndarray
    freqs_4096: np.ndarray
    harm_mag_2048: np.ndarray
    perc_mag_2048: np.ndarray
    mel_spec: np.ndarray
    chroma_12: np.ndarray

    @classmethod
    def from_audio(
        cls,
        y: np.ndarray,
        sr: int = SR,
        hop_length: int = HOP_LENGTH,
    ) -> "SharedCausalSpectra":
        """Compute all shared causal spectra once for an audio signal."""
        if sr != SR:
            y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=SR)
            sr = SR
        else:
            y = np.asarray(y, dtype=np.float32).reshape(-1)

        mag_2048 = compute_causal_stft_mag(y, N_FFT_SHORT, hop_length)
        n_frames = mag_2048.shape[1]
        mag_4096 = compute_causal_stft_mag(y, N_FFT_LONG, hop_length)[:, :n_frames]
        freqs_2048 = librosa.fft_frequencies(sr=sr, n_fft=N_FFT_SHORT).astype(np.float32)
        freqs_4096 = librosa.fft_frequencies(sr=sr, n_fft=N_FFT_LONG).astype(np.float32)

        harm_mag_2048, perc_mag_2048 = causal_hpss(mag_2048)

        mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT_SHORT, n_mels=64, fmin=27.5, fmax=8000.0)
        mel_spec = (mel_fb @ (mag_2048 ** 2)).astype(np.float32)

        chroma_fb = librosa.filters.chroma(
            sr=sr, n_fft=N_FFT_LONG, n_chroma=12, tuning=0.0, octwidth=None
        )
        chroma_raw = chroma_fb @ (mag_4096 ** 2)
        chroma_sum = chroma_raw.sum(axis=0, keepdims=True) + 1e-8
        chroma_12 = (chroma_raw / chroma_sum).astype(np.float32)

        return cls(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_frames=n_frames,
            mag_2048=mag_2048,
            freqs_2048=freqs_2048,
            mag_4096=mag_4096,
            freqs_4096=freqs_4096,
            harm_mag_2048=harm_mag_2048,
            perc_mag_2048=perc_mag_2048,
            mel_spec=mel_spec,
            chroma_12=chroma_12,
        )
