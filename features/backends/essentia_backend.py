"""Optional Linux/Docker Essentia backend adapter (falls back cleanly when absent)."""

from features.backends.librosa_backend import extract_all_channels_librosa
from features.clock import SR


def is_essentia_available() -> bool:
    """Return True if `essentia.standard` is importable in the current environment."""
    try:
        import essentia.standard  # noqa: F401

        return True
    except ImportError:
        return False


def extract_all_channels_essentia(y, sr: int = SR, tier: int = 3) -> dict[str, object]:
    """Extract cognitive channels using Essentia Plomp dissonance where available.

    Aligns strictly to the unified 50 Hz left-padded frame clock (`center=False`).
    Raises RuntimeError if Essentia is not installed (use `librosa` backend on Windows).
    """
    if not is_essentia_available():
        raise RuntimeError(
            "Essentia is not installed in this environment. Use backend='librosa' on Windows, "
            "or build the Docker image with --build-arg INSTALL_ESSENTIA=true."
        )

    import essentia.standard as es
    import numpy as np

    channels = extract_all_channels_librosa(y, sr=sr, tier=tier)
    if tier >= 3:
        from features.clock import HOP_LENGTH, N_FFT_LONG, left_pad_audio

        y_pad, n_frames = left_pad_audio(y, N_FFT_LONG, HOP_LENGTH)
        if sr != SR:
            import librosa

            y_pad = librosa.resample(y_pad, orig_sr=sr, target_sr=SR)

        win = es.Windowing(type="hann")
        spec = es.Spectrum(size=N_FFT_LONG)
        peaks = es.SpectralPeaks(
            orderBy="frequency",
            magnitudeThreshold=0.0001,
            maxPeaks=12,
            minFrequency=40.0,
            maxFrequency=5000.0,
            sampleRate=float(SR),
        )
        dissonance = es.Dissonance()
        diss_vals = np.zeros(n_frames, dtype=np.float32)
        for idx in range(n_frames):
            start = idx * HOP_LENGTH
            frame = y_pad[start : start + N_FFT_LONG].astype(np.float32)
            mag = spec(win(frame))
            freqs, mags = peaks(mag)
            if len(freqs) >= 2:
                diss_vals[idx] = np.float32(dissonance(freqs, mags))
        channels["sensory_dissonance_plomp"] = diss_vals

    return channels
