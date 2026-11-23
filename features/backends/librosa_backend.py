"""Default cross-platform librosa + scipy + numpy causal feature backend."""

from features.clock import SR, SharedCausalSpectra
from features.harmony import extract_harmony_features
from features.pitch import extract_melody_features
from features.rhythm import extract_rhythm_features
from features.timbre import extract_timbre_features


def extract_all_channels_librosa(y, sr: int = SR, tier: int = 3) -> dict[str, object]:
    """Compute requested cognitive feature channels at base 50 Hz clock.

    Computes only the cognitive branches required by `tier` (1, 2, or 3)
    while sharing causal spectra across branches.
    """
    spectra = SharedCausalSpectra.from_audio(y, sr=sr)
    channels: dict[str, object] = {}
    channels.update(extract_melody_features(spectra))
    channels.update(extract_rhythm_features(spectra))
    channels.update(extract_timbre_features(spectra))
    if tier >= 3:
        channels.update(extract_harmony_features(spectra))
    return channels
