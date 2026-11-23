"""Typed registry of the 23-channel cognitive musicality feature schema."""

from dataclasses import dataclass

BRANCH_NAMES: tuple[str, ...] = ("melody", "rhythm", "timbre", "harmony")


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for a single frame-level cognitive feature."""

    name: str
    branch: str
    min_tier: int
    unit: str
    causal_horizon_s: float
    description: str


FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    # Branch 1: Melody & Pitch Contour (6 channels)
    FeatureSpec(
        name="pitch_f0_midi",
        branch="melody",
        min_tier=1,
        unit="MIDI",
        causal_horizon_s=0.093,
        description="Salient melodic pitch height in MIDI (causally forward-filled across unvoiced gaps)",
    ),
    FeatureSpec(
        name="pitch_voicing_prob",
        branch="melody",
        min_tier=1,
        unit="prob [0,1]",
        causal_horizon_s=0.093,
        description="Causal periodicity/voicing confidence preventing 0 Hz unvoiced artifacts",
    ),
    FeatureSpec(
        name="pitch_contour_slope",
        branch="melody",
        min_tier=3,
        unit="semitones/s",
        causal_horizon_s=0.300,
        description="Trailing 0.30s OLS slope of voiced melodic pitch contour",
    ),
    FeatureSpec(
        name="pitch_interval_abs",
        branch="melody",
        min_tier=3,
        unit="semitones",
        causal_horizon_s=0.020,
        description="Instantaneous absolute melodic interval step/leap |Δf0|",
    ),
    FeatureSpec(
        name="pitch_register_centroid",
        branch="melody",
        min_tier=3,
        unit="MIDI",
        causal_horizon_s=0.186,
        description="Polyphony-robust log-frequency spectral centroid on MIDI scale",
    ),
    FeatureSpec(
        name="pitch_entropy_local",
        branch="melody",
        min_tier=3,
        unit="nats",
        causal_horizon_s=2.000,
        description="Trailing 2.0s Shannon entropy of 12-bin pitch-class distribution",
    ),
    # Branch 2: Rhythm, Onsets & Pulse Entrainment (6 channels)
    FeatureSpec(
        name="onset_strength_flux",
        branch="rhythm",
        min_tier=1,
        unit="log-mel flux",
        causal_horizon_s=0.093,
        description="Half-wave rectified backward log-Mel spectral flux onset contour",
    ),
    FeatureSpec(
        name="onset_density_local",
        branch="rhythm",
        min_tier=3,
        unit="events/s",
        causal_horizon_s=2.000,
        description="Trailing 2.0s adaptive onset event density",
    ),
    FeatureSpec(
        name="tempo_local_bpm",
        branch="rhythm",
        min_tier=3,
        unit="BPM",
        causal_horizon_s=4.000,
        description="Causal 4.0s autocorrelation tempogram peak tempo (40-200 BPM prior)",
    ),
    FeatureSpec(
        name="pulse_clarity",
        branch="rhythm",
        min_tier=3,
        unit="ratio [0,1]",
        causal_horizon_s=4.000,
        description="Normalized beat periodicity strength from causal 4.0s tempogram",
    ),
    FeatureSpec(
        name="rhythmic_syncopation_proxy",
        branch="rhythm",
        min_tier=3,
        unit="norm entropy [0,1]",
        causal_horizon_s=4.000,
        description="Normalized spectral entropy of causal 4.0s onset modulation spectrum",
    ),
    FeatureSpec(
        name="low_band_pulse_energy",
        branch="rhythm",
        min_tier=3,
        unit="flux",
        causal_horizon_s=0.093,
        description="Sub-bass (20-150 Hz) half-wave rectified spectral flux",
    ),
    # Branch 3: Timbre, Brightness & Loudness Dynamics (5 channels)
    FeatureSpec(
        name="loudness_perceptual_db",
        branch="timbre",
        min_tier=2,
        unit="dB(A)",
        causal_horizon_s=0.093,
        description="A-weighted causal STFT frame loudness in dB",
    ),
    FeatureSpec(
        name="loudness_delta",
        branch="timbre",
        min_tier=2,
        unit="dB/s",
        causal_horizon_s=0.500,
        description="Trailing 0.50s loudness swell/decay slope (crescendo/decrescendo)",
    ),
    FeatureSpec(
        name="spectral_centroid_bark",
        branch="timbre",
        min_tier=1,
        unit="Bark",
        causal_horizon_s=0.093,
        description="Timbral brightness (spectral centroid mapped to psychoacoustic Bark scale)",
    ),
    FeatureSpec(
        name="spectral_rolloff_bark",
        branch="timbre",
        min_tier=3,
        unit="Bark",
        causal_horizon_s=0.093,
        description="High-frequency 85% spectral rolloff mapped to Bark scale",
    ),
    FeatureSpec(
        name="spectral_flatness",
        branch="timbre",
        min_tier=3,
        unit="ratio [0,1]",
        causal_horizon_s=0.093,
        description="Wiener entropy (tonal purity vs. noisy/distorted texture)",
    ),
    # Branch 4: Harmony, Tonality & Consonance (6 channels)
    FeatureSpec(
        name="sensory_dissonance_plomp",
        branch="harmony",
        min_tier=3,
        unit="norm roughness",
        causal_horizon_s=0.186,
        description="Plomp-Levelt / Sethares pairwise critical-band spectral peak roughness",
    ),
    FeatureSpec(
        name="harmonic_to_percussive_ratio",
        branch="harmony",
        min_tier=3,
        unit="log ratio",
        causal_horizon_s=0.372,
        description="Causal STFT HPSS log harmonic-to-percussive energy ratio",
    ),
    FeatureSpec(
        name="tonal_centroid_flux",
        branch="harmony",
        min_tier=3,
        unit="L2/frame",
        causal_horizon_s=0.186,
        description="Frame-to-frame L2 velocity in 6D Tonnetz harmonic space",
    ),
    FeatureSpec(
        name="mode_major_minor_score",
        branch="harmony",
        min_tier=3,
        unit="corr diff",
        causal_horizon_s=3.000,
        description="Trailing 3.0s Krumhansl-Schmuckler max(Major) - max(Minor) correlation",
    ),
    FeatureSpec(
        name="key_clarity",
        branch="harmony",
        min_tier=3,
        unit="corr [-1,1]",
        causal_horizon_s=3.000,
        description="Trailing 3.0s peak correlation across all 24 major/minor key templates",
    ),
    FeatureSpec(
        name="harmonic_expectancy_surprise",
        branch="harmony",
        min_tier=3,
        unit="cosine dist",
        causal_horizon_s=3.000,
        description="Cosine distance between current chroma and causal EMA tonal memory (τ=3.0s)",
    ),
)

FEATURE_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SCHEMA}
