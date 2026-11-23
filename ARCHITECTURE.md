# Cognition-First, Time-Series Music Emotion Recognition (`MER_musicality`)

> **Status:** Phase 0 v2 — MUSEC Past Outcomes Synthesized & Architecture Realigned (No implementation code yet)  
> **Prior Work Anchors:** `MUSEC_Review MER` & `MUSEC progress` (Apr–Jun 2024)  
> **Target Environment:** Native Windows / Python (with optional Linux/WSL2 `essentia` backend)

---

## 1. MUSEC Research Foundation & Continuity Summary

### 1.1 Core Scientific Thesis (`MUSEC_Review MER`)
Traditional Music Emotion Recognition (MER) models routinely achieve high benchmark scores (~75–80% accuracy/F1; e.g., Panda et al., 2020; Louro et al., 2024) by extracting 100–1,714 track-level summary statistics (`mean`, `std`, `skew`, percentiles) and fitting black-box classifiers or spectrograms. While effective on static benchmarks, this engineering workflow raises the core question formulated in `MUSEC_Review MER`:

> *"Does music emotion recognition indeed contribute to understanding music emotion mechanisms in cognitive science/psychology studies?"*

Grounded in **Aucouturier & Bigand (2013)** ("Seven problems that keep MIR from attracting the interest of cognition and neuroscience") and large-scale cross-cultural music cognition studies (**Savage et al., 2015; Mehr et al., 2019; Ozaki et al., 2022; Honing, 2023; Singh & Mehr, 2023; Putkinen et al., 2024**), this project bridges cognitive science and affective computing through four principles:
1. **Cross-Cultural Musical Universals First:** Prioritize core perceptual elements shown in human behavioral studies to carry musical recognition and emotion—**Melody**, **Rhythm**, and **Timbre** (extended with **Loudness** and **Harmony/Consonance**).
2. **Contour Perception & Relative Representations:** Ground frame-by-frame time-series inputs in **Contour Perception Theory** (**Wilson, 1982; Dowling, 1994; Dyson & Watkins, 1984; Galvin et al., 2009; Graves et al., 2019; Schmuckler & Moranis, 2023**) and **McDermott, Lehr, & Oxenham (2008)**, which argue that the auditory system tracks relative contours of **pitch**, **timbre (brightness)**, and **loudness** online rather than waiting for track-level global averages.
3. **Human Behavioral Causality (Production & Perception):** Following **Grimaud & Eerola (2022)**, evaluate not only predictive benchmark accuracy (**Experiment 1**) but also whether perturbing musical cues causes emotion predictions to shift in the same direction as human listeners (**Experiment 2**).
4. **Element Separation over Instrument Separation:** Extending **de Berardinis et al. (2020) (`EmoMucs`)**, separate modeling streams by **biological musical elements** (`Melody`, `Rhythm`, `Timbre/Loudness`, `Harmony`) rather than Demucs instrument stems (`Vocals`, `Bass`, `Drums`, `Other`), since human listeners experience musical emotion without needing explicit instrument identification.

---

### 1.2 June 2024 Baseline & 6-Point Root-Cause Diagnosis (`MUSEC progress`)
Between April and June 2024, the initial MUSEC prototype implemented a 3-contour feature pipeline (Essentia Melody + Essentia Rhythm + Librosa Spectral Centroid) on the **Coimbra 4-Quadrant (4Q)** dataset (Panda et al., 2020) and evaluated an adapted **Louro et al. (2024) Fig. 4 DNN** (BatchNorm removed, `T = 1100`, 500 epochs, `batch_size = 64`, `lr = 0.001`, 10-fold CV).

- **June 14, 2024 Baseline Result:** `Accuracy = 39.84%`, `Macro F1 = 36.07%` (vs. Panda et al. 2020 SVM baselines: `67.5% F1` with 70 baseline features, `76.4% F1` with 100 baseline+novel features).

Inspection of the June 2024 logs, loss curves, tensor shapes, and outlier tables reveals **6 specific technical root causes**—none of which refute the cognitive hypothesis:

| # | Observed Symptom (June 2024 Logs) | Root Cause Diagnosis | Exact Fix in `MER_musicality` v2 |
|---|---|---|---|
| 1 | Output shape `[64, 3, 4]` squeezed to `[64, 1, 4]` from `[64, 3, 1100]` input; unclear how PyTorch computed CE loss | **Flat MLP Channel-Isolation Bug:** Louro et al. Fig. 4 is a `Dense` MLP for tabular vectors `[B, F]`. Feeding `[64, 3, 1100]` applied identical `Dense(1100→...)` weights independently to each of the 3 channels with zero cross-channel interaction and zero shift invariance along `t`. | Replace the flat MLP with a **Multi-Branch Temporal Network (`CogMucs`)** using Causal TCN / GRU encoders per cognitive branch + temporal attention pooling / sequence readout. |
| 2 | Initial train/val loss **> 150** at epoch 0 (expected `ln(4) ≈ 1.39` for 4-class CE) | **Unnormalized Physical Units + BatchNorm Removed:** Raw `f0` (`0–1500 Hz`), `spectral_centroid` (`500–6000 Hz`), and unscaled onset strengths produced huge dot products and unstable gradients once BatchNorm was removed. | Log/MIDI-scale pitch, Bark-scale spectral centroid, dB-scale loudness, and fit `RobustScaler`/`StandardScaler` strictly on the **training split** per fold. |
| 3 | Rhythm `.npy` files unexpectedly huge; mismatched lengths across melody, rhythm, and timbre | **Unaligned Time Clocks Across Toolboxes:** Essentia melody/rhythm and Librosa timbre used incompatible sample rates/hop sizes, breaking temporal alignment when truncated/padded to `1,100` steps. | Enforce a single **Unified `50.0 Hz` Base Clock** (`sr = 22050 Hz`, `hop_length = 441` = `20.0 ms` exact; causal `center=False` + left-only padding) downsampled cleanly to `10.0 Hz` or `2.0 Hz`. |
| 4 | Sparse/zero-heavy pitch curves | **Unvoiced `0 Hz` Artifact:** Monophonic pitch trackers emit `0 Hz` on unvoiced/percussive frames, which linear networks read as extreme low-frequency drops rather than "no pitch present." | Convert voiced `f0` to continuous MIDI (`pitch_f0_midi`), hold/mask unvoiced frames, and provide explicit `pitch_voicing_prob ∈ [0, 1]` as a paired channel. |
| 5 | `~40%` 4Q accuracy (barely above 25% chance, weak valence separation) | **Missing Loudness & Harmony/Consonance Cues:** Pitch/onset/centroid alone track Arousal partially, but separating **Q1 vs. Q2** (high arousal, positive vs. negative valence) and **Q3 vs. Q4** requires loudness dynamics (McDermott et al., 2008) and consonance/mode/tonal stability. | Implement a **Controlled 3-Tier Feature Ladder** starting from Tier 1 (pure 3-contour redemption) → Tier 2 (+ Loudness) → Tier 3 (+ Harmony, Consonance, Rhythm Pulse). |
| 6 | Random re-splitting per fold + dropping clips (`MT0013621344`, `MT0013914319` @ 435; `MT0005171805`, `MT0011051663`) | **Unstable CV & Unmasked Truncation:** Re-randomizing train/val/test each fold mixes evaluation variance, and dropping valid short/long clips discards data unnecessarily. | Use deterministic **stratified / artist-disjoint CSV splits** in `data/splits/` and **boolean sequence masks** inside temporal pooling and loss computation. |

---

## 2. Clean Modular Repository Structure

```text
MER_musicality/
├── ARCHITECTURE.md               # System blueprint, diagnosis, feature ladder & CogMucs spec
├── README.md                     # Quickstart, setup, and experiment reproduction commands
├── pyproject.toml                # Dependencies (librosa, scipy, torch, scikit-learn, pyyaml)
├── .gitignore                    # Excludes raw audio, feature caches, checkpoints, .venv
├── .agent/
│   └── HANDOVER.md               # Cross-machine & cross-session continuity log
├── configs/                      # Declarative YAML configs (tiers, datasets, models, runs)
│   ├── features_tier1.yaml       # 4 ch: Pure MUSEC 3-Contour (+ voicing prob)
│   ├── features_tier2.yaml       # 6 ch: + Loudness contours (McDermott et al., 2008)
│   ├── features_tier3.yaml       # 23 ch: Full 4-domain Cognitive Musicality schema
│   ├── dataset_coimbra4q.yaml    # Stage 1: Panda et al. (2020) Coimbra 4Q benchmark
│   ├── dataset_deam.yaml         # Stage 2: DEAM 2 Hz dynamic V-A benchmark
│   └── models/
│       ├── cogmucs_tcn.yaml
│       └── cogmucs_gru.yaml
├── data/                         # Dataset loaders, deterministic splits & alignment
│   ├── raw/                      # Gitignored external datasets
│   ├── processed/                # Gitignored cached causal feature tensors (.npz)
│   ├── splits/                   # Deterministic stratified & song/artist-disjoint CSVs (tracked)
│   ├── base.py                   # Abstract MERDataset (supports static & dynamic masks)
│   ├── coimbra4q.py              # Stage 1: Coimbra 4Q loader + Panda static feature benchmark
│   ├── deam.py                   # Stage 2: DEAM loader (0-45s warmup, masked t>=15s 2Hz V-A)
│   ├── pmemo.py                  # Stage 2b: PMEmo loader (2Hz V-A + optional EDA signals)
│   ├── emomusic.py               # Stage 3: EmoMusic / MediaEval 2013-2014 loader
│   ├── amg1608.py                # Stage 3: AMG1608 static V-A loader
│   └── alignment.py              # Variable-length padding masks, warmup masks, lag shifts
├── features/                     # Unified-clock causal DSP extractors
│   ├── schema.py                 # Typed FeatureSchema registry (names, domains, units, causal windows)
│   ├── tiers.py                  # Tier 1 (4 ch) / Tier 2 (6 ch) / Tier 3 (23 ch) subset selector
│   ├── clock.py                  # Unified 50 Hz clock (sr=22050, hop=441) & left-only causal padding
│   ├── pitch.py                  # Branch 1: Melody & Pitch Contour (6 ch)
│   ├── rhythm.py                 # Branch 2: Rhythm, Onsets & Pulse Entrainment (6 ch)
│   ├── timbre.py                 # Branch 3: Timbre, Brightness & Loudness (5 ch)
│   ├── harmony.py                # Branch 4: Harmony, Tonality & Consonance (6 ch)
│   ├── pipeline.py               # End-to-end extraction, caching & backend routing
│   └── backends/
│       ├── librosa_backend.py    # Default cross-platform Windows/macOS/Linux backend
│       └── essentia_backend.py   # Optional Linux/WSL2/Docker adapter (Plomp dissonance, pitch/onset)
├── models/                       # CogMucs temporal architectures & SOTA baselines
│   ├── cogmucs.py                # Multi-Branch Cognitive Network (TCN or GRU branches, E/M/L fusion)
│   ├── tcn.py                    # Dilated Causal TCN branch block
│   ├── gru.py                    # Feature-Gated Causal/BiGRU branch block
│   ├── losses.py                 # Masked CrossEntropy (4Q) & Masked Lin's CCC + MSE (2 Hz V-A)
│   └── baselines/
│       ├── panda_tabular.py      # SVM & Louro et al. (2024) Fig. 4 DNN on Panda 70/100/1714 features
│       ├── summary_baseline.py   # Track-level mean/std of Tier 1/2/3 contours (temporal ablation)
│       └── mel_crnn.py           # Choi FCN-4 / Mel-CRNN spectrogram baseline
├── evaluation/                   # Metrics, ablations, perturbations & interpretability
│   ├── metrics.py                # Accuracy, Macro-F1 (4Q); Masked per-song CCC, Pearson r, RMSE (V-A)
│   ├── ablation.py               # Tier 1->2->3 progression, Leave-One-Branch-Out, E/M/L fusion tests
│   ├── perturbation.py           # Experiment 2: Causal pitch/tempo/brightness/loudness interventions
│   ├── interpretability.py       # Branch element gates, feature attention weights & lag cross-corr
│   └── visualization.py          # Aligned audio + contour + gate + V/A prediction plots
├── scripts/                      # Thin CLI entrypoints (extract, train, evaluate, perturb)
├── notebooks/                    # Exploratory DSP verification & single-song case studies
└── tests/                        # Unit tests (causality check, synthetic DSP, masking, shape/NaN guards)
```

---

## 3. Unified Clock & Controlled 3-Tier Cognitive Feature Ladder

### 3.1 Unified Causal Clock & Windows Portability
- **Audio Sample Rate:** `sr = 22050 Hz` (mono).
- **Base Extraction Clock:** **`50.0 Hz`** (`hop_length = 441` samples = `20.0 ms` exact). Because `441 × 25 = 11,025 samples = 0.500 s` exact, every 25 base frames align cleanly to one `2.0 Hz` dynamic emotion label with zero drift.
- **Model Input Clock:** Causal anti-aliased downsampling (or strided `Conv1d(stride=5)`) to **`10.0 Hz`** (`100 ms` steps; `T = 450` per 45 s clip, `5` steps per `0.5 s` label), plus an ultra-fast **`2.0 Hz`** (`500 ms` causal local summary) ablation option.
- **Strict DSP Causality Invariant (`center=False` + Left-Only Padding):**  
  `librosa` defaults to `center=True`, peeking `win_length // 2` into the future. All STFT, CQT, pYIN, onset, tempogram, and rolling meso-scale windows MUST pad on the **left only** and compute with `center=False` so frame `t` depends strictly on audio samples `≤ t`.
- **Windows Compatibility:** `essentia` does not ship official prebuilt PyPI wheels for native Windows (targets Linux, macOS, WSL2, or Docker). All 23 features have a **100% native Windows `librosa + scipy + numpy` default backend**, plus an optional `essentia` adapter (`features/backends/essentia_backend.py`) when running in Linux/WSL2/Docker.

### 3.2 Nested Feature Ladder (Tier 1 → Tier 2 → Tier 3)

| Tier | Total Channels | Included Cognitive Channels | Scientific Purpose |
|---|---|---|---|
| **Tier 1: MUSEC Minimal 3-Contour** | **4 ch** *(3 branches)* | **Melody:** `pitch_f0_midi`, `pitch_voicing_prob`<br>**Rhythm:** `onset_strength_flux`<br>**Timbre:** `spectral_centroid_bark` | Direct redemption test of the Apr–Jun 2024 MUSEC hypothesis with all 6 engineering bugs fixed (`0 Hz` masking, unified clock, scaling, temporal modeling). |
| **Tier 2: Auditory Relative-Contour** | **6 ch** *(3 branches)* | Tier 1 + **Timbre/Dynamics:** `loudness_perceptual_db`, `loudness_delta` | Directly tests **McDermott, Lehr, & Oxenham (2008)** & **Honing (2023)** (*pitch, brightness, and loudness* relative auditory contours). |
| **Tier 3: Full Cognition-First Musicality** | **23 ch** *(4 branches)* | Tier 2 + melodic derivatives/entropy + Plomp-Levelt dissonance, Krumhansl mode/key, ITPRA surprise + rolloff/flatness + tempo, pulse clarity, syncopation, sub-bass | Full psychologically & biologically grounded representation separating **Valence** (mode, consonance, brightness) and **Arousal** (loudness, pulse, onset rate, pitch leaps). |

### 3.3 Complete 23-Channel Non-Overlapping Specification (Tier 3)

#### Branch 1: Melody & Pitch Contour (6 Channels)
*Anchors: Wilson (1982), Dowling (1994), Honing (2023), Savage et al. (2015), Mehr et al. (2019), Ozaki et al. (2022), Huron (2006).*

| # | Channel | Tier | Cognitive Construct | Exact DSP Implementation (`sr=22050, hop=441, center=False`) | Causal Horizon |
|---|---|---|---|---|---|
| 1 | `pitch_f0_midi` | **T1, T2, T3** | Salient melodic pitch height | `librosa.pyin` on HPSS harmonic audio (`frame_length=2048`, `C2–C7`, left-padded), converted to MIDI & forward-filled across unvoiced gaps | `92.9 ms` frame |
| 2 | `pitch_voicing_prob` | **T1, T2, T3** | Melodic / vocal presence | Continuous `pyin` voicing probability `[0, 1]` (prevents `0 Hz` unvoiced collapse) | `92.9 ms` frame |
| 3 | `pitch_contour_slope` | T3 | Melodic direction (rising vs. falling) | Causal OLS linear slope (semitones/s) over voiced `f0` frames in trailing window | `0.30 s` (15 frames) |
| 4 | `pitch_interval_abs` | T3 | Stepwise proximity vs. disjunct leaps | Instantaneous `\|f0(t) - f0(t-1)\|` in semitones (masked on unvoiced transitions) | `20 ms` step |
| 5 | `pitch_register_centroid` | T3 | Polyphony-robust spectral pitch height | CQT log-frequency spectral centroid (`bins_per_octave=36`) on MIDI-like scale | `~185 ms` CQT |
| 6 | `pitch_entropy_local` | T3 | Melodic predictability / complexity | Shannon entropy of causal L1-normalized 12-bin pitch-class histogram | `2.0 s` (100 frames) |

#### Branch 2: Rhythm, Onsets & Pulse Entrainment (6 Channels)
*Anchors: Honing (2023), Savage et al. (2015), Schmuckler & Moranis (2023), Lartillot, Hove.*

| # | Channel | Tier | Cognitive Construct | Exact DSP Implementation (`sr=22050, hop=441, center=False`) | Causal Horizon |
|---|---|---|---|---|---|
| 7 | `onset_strength_flux` | **T1, T2, T3** | Rhythmic attack / onset contour | Half-wave rectified spectral flux (`librosa.onset.onset_strength`, `n_fft=2048`, `center=False`) | `92.9 ms` frame |
| 8 | `onset_density_local` | T3 | Articulation busyness / event rate | Causal rolling adaptive peak-picked onset density | `2.0 s` (100 frames) |
| 9 | `tempo_local_bpm` | T3 | Perceived local tempo (`log2 BPM`) | Causal windowed autocorrelation tempogram peak (`40–200 BPM` Gaussian prior) | `4.0 s` (200 frames) |
| 10 | `pulse_clarity` | T3 | Beat salience / isochronous regularity | Peak prominence / normalized autocorrelation strength of causal `4.0 s` tempogram | `4.0 s` (200 frames) |
| 11 | `rhythmic_syncopation_proxy` | T3 | Metrical tension / groove complexity | Normalized spectral entropy of causal `4.0 s` Fourier tempogram (or off-beat vs. on-beat ratio) | `4.0 s` (200 frames) |
| 12 | `low_band_pulse_energy` | T3 | Sub-bass motor entrainment cue | Half-wave rectified spectral flux restricted to `20–150 Hz` STFT bins | `92.9 ms` frame |

#### Branch 3: Timbre, Brightness & Loudness Dynamics (5 Channels)
*Anchors: McAdams (2019), McDermott, Lehr, & Oxenham (2008), Juslin & Laukka (2003), Wagner & Hoeschele (2022), Honing (2023).*

| # | Channel | Tier | Cognitive Construct | Exact DSP Implementation (`sr=22050, hop=441, center=False`) | Causal Horizon |
|---|---|---|---|---|---|
| 13 | `loudness_perceptual_db` | **T2, T3** | Perceived acoustic intensity | A-weighted (`librosa.perceptual_weighting`) STFT power (`n_fft=2048`) in dB | `92.9 ms` frame |
| 14 | `loudness_delta` | **T2, T3** | Dynamic swell (crescendo/decrescendo) | Causal linear slope of `loudness_perceptual_db` over trailing `0.50 s` | `0.50 s` (25 frames) |
| 15 | `spectral_centroid_bark` | **T1, T2, T3** | Timbral brightness / sharpness | `librosa.feature.spectral_centroid` mapped to Bark scale (`13·atan(0.76f) + 3.5·atan((f/7.5)^2)`) | `92.9 ms` frame |
| 16 | `spectral_rolloff_bark` | T3 | High-frequency acoustic brilliance | 85% `librosa.feature.spectral_rolloff` mapped to Bark scale | `92.9 ms` frame |
| 17 | `spectral_flatness` | T3 | Tonal purity vs. noisy distortion | `librosa.feature.spectral_flatness` (Wiener entropy) | `92.9 ms` frame |

#### Branch 4: Harmony, Tonality & Consonance (6 Channels)
*Anchors: Savage et al. (2015), Krumhansl (1990), Plomp & Levelt (1965), Huron (2006).*

| # | Channel | Tier | Cognitive Construct | Exact DSP Implementation (`sr=22050, hop=441, center=False`) | Causal Horizon |
|---|---|---|---|---|---|
| 18 | `sensory_dissonance_plomp` | T3 | Acoustic roughness / beating | Pairwise Plomp-Levelt/Sethares roughness from spectral peaks (`n_fft=4096`; optional `essentia.standard.Dissonance` on Linux/WSL2) | `185.8 ms` frame |
| 19 | `harmonic_to_percussive_ratio` | T3 | Pitched harmonicity vs. percussive noise | Log energy ratio `log(E_harm / (E_perc + ε))` via causal-windowed STFT HPSS | `92.9 ms` frame |
| 20 | `tonal_centroid_flux` | T3 | Harmonic change rate / chord motion | Frame-to-frame L2 velocity `\|\|T(t) - T(t-1)\|\|_2` in 6D Tonnetz space (`librosa.feature.tonnetz`) | `20 ms` step |
| 21 | `mode_major_minor_score` | T3 | Major vs. minor modal affect | Causal Krumhansl-Schmuckler chroma correlation gap: `max_k r(c, Maj_k) - max_k r(c, Min_k)` | `3.0 s` (150 frames) |
| 22 | `key_clarity` | T3 | Tonal stability vs. ambiguity | Maximum correlation `max_k r(c, Key_k)` across all 24 major/minor profiles | `3.0 s` (150 frames) |
| 23 | `harmonic_expectancy_surprise` | T3 | Prediction error (ITPRA tonal violation) | Cosine distance `1 - cos(c(t), m(t-1))` against causal EMA tonal memory (`τ = 3.0 s`) | `τ = 3.0 s` EMA |

---

## 4. Model Architecture: `CogMucs` (Cognitive Multi-Branch Temporal Network)

Building directly on the user's **24 May 2024 insight** comparing **de Berardinis et al. (2020) (`EmoMucs`)** against **Louro et al. (2024)**:

```text
  Audio ──▶ Unified Causal 50Hz Extractor (Tier 1 / 2 / 3) ──▶ 10Hz Causal Grid [B, T, C] + mask [B, T]
               │
               ├──▶ Branch 1: Melody   (Causal TCN or Gated-GRU) ──┐
               ├──▶ Branch 2: Rhythm   (Causal TCN or Gated-GRU) ──┼──▶ E / M / L Fusion + Element Gates w_d(t)
               ├──▶ Branch 3: Timbre   (Causal TCN or Gated-GRU) ──┤        │
               └──▶ Branch 4: Harmony  (Causal TCN or Gated-GRU) ──┘        ▼
                                                            ┌───────────────┴───────────────┐
                                                            ▼                               ▼
                                                Static Head (Coimbra 4Q)       Dynamic Head (DEAM / PMEmo)
                                                Masked Attention Pooling        Aligned 2 Hz [V(t), A(t)]
                                                → 4-Quadrant Softmax            → Masked Lin's CCC + MSE
```

### 4.1 Branch Encoder Options (Replacing the Broken Flat MLP)
1. **Branch Encoder A — Dilated Causal TCN (`models/tcn.py`, ~75k params total):**  
   4–5 residual causal dilated `Conv1d` blocks (`kernel_size=3`, `dilations=[1, 2, 4, 8, 16]`, `channels=32–48`, LayerNorm + GELU + Dropout). Exact `12.5 s` causal receptive field at `10 Hz`, fast parallel training on CPU/GPU, and crisp impulse-lag analysis.
2. **Branch Encoder B — Feature-Gated Causal GRU (`models/gru.py`, ~50k params total):**  
   Lightweight per-timestep feature attention gate `α(t, f) ∈ [0, 1]` followed by a 2-layer `GRU(hidden=48–64)` (causal default, with optional `bidirectional=True` flag for non-causal upper-bound ablations). Retains unbounded recurrent memory across 45 s clips and `0–15 s` warmup.

### 4.2 Demystifying `E / M / L` Fusion from `EmoMucs` (de Berardinis et al., 2020)
In your 24 May 2024 notes (`prog_img4.png` / `prog_img7.png`), `E`, `M`, and `L` mark the three fusion depths tested in `EmoMucs`:
- **`E` (Early / Feature-level Fusion):** Concatenate branch latent feature sequences immediately after each element's temporal encoder, before shared downstream layers.
- **`M` (Mid-level Fusion):** Each cognitive branch passes through its own projection/FC block first, then branch representations are fused via learnable **Cognitive Element Attention Gates** (`w_melody, w_rhythm, w_timbre, w_harmony`). *(Recommended default for interpretability.)*
- **`L` (Late / Decision-level Fusion):** Each branch makes an independent emotion prediction (e.g., Melody-only V-A / 4Q logits), and predictions are combined by weighted summation. Useful for quantifying standalone predictive power of each musicality element.

### 4.3 Benchmark Baselines
- **Panda et al. (2020) Tabular Controls (`models/baselines/panda_tabular.py`):** SVM & Louro et al. (2024) Fig. 4 DNN on Panda's `70` baseline, `top-100`, and `1714` static features.
- **Static Summary Control (`models/baselines/summary_baseline.py`):** Track-level `mean`/`std` of our *exact same* Tier 1/2/3 channels → Ridge / MLP (isolates the value of frame-by-frame trajectories over global averages).
- **Spectrogram Control (`models/baselines/mel_crnn.py`):** Compact log-Mel Choi FCN-4 / CRNN baseline (quantifies performance-vs-interpretability trade-off).

---

## 5. Two-Experiment Scientific Protocol & Multi-Dataset Roadmap

### 5.1 Experiment 1: Predictive Parsimony & Multi-Dataset Ladder
Following the handwritten 6-dataset plan (`prog_img8.png`) and dataset survey (`rev_img2.png`):

| Stage | Dataset | Task & Labels | Protocol & Anti-Leakage Rules | Research Question Answered |
|---|---|---|---|---|
| **Stage 1 (Redemption)** | **Coimbra 4Q** (Panda et al., 2020; ~900 × 30 s clips) | Static 4-Quadrant (`Q1–Q4`) classification | Deterministic stratified 10-fold CV; masked sequence pooling (keep all clips—no z-score outlier dropping!); scaler fit per-fold on train only | Does fixing the 6 bugs + Tier 1/2/3 contours jump from `36.07% F1` toward Panda's `67.5%–76.4% F1` using 10–40× fewer, human-grounded features? |
| **Stage 2 (Core Time-Series)** | **DEAM** (1,744 × 45 s + 58 full songs) → **PMEmo** (794 pop clips) | Continuous **`2.0 Hz`** `[V(t), A(t)]` trajectories | Feed full `0–45 s` audio; **mask Lin's CCC + MSE loss & metrics to `t ≥ 15.0 s`** on DEAM; song/artist-disjoint splits | Do frame-by-frame cognitive trajectories beat static mean/std controls and generalize to pop music (PMEmo + optional EDA comparison)? |
| **Stage 3 (Generalization)** | **EmoMusic** (744 clips) & **AMG1608** (1,608 clips) *(+ optional MUSAV)* | Static & dynamic V-A regression | Deterministic song/artist-disjoint splits | Cross-corpus robustness across the full 6-dataset benchmark envisioned in `MUSEC progress`. |

### 5.2 Experiment 2: Causal Cognitive Perturbation Validation (`evaluation/perturbation.py`)
Operationalizes the **10 May 2024 Experiment 2** design (**Grimaud & Eerola, 2022**): apply controlled interventions to neutral/test excerpts—both at the **raw audio level** (so DSP recalculates) and **feature-contour level**—and verify directional consistency with human music psychology:
- **Pitch shift (`±4 semitones`):** Higher pitch → expected increase in Valence / Arousal (shift toward `Q1`).
- **Tempo stretch (`0.8×` vs. `1.25×`):** Faster tempo / higher onset density → expected significant positive shift in Arousal (`A(t)`).
- **Spectral tilt / Brightness EQ (`±6 dB` high-shelf):** Brighter centroid → higher Arousal and sharpness.
- **Loudness swell / gain (`±6 dB`) & Dissonance injection (tritone/cluster overlay):** Increased roughness → expected negative shift in Valence (`V(t)`) + higher Arousal (`Q2`).

---

## 6. Phased Implementation Roadmap

- **Phase 0 (Current): Synthesis & Realigned Architecture**  
  Synthesized `MUSEC_Review MER` + `MUSEC progress`, diagnosed the June 2024 `36.07% F1` run, and finalized `ARCHITECTURE.md` v2 + `.agent/HANDOVER.md` v2.
- **Phase 1: Repo Scaffold, Unified Causal DSP Pipeline & Synthetic Verification**  
  Initialize Git, `pyproject.toml`, modular folders, `50 Hz` causal clock, Tier 1/2/3 extractors, and synthetic DSP unit tests (pure sine pitch, major vs. minor triad, tritone dissonance, 120 BPM click track, strict causality invariance check, zero NaNs).
- **Phase 2: Stage 1 Redemption on Coimbra 4Q**  
  Implement `data/coimbra4q.py`, `CogMucs` (`TCN` & `GRU` with padding masks), Panda tabular baselines, and Tier 1 → Tier 2 → Tier 3 ablations.
- **Phase 3: Stage 2 Dynamic Time-Series on DEAM & PMEmo**  
  Implement `deam.py` (`t ≥ 15 s` warmup masking), `pmemo.py`, Masked Lin's CCC loss, and static-vs-dynamic comparison.
- **Phase 4: Experiment 2 Causal Perturbations & Interpretability Suite**  
  Build `evaluation/perturbation.py`, E/M/L fusion & Leave-One-Branch-Out ablations, and interactive/notebook trajectory dashboards.
- **Phase 5: Cross-Dataset Generalization (`EmoMusic`, `AMG1608`) & Manuscript Synthesis**  
  Consolidate figures and tables back into `MUSEC_Review MER` and experimental reports.
