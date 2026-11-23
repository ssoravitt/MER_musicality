# Project Handover & Cross-Machine Continuity Log

> Read `ARCHITECTURE.md` and this file first when resuming work on a new machine or in a fresh agent session.

---

## 1. Current Status
- **Stage:** `Phase 1 — Core Scaffolding, Docker Setup, Causal DSP & CogMucs Implemented & Tested`
- **Last Updated:** 2026-11-23
- **Workspace Root:** `MER_musicality/`

## 2. Source Google Docs (MUSEC Continuity Anchors)
| Document | Google Doc ID | URL |
|---|---|---|
| `MUSEC_Review MER` | `1_3XOZJG--xjW-HibT0rt-m5dXtZb-YJMPDXRMzUNnAI` | https://docs.google.com/document/d/1_3XOZJG--xjW-HibT0rt-m5dXtZb-YJMPDXRMzUNnAI/edit |
| `MUSEC progress ` | `1B7BSoqhfiDEjmPaKGPbF75SkVFOswsqDojoeoVigYHI` | https://docs.google.com/document/d/1B7BSoqhfiDEjmPaKGPbF75SkVFOswsqDojoeoVigYHI/edit |

## 3. Record of June 2024 MUSEC Baseline & Diagnosed Root Causes
Do not repeat these issues in Phase 1+:
- **Baseline Recorded (14 June 2024):** Coimbra 4Q dataset (Panda et al., 2020) using Essentia Melody + Essentia Rhythm + Librosa Spectral Centroid (`T = 1100`, Louro et al. 2024 Fig. 4 DNN without BatchNorm, 500 epochs, `lr = 0.001`, 10 folds) → **`Accuracy = 39.84%`, `Macro F1 = 36.07%`** (vs. Panda et al. SVM `67.5%–76.4% F1`).
- **6 Root Causes Diagnosed:**
  1. **Flat MLP Channel Isolation:** `[64, 3, 1100]` passed into `Dense` layers processed each channel independently (`[64, 3, 4]` → squeezed to `[64, 1, 4]`) with zero cross-element fusion or time-shift invariance.
  2. **Unnormalized Physical Units + BatchNorm Removed:** Raw `Hz` pitch/centroid values drove initial CE loss `> 150` (vs. `ln(4) ≈ 1.39`).
  3. **Mismatched Toolbox Time Clocks:** Essentia melody/rhythm and Librosa timbre had incompatible frame/hop rates (causing oversized rhythm `.npy` dumps and misaligned indices).
  4. **Unvoiced `0 Hz` Artifacts:** Unvoiced pitch drops read as `0 Hz` rather than MIDI + explicit `voicing_prob` mask.
  5. **Missing Loudness & Harmony Cues:** Pitch/onset/centroid alone under-resolve valence contrast (`Q1 vs. Q2` and `Q3 vs. Q4`).
  6. **Unstable Random Re-Splitting & Outlier Clip Dropping:** Dropped short/long clips (`MT0013621344`, `MT0013914319` @ 435; `MT0005171805`, `MT0011051663`) instead of sequence padding + boolean masking.

## 4. Non-Negotiable Architectural Invariants (v2)
1. **3-Tier Cognitive Feature Ladder (Non-Overlapping):**
   - **Tier 1 (4 ch):** `pitch_f0_midi`, `pitch_voicing_prob`, `onset_strength_flux`, `spectral_centroid_bark` (pure MUSEC redemption).
   - **Tier 2 (6 ch):** Tier 1 + `loudness_perceptual_db`, `loudness_delta` (McDermott et al., 2008 auditory contours).
   - **Tier 3 (23 ch):** Full 4-branch schema — Melody (6) + Rhythm (6) + Timbre (5) + Harmony (6). No blind openSMILE/MFCC summary dumps.
2. **Unified Causal `50.0 Hz` Base Clock:** `sr = 22050 Hz`, `hop_length = 441` (`20.0 ms` exact; `25` frames = `0.500 s` exact for `2 Hz` labels), downsampled causally to `10.0 Hz` (or `2.0 Hz`).
3. **Strict Causality (`center=False` + Left-Only Padding):** Never use `librosa` default `center=True` in causal extraction. Frame `t` depends strictly on audio `≤ t`.
4. **Native Windows + Docker Compatibility:** `librosa + scipy + numpy` default backend works natively across Windows/macOS/Linux; `essentia` is an optional Linux/Docker adapter (`INSTALL_ESSENTIA=true`).
5. **CogMucs Multi-Branch Architecture:** Per-element temporal encoders (`Causal TCN` or `Feature-Gated GRU`) supporting `E/M/L` fusion, boolean padding masks (no dropping valid clips), and dual readout heads (Static 4Q/V-A + Dynamic `2 Hz` V-A).
6. **Anti-Leakage & Warmup Masking:** Deterministic CSV splits in `data/splits/`; train-only scaler fitting; on DEAM, feed full `0–45 s` audio for causal warmup and mask loss/metrics strictly to `t ≥ 15.0 s`.
7. **Two-Experiment Scientific Verification:** Experiment 1 (predictive parsimony vs. SOTA across datasets) + Experiment 2 (`evaluation/perturbation.py` causal audio/contour interventions verifying human-aligned emotion shifts).

## 5. Confirmed User Decisions
| # | Decision | Confirmed Choice | Notes |
|---|---|---|---|
| 1 | **Datasets (Phase 1+)** | **Coimbra 4Q** (Stage 1) + **DEAM** (Stage 2) | Local copy lost from former PC; verified download sources documented in `README.md` (University of Coimbra `http://mir.dei.uc.pt/downloads.html` + Kaggle `sparshg/4q-audio-emotion-dataset-russell`; CVML Geneva `http://cvml.unige.ch/databases/DEAM/` + Kaggle `imuhammad/deam-dataset-emotional-analysis-in-music`). Both loaders wired in Phase 1. |
| 2 | **Compute & Docker / Unattended Runs** | **Docker (`python:3.11-slim`) + Unattended Cloud/NAS Runners** | Local Docker Desktop stops when PC powers down (and cannot bind-mount Windows UNC paths `\\kfs\...` directly; clone to local disk first). Best unattended options: (a) **Kaggle Free GPU "Save & Run All" Background Commit**, (b) cheap cloud Docker GPU (RunPod/Vast.ai) or always-on home NAS, (c) GitHub Actions CPU runs. Optional `essentia` Linux build arg supported in `Dockerfile`. |
| 3 | **Feature & Model Scope** | **Full Tier 1 → Tier 2 → Tier 3 Ladder + `CogMucs` (`TCN` & `GRU`)** | User approved full plan. Modular config selects Tier 1/2/3 and `tcn`/`gru` cleanly for Day-1 ablations. |

## 6. Next Steps (Phase 2 — First Real Dataset Runs)
1. Push repository to GitHub (from a local clone or once Git safe directory is configured).
2. Download **Coimbra 4Q** and/or **DEAM** into `data/raw/` (or run directly on **Kaggle** where both datasets are pre-hosted).
3. Run feature extraction CLI: `python -m scripts.extract --dataset coimbra4q --tier 3` (and `deam`).
4. Run Stage 1 **Coimbra 4Q** ablation table (`Tier 1 -> Tier 2 -> Tier 3` with `CogMucs-TCN` & `CogMucs-GRU` vs. June 2024 `36.07% F1` and Panda `67.5%-76.4% F1`).
5. Run Stage 2 **DEAM** dynamic `2 Hz` V-A training + Experiment 2 causal perturbations (`evaluation/perturbation.py`).

## 7. Cross-Machine Sync Notes (GitHub Workflow)
- **Never commit large binaries:** `data/raw/`, `data/processed/`, `checkpoints/`, `.venv/`, and local logs are in `.gitignore`.
- **Track reproducible specs:** Commit `configs/**/*.yaml`, `data/splits/*.csv`, `ARCHITECTURE.md`, `Dockerfile`, and `.agent/HANDOVER.md`.
- **Path portability:** Override dataset root via `MER_DATA_DIR` environment variable across Windows, Docker (`/workspace/data/raw`), and Kaggle (`/kaggle/input/...`).
