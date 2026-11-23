# `MER_musicality` — Cognition-First Time-Series Music Emotion Recognition

A research & hobby framework investigating whether **psychologically and biologically grounded musicality trajectories** (Melody, Rhythm, Timbre/Loudness, and Harmony) modeled frame-by-frame with lightweight causal temporal networks (`CogMucs`) rival or outperform brute-force track-level summary statistics while remaining mechanistically interpretable.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full scientific foundation, June 2024 MUSEC diagnosis, 23-channel feature schema, and `CogMucs` design.

---

## 1. Datasets (Verified Sources)

Large raw audio files are gitignored (`data/raw/`). Override storage location via the `MER_DATA_DIR` environment variable.

### Stage 1: Coimbra 4-Quadrant Audio Emotion Dataset (Panda et al., 2020)
- **900** × ~30 s clips annotated into Russell's 4 quadrants (`Q1` Happy, `Q2` Tense/Angry, `Q3` Sad, `Q4` Calm) + static baseline/novel feature CSVs.
- **Official University of Coimbra MIR Group Page:** http://mir.dei.uc.pt/downloads.html
- **Kaggle Mirror:** https://www.kaggle.com/datasets/sparshg/4q-audio-emotion-dataset-russell
- Place extracted files under `data/raw/coimbra4q/` (expects either `Q1/..Q4/` folders or audio + quadrant metadata CSV).

### Stage 2: DEAM (MediaEval Database for Emotional Analysis in Music)
- **1,744** × 45 s CC-licensed excerpts + **58** full songs with **`2.0 Hz`** continuous Valence & Arousal trajectories (`t = 15 s .. 45 s`).
- **Official University of Geneva CVML Page:** http://cvml.unige.ch/databases/DEAM/
- **Kaggle Mirror:** https://www.kaggle.com/datasets/imuhammad/deam-dataset-emotional-analysis-in-music
- Place extracted files under `data/raw/deam/` (`MEMD_audio/` + `annotations/`).

---

## 2. Running Locally with Docker

> **Important Windows Note:** If this folder lives on a UNC network share (`\\kfs\...`), Docker Desktop on Windows cannot bind-mount `\\...` paths directly. Clone or copy the repo to a local drive (e.g., `C:\Users\<you>\MER_musicality`) before running `docker compose`.

```bash
# Build CPU image (add --build-arg INSTALL_ESSENTIA=true for optional Essentia on Linux x86_64)
docker compose build mer

# Run full DSP & model unit-test suite
docker compose run --rm mer pytest -v

# Extract Tier 3 features on Coimbra 4Q
docker compose run --rm mer python -m scripts.extract --dataset coimbra4q --tier 3

# Train CogMucs-TCN on Coimbra 4Q (with Tier 3 features)
docker compose run --rm mer python -m scripts.train --dataset coimbra4q --tier 3 --model tcn

# Train CogMucs-GRU dynamically on DEAM (2 Hz V-A, masked t >= 15s)
docker compose run --rm mer python -m scripts.train --dataset deam --tier 3 --model gru
```

---

## 3. Running Experiments While Your PC Is Turned Off

Because local Docker containers stop as soon as your PC sleeps or shuts down, here are the three simplest unattended options for a hobby project:

1. **Free Kaggle GPU "Save & Run All" Background Commit (Recommended — \$0):**
   - Both **Coimbra 4Q** and **DEAM** are already hosted as public Kaggle datasets (zero manual upload needed!).
   - Clone your GitHub repo inside a Kaggle Notebook, enable GPU (`T4 x2` / `P100`, up to ~30 hrs/week free), click **Save Version → Save & Run All (Commit)**, and turn off your PC. Kaggle executes unattended for up to 12 hours and saves all logs, metrics, and checkpoints.
2. **Always-On Home NAS or Low-Cost Cloud Docker (RunPod / Vast.ai / Hetzner):**
   - Push this repo to GitHub, pull it onto an always-on Linux/NAS box or a cheap cloud GPU/CPU instance (~USD 0.10–0.30/hr), and run `docker compose up -d` or `nohup docker compose run ... > train.log 2>&1 &`.
3. **GitHub Actions (Free CPU CI):**
   - Ideal for automatically running `pytest -v` on every push and short CPU benchmark experiments cleanly in GitHub's cloud.
