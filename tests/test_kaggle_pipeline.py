"""Tests for Kaggle path resolution, Tier slicing parity, and Stage 1 Coimbra 4Q benchmark."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.base import resolve_data_dir
from data.coimbra4q import auto_discover_kaggle_4q, build_coimbra4q_splits, discover_coimbra4q
from features.clock import SR
from features.pipeline import extract_features
from features.tiers import slice_tier_from_full
from scripts.benchmark_coimbra4q import run_benchmark


def test_resolve_data_dir_kaggle_direct_path(tmp_path: Path) -> None:
    """Explicit `--data-dir` works directly without forcing a `/coimbra4q` child folder."""
    mount = tmp_path / "4q-audio-emotion-dataset-russell"
    (mount / "Q1").mkdir(parents=True)
    assert resolve_data_dir("coimbra4q", mount) == mount

    nested_root = tmp_path / "raw"
    (nested_root / "coimbra4q").mkdir(parents=True)
    assert resolve_data_dir("coimbra4q", nested_root) == nested_root / "coimbra4q"


def test_auto_discover_kaggle_4q(tmp_path: Path) -> None:
    """Locate `Q1..Q4` directories under a Kaggle-style `/kaggle/input` mount tree."""
    dataset_root = tmp_path / "kaggle" / "input" / "4q-audio-emotion-dataset-russell" / "archive"
    for idx, q in enumerate(("Q1", "Q2", "Q3", "Q4")):
        q_dir = dataset_root / q
        q_dir.mkdir(parents=True)
        (q_dir / f"track_{idx}.mp3").write_bytes(b"")

    found = auto_discover_kaggle_4q(tmp_path / "kaggle" / "input")
    assert found == dataset_root
    records = discover_coimbra4q(found)
    assert len(records) == 4
    assert [r.label for r in records] == [0, 1, 2, 3]


@pytest.mark.parametrize("target_tier", [1, 2])
def test_slice_tier_from_full_matches_direct_extraction(target_tier: int) -> None:
    """Slicing cached Tier-3 arrays matches extracting Tier 1 or Tier 2 directly."""
    t = np.arange(int(3.0 * SR), dtype=np.float32) / SR
    y = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    full = extract_features(y, sr=SR, tier=3, target_hz=10.0)
    direct = extract_features(y, sr=SR, tier=target_tier, target_hz=10.0)
    sliced = slice_tier_from_full(full.features, target_tier)

    assert sliced.shape == direct.features.shape
    np.testing.assert_allclose(sliced, direct.features, atol=1e-6)


def test_kaggle_notebook_valid_json() -> None:
    """Ensure the Kaggle starter notebook is valid nbformat JSON with key commands."""
    nb_path = Path("notebooks/kaggle_stage1_coimbra4q.ipynb")
    if not nb_path.exists():
        nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_stage1_coimbra4q.ipynb"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    joined = "\n".join("".join(c["source"]) for c in data["cells"])
    assert "scripts.extract" in joined
    assert "scripts.benchmark_coimbra4q" in joined


def test_benchmark_coimbra4q_end_to_end_smoke(tmp_path: Path) -> None:
    """End-to-end 2-fold smoke run produces all tables, plots, and fold-0 checkpoint."""
    rng = np.random.default_rng(2026)
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"

    # 16 synthetic Tier-3 tracks (4 per quadrant)
    records = []
    from data.base import MERRecord

    for q in range(4):
        for k in range(4):
            tid = f"track_q{q}_{k}"
            feats = rng.normal(loc=float(q), scale=0.5, size=(30, 23)).astype(np.float32)
            np.savez_compressed(proc_dir / f"{tid}.npz", features=feats)
            records.append(MERRecord(track_id=tid, audio_path=tmp_path / f"{tid}.mp3", label=q))

    splits_csv = tmp_path / "splits.csv"
    build_coimbra4q_splits(records, n_folds=2, seed=42, out_csv=splits_csv)

    summary_df = run_benchmark(
        processed_dir=proc_dir,
        splits_csv=splits_csv,
        out_dir=out_dir,
        n_folds=2,
        epochs=2,
        batch_size=4,
        seed=42,
        skip_ablation=False,
    )

    assert len(summary_df) >= 12  # 4 models x 3 tiers + 4 LOBO ablations = 16 rows
    for fname in (
        "fold_metrics.csv",
        "summary_table.csv",
        "summary_report.md",
        "branch_attention_weights.csv",
        "tier_ablation_plot.png",
        "branch_weights_by_quadrant.png",
        "confusion_matrices.png",
        "checkpoints/best_tier3_fold0.pt",
    ):
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing benchmark output: {fpath}"
        assert fpath.stat().st_size > 0
    pd.read_csv(out_dir / "summary_table.csv")
