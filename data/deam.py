"""DEAM (MediaEval Database for Emotional Analysis in Music) loader & dynamic V-A aligner."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.base import MERRecord, resolve_data_dir


def discover_deam(data_root: str | Path | None = None) -> list[MERRecord]:
    """Discover DEAM audio files under `MEMD_audio/` or nested DEAM archive directory."""
    root = resolve_data_dir("deam", data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"DEAM directory not found at {root}. "
            "Download from http://cvml.unige.ch/databases/DEAM/ or Kaggle "
            "'imuhammad/deam-dataset-emotional-analysis-in-music' into data/raw/deam/."
        )

    records: list[MERRecord] = []
    for p in sorted(root.rglob("*.mp3")):
        if p.stem.isdigit():
            records.append(MERRecord(track_id=p.stem, audio_path=p))
    if not records:
        for p in sorted(root.rglob("*.mp3")):
            records.append(MERRecord(track_id=p.stem, audio_path=p))
    if not records:
        raise RuntimeError(f"No DEAM MP3 audio files found under {root}.")
    return records


def load_deam_dynamic_va(
    valence_csv: str | Path,
    arousal_csv: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load DEAM 2 Hz dynamic valence & arousal CSVs (`sample_15000ms` .. `sample_45000ms`).

    Returns `(targets_by_song_id, label_times_s)` where each target array has shape `[T_2hz, 2]`.
    """
    v_df = pd.read_csv(valence_csv)
    a_df = pd.read_csv(arousal_csv)

    v_df.columns = [c.strip() for c in v_df.columns]
    a_df.columns = [c.strip() for c in a_df.columns]
    v_df["song_id"] = v_df["song_id"].astype(str)
    a_df["song_id"] = a_df["song_id"].astype(str)

    sample_cols = [c for c in v_df.columns if c.startswith("sample_") and c.endswith("ms")]
    if not sample_cols:
        raise ValueError("No `sample_*ms` time columns found in DEAM valence CSV.")

    sample_cols = sorted(sample_cols, key=lambda c: int(c.replace("sample_", "").replace("ms", "")))
    times_s = np.array(
        [int(c.replace("sample_", "").replace("ms", "")) / 1000.0 for c in sample_cols],
        dtype=np.float32,
    )

    a_indexed = a_df.set_index("song_id")
    targets: dict[str, np.ndarray] = {}
    for _, row in v_df.iterrows():
        sid = row["song_id"]
        if sid not in a_indexed.index:
            continue
        v_vals = row[sample_cols].to_numpy(dtype=np.float32)
        a_vals = a_indexed.loc[sid, sample_cols].to_numpy(dtype=np.float32)
        va = np.stack([v_vals, a_vals], axis=-1)
        if np.all(np.isfinite(va)):
            targets[sid] = va
    return targets, times_s


def build_deam_splits(
    records: list[MERRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Create deterministic song-disjoint train/val/test splits for DEAM."""
    ids = [r.track_id for r in records]
    train_ids, rem_ids = train_test_split(
        ids, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
    )
    rel_test = test_ratio / max(val_ratio + test_ratio, 1e-8)
    val_ids, test_ids = train_test_split(
        rem_ids, test_size=rel_test, random_state=seed, shuffle=True
    )

    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "val" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})

    df = pd.DataFrame(
        [
            {
                "track_id": r.track_id,
                "split": split_map[r.track_id],
                "audio_path": str(r.audio_path),
            }
            for r in records
        ]
    )
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
