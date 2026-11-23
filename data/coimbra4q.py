"""Coimbra 4-Quadrant Audio Emotion Dataset (Panda et al., 2020) loader & splitter."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data.base import MERRecord, resolve_data_dir

QUADRANT_TO_IDX: dict[str, int] = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
AUDIO_EXTS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")


def auto_discover_kaggle_4q(input_root: str | Path = "/kaggle/input") -> Path | None:
    """Locate a mounted Kaggle 4Q dataset root containing `Q1..Q4` directories."""
    base = Path(input_root)
    if not base.exists():
        return None
    candidates = sorted(p for p in base.rglob("Q1") if p.is_dir())
    for q1_dir in candidates:
        parent = q1_dir.parent
        if all((parent / q_name).is_dir() for q_name in ("Q2", "Q3", "Q4")):
            return parent
    return None


def discover_coimbra4q(data_root: str | Path | None = None) -> list[MERRecord]:
    """Discover Coimbra 4Q tracks from either `Q1..Q4` folders or metadata CSVs."""
    root = resolve_data_dir("coimbra4q", data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Coimbra 4Q directory not found at {root}. "
            "Download from http://mir.dei.uc.pt/downloads.html or Kaggle "
            "'sparshg/4q-audio-emotion-dataset-russell' into data/raw/coimbra4q/."
        )

    records: list[MERRecord] = []
    # Layout A: subdirectories Q1, Q2, Q3, Q4 (recursive search handles nested archive roots)
    for q_name, q_idx in QUADRANT_TO_IDX.items():
        for q_dir in root.rglob(q_name):
            if not q_dir.is_dir():
                continue
            for audio_path in sorted(q_dir.iterdir()):
                if audio_path.suffix.lower() in AUDIO_EXTS:
                    records.append(
                        MERRecord(
                            track_id=audio_path.stem,
                            audio_path=audio_path,
                            label=q_idx,
                        )
                    )

    if records:
        dedup = {r.track_id: r for r in records}
        return [dedup[k] for k in sorted(dedup)]

    # Layout B: CSV metadata mapping filename -> Quadrant
    audio_map = {
        p.stem: p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    }
    for csv_path in sorted(root.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        q_col = cols.get("quadrant") or cols.get("emotion") or cols.get("label")
        id_col = cols.get("song") or cols.get("audio") or cols.get("id") or cols.get("filename")
        if q_col and id_col:
            for _, row in df.iterrows():
                stem = Path(str(row[id_col])).stem
                q_str = str(row[q_col]).strip().upper()
                if stem in audio_map and q_str in QUADRANT_TO_IDX:
                    records.append(
                        MERRecord(
                            track_id=stem,
                            audio_path=audio_map[stem],
                            label=QUADRANT_TO_IDX[q_str],
                        )
                    )
            if records:
                return records

    raise RuntimeError(f"Could not locate Q1-Q4 audio tracks under {root}.")


def build_coimbra4q_splits(
    records: list[MERRecord],
    n_folds: int = 10,
    seed: int = 42,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Create deterministic stratified K-fold splits without dropping valid clips."""
    labels = [int(r.label) for r in records if r.label is not None]
    if len(labels) != len(records):
        raise ValueError("All Coimbra 4Q records must have integer quadrant labels.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    indices = list(range(len(records)))
    fold_assign = [0] * len(records)
    for fold_id, (_, val_idx) in enumerate(skf.split(indices, labels)):
        for idx in val_idx:
            fold_assign[idx] = fold_id

    for rec, fold_id in zip(records, fold_assign):
        rows.append(
            {
                "track_id": rec.track_id,
                "label": rec.label,
                "fold": fold_id,
                "audio_path": str(rec.audio_path),
            }
        )

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
