"""Shared dataset abstractions and portable path resolution."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MERRecord:
    """Single track record independent of raw storage location."""

    track_id: str
    audio_path: Path
    label: int | None = None
    split: str | None = None
    fold: int | None = None
    artist_id: str | None = None


def resolve_data_dir(subdir: str = "", base_dir: str | Path | None = None) -> Path:
    """Resolve raw dataset directory via `base_dir` -> `$MER_DATA_DIR` -> `data/raw`.

    When `base_dir` is explicitly provided, uses `base_dir / subdir` if that
    subdirectory exists, otherwise uses `base_dir` directly (e.g. Kaggle mounts).
    """
    if base_dir is not None:
        root = Path(base_dir)
        if subdir and (root / subdir).exists():
            return root / subdir
        return root

    if env_dir := os.environ.get("MER_DATA_DIR"):
        root = Path(env_dir)
    else:
        root = Path("data") / "raw"

    if not subdir:
        return root
    if (root / subdir).exists():
        return root / subdir
    if root.exists() and any((root / marker).exists() for marker in ("Q1", "MEMD_audio")):
        return root
    return root / subdir
