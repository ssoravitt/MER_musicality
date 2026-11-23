"""Dataset loaders, deterministic splitters, and temporal alignment utilities."""

from data.alignment import collate_padded_sequences, make_deam_warmup_mask
from data.base import MERRecord, resolve_data_dir
from data.coimbra4q import QUADRANT_TO_IDX, build_coimbra4q_splits, discover_coimbra4q
from data.deam import build_deam_splits, discover_deam, load_deam_dynamic_va

__all__ = [
    "QUADRANT_TO_IDX",
    "MERRecord",
    "build_coimbra4q_splits",
    "build_deam_splits",
    "collate_padded_sequences",
    "discover_coimbra4q",
    "discover_deam",
    "load_deam_dynamic_va",
    "make_deam_warmup_mask",
    "resolve_data_dir",
]
