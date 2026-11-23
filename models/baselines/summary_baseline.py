"""Static track-level mean/std baseline over the same cognitive feature channels."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def pool_summary_stats(sequences: list[np.ndarray]) -> np.ndarray:
    """Collapse variable-length `[T_i, C]` time-series into `[N, 2*C]` (mean, std) vectors."""
    rows = []
    for seq in sequences:
        mu = np.mean(seq, axis=0)
        sd = np.std(seq, axis=0)
        rows.append(np.concatenate([mu, sd], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def build_summary_4q_baseline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Standardized logistic regression baseline over global track mean/std stats."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=42)),
        ]
    )


def build_summary_va_baseline(alpha: float = 10.0) -> Pipeline:
    """Standardized Ridge regression baseline over global track mean/std stats."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha, random_state=42)),
        ]
    )
