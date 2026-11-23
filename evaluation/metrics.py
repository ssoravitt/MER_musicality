"""Evaluation metrics for static 4Q classification and dynamic 2 Hz V-A regression."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_4q_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute overall Accuracy and Macro-F1 for 4-quadrant classification."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def lins_ccc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Lin's Concordance Correlation Coefficient for 1D arrays."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(yt) < 2:
        return 0.0
    mu_t, mu_p = yt.mean(), yp.mean()
    var_t, var_p = np.var(yt), np.var(yp)
    cov = np.mean((yt - mu_t) * (yp - mu_p))
    return float((2.0 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2 + eps))


def compute_dynamic_va_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute masked global and mean per-song CCC, Pearson r, and RMSE for `[B, T, 2]` V-A."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if mask is None:
        m = np.ones(p.shape[:2], dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)

    dim_names = ("valence", "arousal")
    out: dict[str, float] = {}
    for d_idx, d_name in enumerate(dim_names):
        p_all = p[..., d_idx][m]
        t_all = t[..., d_idx][m]
        out[f"{d_name}_ccc_global"] = lins_ccc(t_all, p_all)
        out[f"{d_name}_rmse_global"] = float(np.sqrt(np.mean((t_all - p_all) ** 2)))

        song_r, song_ccc = [], []
        for b in range(p.shape[0]):
            mb = m[b]
            if mb.sum() < 2:
                continue
            tb, pb = t[b, mb, d_idx], p[b, mb, d_idx]
            song_ccc.append(lins_ccc(tb, pb))
            if np.std(tb) > 1e-8 and np.std(pb) > 1e-8:
                song_r.append(float(np.corrcoef(tb, pb)[0, 1]))
        out[f"{d_name}_ccc_persong"] = float(np.mean(song_ccc)) if song_ccc else 0.0
        out[f"{d_name}_r_persong"] = float(np.mean(song_r)) if song_r else 0.0
    return out
