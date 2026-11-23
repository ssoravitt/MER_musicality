"""10-Fold Stage 1 Benchmark Runner for Coimbra 4Q (Tiers 1->2->3, LOBO Ablations & Plots)."""

import argparse
import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.alignment import collate_padded_sequences
from evaluation.metrics import compute_4q_metrics
from features.schema import BRANCH_NAMES
from features.tiers import get_branch_indices, slice_tier_from_full
from models.baselines.panda_tabular import build_panda_svm
from models.baselines.summary_baseline import build_summary_4q_baseline, pool_summary_stats
from models.cogmucs import CogMucs
from models.losses import static_4q_loss

QUADRANT_NAMES: tuple[str, ...] = ("Q1 (Happy)", "Q2 (Tense/Angry)", "Q3 (Sad)", "Q4 (Calm)")
SHORT_QUADRANTS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cached_tracks(
    df: pd.DataFrame, processed_dir: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load all cached Tier-3 `[T, 23]` sequences whose `.npz` files exist."""
    kept_rows = []
    seqs: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        tid = str(row["track_id"])
        npz_path = processed_dir / f"{tid}.npz"
        if npz_path.exists():
            arr = np.load(npz_path)["features"].astype(np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                seqs[tid] = arr
                kept_rows.append(row)
    if not kept_rows:
        raise RuntimeError(f"No valid cached .npz feature files found under {processed_dir}.")
    return pd.DataFrame(kept_rows).reset_index(drop=True), seqs


def _split_inner_val(
    pool_df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create inner train/val split for early stopping without leaking outer test fold."""
    labels = pool_df["label"].to_numpy(dtype=int)
    n_classes = len(np.unique(labels))
    n_pool = len(pool_df)
    val_count = max(n_classes, round(0.12 * n_pool))
    if n_pool >= 2 * n_classes and val_count <= n_pool - n_classes:
        tr_df, va_df = train_test_split(
            pool_df, test_size=val_count, stratify=labels, random_state=seed
        )
    else:
        tr_df, va_df = train_test_split(
            pool_df, test_size=max(1, round(0.2 * n_pool)), random_state=seed
        )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


def _train_eval_cogmucs_fold(
    tr_seqs: list[np.ndarray],
    tr_labels: np.ndarray,
    va_seqs: list[np.ndarray],
    va_labels: np.ndarray,
    te_seqs: list[np.ndarray],
    te_labels: np.ndarray,
    branch_indices: dict[str, list[int]],
    branch_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, object]:
    """Fit scaler on inner-train only, train CogMucs, select best checkpoint on inner-val."""
    scaler = StandardScaler()
    scaler.fit(np.concatenate(tr_seqs, axis=0))

    tr_norm = [scaler.transform(s).astype(np.float32) for s in tr_seqs]
    va_norm = [scaler.transform(s).astype(np.float32) for s in va_seqs]
    te_norm = [scaler.transform(s).astype(np.float32) for s in te_seqs]

    model = CogMucs(
        branch_indices=branch_indices,
        branch_type=branch_type,
        fusion="mid",
        task="static_4q",
        hidden_dim=32 if branch_type == "tcn" else 48,
        dropout=0.2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    tr_y_t = torch.tensor(tr_labels, dtype=torch.long)

    best_score = (-1.0, float("inf"))
    best_state = copy.deepcopy(model.state_dict())

    xv_pad, mv_pad = collate_padded_sequences(va_norm)
    xv_dev, mv_dev = xv_pad.to(device), mv_pad.to(device)
    va_y_t = torch.tensor(va_labels, dtype=torch.long, device=device)

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_norm))
        for start in range(0, len(tr_norm), batch_size):
            idx = perm[start : start + batch_size].tolist()
            xb, mb = collate_padded_sequences([tr_norm[i] for i in idx])
            yb = tr_y_t[idx].to(device)
            opt.zero_grad()
            out = model(xb.to(device), mask=mb.to(device))
            loss = static_4q_loss(out.pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_out = model(xv_dev, mask=mv_dev)
            va_loss = static_4q_loss(va_out.pred, va_y_t).item()
            va_pred = va_out.pred.argmax(dim=-1).cpu().numpy()
            va_f1 = compute_4q_metrics(va_labels, va_pred)["macro_f1"]
        score = (va_f1, -va_loss)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xt_pad, mt_pad = collate_padded_sequences(te_norm)
        te_out = model(xt_pad.to(device), mask=mt_pad.to(device))
        te_preds = te_out.pred.argmax(dim=-1).cpu().numpy()
        te_bw = te_out.branch_weights.cpu().numpy()

    metrics = compute_4q_metrics(te_labels, te_preds)
    return {
        "metrics": metrics,
        "preds": te_preds,
        "branch_weights": te_bw,
        "state_dict": best_state,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }


def _plot_tier_ablation(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot Tier 1 -> Tier 2 -> Tier 3 Macro-F1 across Summary vs CogMucs models."""
    main_df = summary_df[summary_df["ablation"] == "full"].copy()
    models = ["Summary-LogReg", "Summary-RBF-SVM", "CogMucs-TCN", "CogMucs-GRU"]
    tiers = [1, 2, 3]
    x = np.arange(len(tiers), dtype=np.float64)
    width = 0.19

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for m_idx, model_name in enumerate(models):
        means, stds = [], []
        for t in tiers:
            sub = main_df[(main_df["model"] == model_name) & (main_df["tier"] == t)]
            means.append(float(sub["macro_f1_mean"].iloc[0]) * 100.0 if len(sub) else 0.0)
            stds.append(float(sub["macro_f1_std"].iloc[0]) * 100.0 if len(sub) else 0.0)
        offset = (m_idx - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_name)

    ax.axhline(36.07, color="#d62728", linestyle="--", linewidth=1.4, label="June 2024 MUSEC (36.1% F1)")
    ax.axhline(67.5, color="#7f7f7f", linestyle=":", linewidth=1.4, label="Panda 2020 SVM 70-feat (67.5% F1)")
    ax.axhline(76.4, color="#2ca02c", linestyle="-.", linewidth=1.4, label="Panda 2020 SVM 100-feat (76.4% F1)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Tier 1 (4 ch)\nMUSEC Minimal", "Tier 2 (6 ch)\n+ Loudness", "Tier 3 (23 ch)\nFull Cognitive"])
    ax.set_ylabel("Macro F1-Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Coimbra 4Q Stage 1: Controlled Feature Ladder & Architecture Comparison")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_branch_weights(bw_df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Plot mean learned element gate weights per true quadrant."""
    branches = [b for b in BRANCH_NAMES if b in bw_df.columns]
    grouped = bw_df.groupby("quadrant")[branches].mean().reindex(SHORT_QUADRANTS).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(grouped.index), dtype=np.float64)
    width = 0.8 / max(len(branches), 1)
    for b_idx, branch in enumerate(branches):
        offset = (b_idx - (len(branches) - 1) / 2.0) * width
        ax.bar(x + offset, grouped[branch].to_numpy(), width, label=branch.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(QUADRANT_NAMES)
    ax.set_ylabel("Mean Attention Gate Weight")
    ax.set_ylim(0, max(0.5, float(grouped.to_numpy().max()) * 1.25))
    ax.set_title(f"{model_name} (Tier 3): Cognitive Branch Attention by Emotion Quadrant")
    ax.legend(title="Musical Element")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path
) -> None:
    """Plot row-normalized 4x4 confusion matrix for out-of-fold predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3], normalize="true")
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(SHORT_QUADRANTS)
    ax.set_yticklabels(SHORT_QUADRANTS)
    ax.set_xlabel("Predicted Quadrant")
    ax.set_ylabel("True Quadrant")
    ax.set_title(title)

    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.55 else "black",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_benchmark(
    processed_dir: str | Path,
    splits_csv: str | Path,
    out_dir: str | Path,
    n_folds: int = 10,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    skip_ablation: bool = False,
) -> pd.DataFrame:
    """Execute full Coimbra 4Q Stage 1 benchmark and save tables, plots, and checkpoint."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(splits_csv)
    df, seqs_t3 = _load_cached_tracks(raw_df, Path(processed_dir))
    folds = sorted(int(f) for f in df["fold"].unique())[:n_folds]

    fold_records: list[dict[str, object]] = []
    oof_store: dict[str, dict[str, list]] = {
        "CogMucs-TCN": {"true": [], "pred": [], "bw": [], "fold0_ckpt": []},
        "CogMucs-GRU": {"true": [], "pred": [], "bw": [], "fold0_ckpt": []},
    }

    for fold_id in folds:
        test_df = df[df["fold"] == fold_id].reset_index(drop=True)
        pool_df = df[df["fold"] != fold_id].reset_index(drop=True)
        tr_df, va_df = _split_inner_val(pool_df, seed=seed + fold_id)

        pool_t3 = [seqs_t3[str(t)] for t in pool_df["track_id"]]
        tr_t3 = [seqs_t3[str(t)] for t in tr_df["track_id"]]
        va_t3 = [seqs_t3[str(t)] for t in va_df["track_id"]]
        te_t3 = [seqs_t3[str(t)] for t in test_df["track_id"]]

        pool_y = pool_df["label"].to_numpy(dtype=np.int64)
        tr_y = tr_df["label"].to_numpy(dtype=np.int64)
        va_y = va_df["label"].to_numpy(dtype=np.int64)
        te_y = test_df["label"].to_numpy(dtype=np.int64)

        for tier in (1, 2, 3):
            pool_seqs = [slice_tier_from_full(s, tier) for s in pool_t3]
            tr_seqs = [slice_tier_from_full(s, tier) for s in tr_t3]
            va_seqs = [slice_tier_from_full(s, tier) for s in va_t3]
            te_seqs = [slice_tier_from_full(s, tier) for s in te_t3]

            # 1. Static Summary Controls (track-level mean & std of same features)
            x_pool_stat = pool_summary_stats(pool_seqs)
            x_te_stat = pool_summary_stats(te_seqs)

            for stat_name, clf in (
                ("Summary-LogReg", build_summary_4q_baseline()),
                ("Summary-RBF-SVM", build_panda_svm()),
            ):
                clf.fit(x_pool_stat, pool_y)
                preds = clf.predict(x_te_stat)
                m = compute_4q_metrics(te_y, preds)
                fold_records.append(
                    {
                        "model": stat_name,
                        "tier": tier,
                        "ablation": "full",
                        "fold": fold_id,
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                    }
                )

            # 2. CogMucs-TCN and CogMucs-GRU
            branch_idx = get_branch_indices(tier)
            for b_type, model_label in (("tcn", "CogMucs-TCN"), ("gru", "CogMucs-GRU")):
                res = _train_eval_cogmucs_fold(
                    tr_seqs=tr_seqs,
                    tr_labels=tr_y,
                    va_seqs=va_seqs,
                    va_labels=va_y,
                    te_seqs=te_seqs,
                    te_labels=te_y,
                    branch_indices=branch_idx,
                    branch_type=b_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                m = res["metrics"]
                fold_records.append(
                    {
                        "model": model_label,
                        "tier": tier,
                        "ablation": "full",
                        "fold": fold_id,
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                    }
                )
                if tier == 3:
                    oof_store[model_label]["true"].append(te_y)
                    oof_store[model_label]["pred"].append(res["preds"])
                    oof_store[model_label]["bw"].append(res["branch_weights"])
                    if fold_id == folds[0]:
                        oof_store[model_label]["fold0_ckpt"].append(
                            {
                                "model_state_dict": res["state_dict"],
                                "scaler_mean": res["scaler_mean"],
                                "scaler_scale": res["scaler_scale"],
                                "branch_indices": branch_idx,
                                "branch_type": b_type,
                                "tier": 3,
                                "fold0_metrics": m,
                            }
                        )

    fold_df = pd.DataFrame(fold_records)
    t3_scores = (
        fold_df[(fold_df["tier"] == 3) & (fold_df["model"].isin(["CogMucs-TCN", "CogMucs-GRU"]))]
        .groupby("model")["macro_f1"]
        .mean()
    )
    best_model_name = str(t3_scores.idxmax())
    best_btype = "tcn" if best_model_name == "CogMucs-TCN" else "gru"

    # 3. Leave-One-Branch-Out (LOBO) Ablation on Tier 3 with best CogMucs architecture
    if not skip_ablation:
        full_b_idx = get_branch_indices(3)
        for fold_id in folds:
            test_df = df[df["fold"] == fold_id].reset_index(drop=True)
            pool_df = df[df["fold"] != fold_id].reset_index(drop=True)
            tr_df, va_df = _split_inner_val(pool_df, seed=seed + fold_id)

            tr_t3 = [seqs_t3[str(t)] for t in tr_df["track_id"]]
            va_t3 = [seqs_t3[str(t)] for t in va_df["track_id"]]
            te_t3 = [seqs_t3[str(t)] for t in test_df["track_id"]]
            tr_y = tr_df["label"].to_numpy(dtype=np.int64)
            va_y = va_df["label"].to_numpy(dtype=np.int64)
            te_y = test_df["label"].to_numpy(dtype=np.int64)

            for drop_branch in BRANCH_NAMES:
                kept_idx = {b: idxs for b, idxs in full_b_idx.items() if b != drop_branch}
                res = _train_eval_cogmucs_fold(
                    tr_seqs=tr_t3,
                    tr_labels=tr_y,
                    va_seqs=va_t3,
                    va_labels=va_y,
                    te_seqs=te_t3,
                    te_labels=te_y,
                    branch_indices=kept_idx,
                    branch_type=best_btype,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                m = res["metrics"]
                fold_records.append(
                    {
                        "model": best_model_name,
                        "tier": 3,
                        "ablation": f"drop_{drop_branch}",
                        "fold": fold_id,
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                    }
                )
        fold_df = pd.DataFrame(fold_records)

    fold_df.to_csv(out_path / "fold_metrics.csv", index=False)

    summary_df = (
        fold_df.groupby(["model", "tier", "ablation"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_folds=("fold", "count"),
        )
        .fillna(0.0)
        .sort_values(["ablation", "tier", "macro_f1_mean"], ascending=[True, True, False])
    )
    summary_df.to_csv(out_path / "summary_table.csv", index=False)

    # Save best fold-0 Tier-3 checkpoint & out-of-fold attention / confusion visualizations
    best_store = oof_store[best_model_name]
    if best_store["fold0_ckpt"]:
        torch.save(best_store["fold0_ckpt"][0], ckpt_dir / "best_tier3_fold0.pt")

    oof_true = np.concatenate(best_store["true"], axis=0)
    oof_pred = np.concatenate(best_store["pred"], axis=0)
    oof_bw = np.concatenate(best_store["bw"], axis=0)

    bw_rows = {
        "quadrant": [SHORT_QUADRANTS[int(q)] for q in oof_true],
        "predicted": [SHORT_QUADRANTS[int(p)] for p in oof_pred],
    }
    for b_idx, b_name in enumerate(BRANCH_NAMES):
        bw_rows[b_name] = oof_bw[:, b_idx]
    bw_df = pd.DataFrame(bw_rows)
    bw_df.to_csv(out_path / "branch_attention_weights.csv", index=False)

    _plot_tier_ablation(summary_df, out_path / "tier_ablation_plot.png")
    _plot_branch_weights(bw_df, best_model_name, out_path / "branch_weights_by_quadrant.png")
    _plot_confusion_matrix(
        oof_true,
        oof_pred,
        title=f"{best_model_name} (Tier 3) Out-of-Fold Confusion Matrix",
        out_path=out_path / "confusion_matrices.png",
    )

    # Markdown report comparing against June 2024 MUSEC & Panda 2020
    lines = [
        "# Coimbra 4Q Stage 1 Benchmark Report",
        "",
        f"- **Device:** `{device}` | **Tracks:** `{len(df)}` | **Folds:** `{len(folds)}` | **Epochs/fold:** `{epochs}`",
        f"- **Best Tier-3 Architecture:** `{best_model_name}`",
        (
            "- **Historical References:** June 2024 MUSEC Baseline = `39.84% Acc / 36.07% F1` | "
            "Panda et al. (2020) SVM Table 3 = `67.5% F1` (70 baseline) / `76.4% F1` (100 baseline+novel)"
        ),
        "",
        "## Model & Tier Summary",
        "",
        "| Model | Tier | Ablation | Accuracy (%) | Macro F1 (%) |",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['tier'])} | {row['ablation']} | "
            f"{row['accuracy_mean'] * 100:.2f} ± {row['accuracy_std'] * 100:.2f} | "
            f"{row['macro_f1_mean'] * 100:.2f} ± {row['macro_f1_std'] * 100:.2f} |"
        )
    lines.extend(["", "## Mean Cognitive Branch Attention by True Quadrant (Tier 3)", ""])
    q_table = bw_df.groupby("quadrant")[list(BRANCH_NAMES)].mean().reindex(SHORT_QUADRANTS)
    lines.append("| Quadrant | Melody | Rhythm | Timbre | Harmony |")
    lines.append("|---|---:|---:|---:|---:|")
    for q_name, q_row in q_table.iterrows():
        lines.append(
            f"| {q_name} | {q_row['melody']:.3f} | {q_row['rhythm']:.3f} | "
            f"{q_row['timbre']:.3f} | {q_row['harmony']:.3f} |"
        )
    (out_path / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10-fold Coimbra 4Q Stage 1 benchmark.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--splits-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/coimbra4q_stage1")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        processed_dir=args.processed_dir,
        splits_csv=args.splits_csv,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        skip_ablation=args.skip_ablation,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
