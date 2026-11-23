"""Thin CLI entrypoint for training and evaluating CogMucs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data.alignment import collate_padded_sequences, make_deam_warmup_mask
from data.deam import load_deam_dynamic_va
from evaluation.metrics import compute_4q_metrics, compute_dynamic_va_metrics
from features.tiers import get_branch_indices
from models.cogmucs import CogMucs
from models.losses import CompositeDynamicVALoss, static_4q_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CogMucs on extracted feature caches.")
    parser.add_argument("--dataset", choices=["coimbra4q", "deam"], required=True)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--model", choices=["tcn", "gru"], default="tcn")
    parser.add_argument("--fusion", choices=["early", "mid", "late"], default="mid")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--splits-csv", type=str, default=None)
    parser.add_argument("--deam-valence-csv", type=str, default=None)
    parser.add_argument("--deam-arousal-csv", type=str, default=None)
    return parser.parse_args()


def _fit_transform_sequences(
    train_seqs: list[np.ndarray], eval_seqs: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_seqs, axis=0))
    return (
        [scaler.transform(s).astype(np.float32) for s in train_seqs],
        [scaler.transform(s).astype(np.float32) for s in eval_seqs],
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch_indices = get_branch_indices(args.tier)
    cache_root = Path(args.processed_dir) / args.dataset / f"tier{args.tier}_{int(args.target_hz)}hz"
    splits_path = Path(args.splits_csv or f"data/splits/{args.dataset}_splits.csv")
    df = pd.read_csv(splits_path)

    if args.dataset == "coimbra4q":
        train_df = df[df["fold"] != args.fold]
        val_df = df[df["fold"] == args.fold]
        tr_seqs = [np.load(cache_root / f"{tid}.npz")["features"] for tid in train_df["track_id"]]
        va_seqs = [np.load(cache_root / f"{tid}.npz")["features"] for tid in val_df["track_id"]]
        tr_seqs, va_seqs = _fit_transform_sequences(tr_seqs, va_seqs)
        tr_y = torch.tensor(train_df["label"].to_numpy(), dtype=torch.long)
        va_y = torch.tensor(val_df["label"].to_numpy(), dtype=torch.long)

        model = CogMucs(
            branch_indices=branch_indices,
            branch_type=args.model,
            fusion=args.fusion,
            task="static_4q",
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(tr_seqs))
            for i in range(0, len(tr_seqs), args.batch_size):
                idx = perm[i : i + args.batch_size].tolist()
                xb, mb = collate_padded_sequences([tr_seqs[j] for j in idx])
                yb = tr_y[idx].to(device)
                opt.zero_grad()
                out = model(xb.to(device), mask=mb.to(device))
                loss = static_4q_loss(out.pred, yb)
                loss.backward()
                opt.step()
            if epoch % 10 == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    xv, mv = collate_padded_sequences(va_seqs)
                    preds = model(xv.to(device), mask=mv.to(device)).pred.argmax(dim=-1).cpu().numpy()
                metrics = compute_4q_metrics(va_y.numpy(), preds)
                print(f"[Epoch {epoch:03d}] val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f}")
        return

    if not args.deam_valence_csv or not args.deam_arousal_csv:
        raise ValueError("Provide --deam-valence-csv and --deam-arousal-csv for DEAM training.")
    targets_map, label_times = load_deam_dynamic_va(args.deam_valence_csv, args.deam_arousal_csv)
    warmup_mask = torch.from_numpy(make_deam_warmup_mask(label_times, warmup_sec=15.0))

    train_ids = [str(t) for t in df[df["split"] == "train"]["track_id"] if str(t) in targets_map]
    val_ids = [str(t) for t in df[df["split"] == "val"]["track_id"] if str(t) in targets_map]
    tr_seqs = [np.load(cache_root / f"{tid}.npz")["features"] for tid in train_ids]
    va_seqs = [np.load(cache_root / f"{tid}.npz")["features"] for tid in val_ids]
    tr_seqs, va_seqs = _fit_transform_sequences(tr_seqs, va_seqs)
    tr_t = [targets_map[tid] for tid in train_ids]
    va_t = [targets_map[tid] for tid in val_ids]

    stride = max(1, round(args.target_hz / 2.0))
    model = CogMucs(
        branch_indices=branch_indices,
        branch_type=args.model,
        fusion=args.fusion,
        task="dynamic_va",
        dynamic_stride=stride,
    ).to(device)
    criterion = CompositeDynamicVALoss(mse_weight=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_seqs))
        for i in range(0, len(tr_seqs), args.batch_size):
            idx = perm[i : i + args.batch_size].tolist()
            xb, mb = collate_padded_sequences([tr_seqs[j] for j in idx])
            yb = torch.stack([torch.from_numpy(tr_t[j]) for j in idx], dim=0).to(device)
            opt.zero_grad()
            pred = model(xb.to(device), mask=mb.to(device)).pred[:, : yb.shape[1], :]
            t_mask = warmup_mask[: yb.shape[1]].unsqueeze(0).expand(yb.shape[0], -1).to(device)
            loss = criterion(pred, yb, mask=t_mask)
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                xv, mv = collate_padded_sequences(va_seqs)
                yv = torch.stack([torch.from_numpy(t) for t in va_t], dim=0)
                pv = model(xv.to(device), mask=mv.to(device)).pred[:, : yv.shape[1], :].cpu().numpy()
                vm = warmup_mask[: yv.shape[1]].unsqueeze(0).expand(yv.shape[0], -1).numpy()
            metrics = compute_dynamic_va_metrics(pv, yv.numpy(), mask=vm)
            print(
                f"[Epoch {epoch:03d}] val_ccc_V={metrics['valence_ccc_persong']:.4f} "
                f"val_ccc_A={metrics['arousal_ccc_persong']:.4f}"
            )


if __name__ == "__main__":
    main()
