"""CLI to extract and cache causal cognitive musicality features."""

import argparse
from pathlib import Path

from tqdm import tqdm

from data.coimbra4q import build_coimbra4q_splits, discover_coimbra4q
from data.deam import build_deam_splits, discover_deam
from features.pipeline import extract_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract causal cognitive musicality features.")
    parser.add_argument("--dataset", choices=["coimbra4q", "deam"], required=True)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--backend", choices=["librosa", "essentia"], default="librosa")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "coimbra4q":
        records = discover_coimbra4q(args.data_dir)
        build_coimbra4q_splits(records, out_csv="data/splits/coimbra4q_splits.csv")
    else:
        records = discover_deam(args.data_dir)
        build_deam_splits(records, out_csv="data/splits/deam_splits.csv")

    cache_root = Path(args.out_dir) / args.dataset / f"tier{args.tier}_{int(args.target_hz)}hz"
    cache_root.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(records, desc=f"Extracting {args.dataset} tier={args.tier}"):
        out_file = cache_root / f"{rec.track_id}.npz"
        if out_file.exists() and not args.overwrite:
            continue
        extract_file(
            rec.audio_path,
            tier=args.tier,
            target_hz=args.target_hz,
            backend=args.backend,
            cache_path=out_file,
        )
    print(f"Cached {len(records)} tracks under {cache_root}")


if __name__ == "__main__":
    main()
