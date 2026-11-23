"""Fault-tolerant parallel CLI to extract and cache causal cognitive musicality features."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.coimbra4q import build_coimbra4q_splits, discover_coimbra4q
from data.deam import build_deam_splits, discover_deam
from features.pipeline import extract_file


def _extract_worker(task: tuple[str, str, str, int, float, str]) -> tuple[str, bool, str]:
    """Top-level picklable worker for safe multi-process extraction."""
    track_id, audio_path, out_file, tier, target_hz, backend = task
    try:
        extract_file(
            audio_path,
            tier=tier,
            target_hz=target_hz,
            backend=backend,
            cache_path=out_file,
        )
        return track_id, True, ""
    except Exception as exc:  # noqa: BLE001 - guard unattended batch against corrupt MP3s
        return track_id, False, repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract causal cognitive musicality features.")
    parser.add_argument("--dataset", choices=["coimbra4q", "deam"], required=True)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--backend", choices=["librosa", "essentia"], default="librosa")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--splits-out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_out = Path(args.splits_out or f"data/splits/{args.dataset}_splits.csv")

    if args.dataset == "coimbra4q":
        records = discover_coimbra4q(args.data_dir)
    else:
        records = discover_deam(args.data_dir)

    cache_root = Path(args.out_dir) / args.dataset / f"tier{args.tier}_{int(args.target_hz)}hz"
    cache_root.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str, int, float, str]] = []
    for rec in records:
        out_file = cache_root / f"{rec.track_id}.npz"
        if out_file.exists() and not args.overwrite:
            continue
        tasks.append(
            (
                rec.track_id,
                str(rec.audio_path),
                str(out_file),
                args.tier,
                args.target_hz,
                args.backend,
            )
        )

    failures: list[dict[str, str]] = []
    if tasks:
        if args.workers <= 1:
            for task in tqdm(tasks, desc=f"Extracting {args.dataset} tier={args.tier}"):
                tid, ok, err = _extract_worker(task)
                if not ok:
                    failures.append({"track_id": tid, "error": err})
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_extract_worker, task) for task in tasks]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Extracting {args.dataset} tier={args.tier} ({args.workers}w)",
                ):
                    tid, ok, err = fut.result()
                    if not ok:
                        failures.append({"track_id": tid, "error": err})

    failed_ids = {f["track_id"] for f in failures}
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_csv = cache_root / "failed_tracks.csv"
        fail_df.to_csv(fail_csv, index=False)
        print(f"WARNING: {len(failures)} tracks failed extraction; logged to {fail_csv}")

    valid_records = [r for r in records if r.track_id not in failed_ids]
    if args.dataset == "coimbra4q":
        build_coimbra4q_splits(valid_records, out_csv=splits_out)
    else:
        build_deam_splits(valid_records, out_csv=splits_out)

    print(
        f"Cached {len(valid_records)}/{len(records)} tracks under {cache_root}; "
        f"splits saved to {splits_out}"
    )


if __name__ == "__main__":
    main()
