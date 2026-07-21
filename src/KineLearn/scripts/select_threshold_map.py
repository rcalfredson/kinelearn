#!/usr/bin/env python3
"""Select one validation-derived threshold per split run from a metric sweep."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from KineLearn.scripts.batch_eval_splits import file_sha256


RANKING_COLUMNS = [
    "f1",
    "min_precision_recall",
    "precision",
    "recall",
    "threshold_closest_to_0.5",
    "lower_threshold",
]


def normalize_run_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "outer_id" not in frame and "outer_split" in frame:
        frame["outer_id"] = frame["outer_split"].astype(str)
    if "inner_seed" not in frame and "inner_split" in frame:
        frame["inner_seed"] = (
            frame["inner_split"].astype(str).str.replace("inner_seed", "", regex=False)
        )
    required = {"outer_id", "inner_seed", "threshold", "f1", "precision", "recall"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Threshold-sweep metrics are missing columns: {missing}")
    frame["outer_id"] = frame["outer_id"].astype(str)
    frame["inner_seed"] = frame["inner_seed"].astype(str)
    for column in ("threshold", "f1", "precision", "recall"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def select_threshold_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_run_columns(frame)
    frame["min_precision_recall"] = frame[["precision", "recall"]].min(axis=1)
    frame["threshold_distance"] = (frame["threshold"] - 0.5).abs()
    ranked = frame.sort_values(
        [
            "outer_id",
            "inner_seed",
            "f1",
            "min_precision_recall",
            "precision",
            "recall",
            "threshold_distance",
            "threshold",
        ],
        ascending=[True, True, False, False, False, False, True, True],
    )
    selected = ranked.groupby(["outer_id", "inner_seed"], sort=True).head(1).copy()
    selected = selected.rename(
        columns={
            "f1": "validation_f1",
            "precision": "validation_precision",
            "recall": "validation_recall",
        }
    )
    columns = [
        "outer_id",
        "inner_seed",
        "threshold",
        "validation_f1",
        "validation_precision",
        "validation_recall",
    ]
    if "behavior" in selected:
        columns.insert(2, "behavior")
    return selected[columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one validation-derived threshold per split run."
    )
    parser.add_argument("metrics", help="CSV containing per-run threshold-sweep metrics.")
    parser.add_argument("--behavior", default=None, help="Optional behavior filter.")
    parser.add_argument(
        "--level",
        default="episode",
        help="Metric level to select when the input has a level column (default: episode).",
    )
    parser.add_argument(
        "--episode-matching-method",
        required=True,
        help="Episode matcher used to produce the threshold-sweep metrics.",
    )
    parser.add_argument(
        "--episode-overlap-denominator",
        required=True,
        help="Overlap denominator used to produce the threshold-sweep metrics.",
    )
    parser.add_argument("--out", required=True, help="Output threshold-map CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    frame = pd.read_csv(metrics_path)
    if args.behavior is not None:
        if "behavior" not in frame:
            raise ValueError("--behavior was provided but the metrics have no behavior column.")
        frame = frame[frame["behavior"] == args.behavior]
    if "level" in frame:
        frame = frame[frame["level"] == args.level]
    if frame.empty:
        raise ValueError("No threshold-sweep rows remain after applying filters.")
    if "behavior" in frame and frame["behavior"].nunique() > 1:
        raise ValueError(
            "Threshold-sweep metrics contain multiple behaviors; provide --behavior."
        )

    selected = select_threshold_rows(frame)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_metrics": str(metrics_path.resolve()),
        "source_sha256": file_sha256(metrics_path),
        "behavior": args.behavior,
        "level": args.level,
        "n_runs": int(len(selected)),
        "ranking": RANKING_COLUMNS,
        "episode_matching_method": args.episode_matching_method,
        "episode_overlap_denominator": args.episode_overlap_denominator,
        "threshold_map": str(out_path.resolve()),
    }
    metadata_path = out_path.with_suffix(".yml")
    with metadata_path.open("w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"📝 Wrote {out_path}")
    print(f"📝 Wrote {metadata_path}")


if __name__ == "__main__":
    main()
