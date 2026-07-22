#!/usr/bin/env python3
"""Screen a trained run's negative training windows for episode-like errors."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np

from KineLearn.core.hard_negatives import (
    score_fully_negative_windows,
    select_diverse_hard_negative_pool,
)
from KineLearn.core.manifests import (
    load_train_manifest,
    resolve_recorded_path,
    resolve_weights_path,
    save_yaml,
)
from KineLearn.core.models import build_sequence_model

try:
    import tensorflow as tf
except Exception:
    tf = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score fully negative training windows with a saved model and write a "
            "fixed, overlap-diversified hard-negative pool."
        )
    )
    parser.add_argument("--manifest", required=True, help="Source train_manifest.yml.")
    parser.add_argument(
        "--rolling-frames",
        type=int,
        default=None,
        help=(
            "Contiguous probability interval used for hardness. Defaults to the "
            "run's checkpoint-selection episode_min_frames setting."
        ),
    )
    parser.add_argument(
        "--pool-fraction",
        type=float,
        default=0.10,
        help="Fraction of fully negative candidates to retain (default: 0.10).",
    )
    parser.add_argument(
        "--min-start-separation",
        type=int,
        default=None,
        help=(
            "Minimum frame-start separation between selected windows from one video. "
            "Defaults to the training window size."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size override.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/hard_negative_screens/<timestamp>/.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_training_artifacts(
    manifest: dict, manifest_path: Path
) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray]:
    artifact = manifest["artifacts"]["train"]
    mmX = np.memmap(
        resolve_recorded_path(artifact["X_path"], manifest_path),
        mode="r",
        dtype=artifact["X_dtype"],
        shape=tuple(int(value) for value in artifact["X_shape"]),
    )
    mmY = np.memmap(
        resolve_recorded_path(artifact["Y_path"], manifest_path),
        mode="r",
        dtype=artifact["Y_dtype"],
        shape=tuple(int(value) for value in artifact["Y_shape"]),
    )
    vids = np.load(
        resolve_recorded_path(artifact["vids_path"], manifest_path),
        allow_pickle=True,
    )
    starts = np.load(
        resolve_recorded_path(artifact["starts_path"], manifest_path),
        allow_pickle=True,
    )
    return mmX, mmY, vids, starts


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "hard_negative_screens" / timestamp


def main() -> None:
    args = parse_args()
    if tf is None:
        raise ImportError("TensorFlow is required to screen hard negatives.")

    manifest_path = Path(args.manifest).resolve()
    manifest = load_train_manifest(manifest_path)
    window_size = int(manifest["window"]["size"])
    selection_cfg = (manifest.get("training") or {}).get("checkpoint_selection") or {}
    rolling_frames = int(
        args.rolling_frames
        if args.rolling_frames is not None
        else selection_cfg.get("episode_min_frames", 16)
    )
    min_start_separation = int(
        args.min_start_separation
        if args.min_start_separation is not None
        else window_size
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else (manifest.get("training") or {}).get("inference_batch_size", 128)
    )
    if rolling_frames <= 0 or rolling_frames > window_size:
        raise ValueError(
            f"--rolling-frames must be in [1, {window_size}], got {rolling_frames}"
        )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    weights_path = resolve_weights_path(manifest, manifest_path)
    model = build_sequence_model(
        window_size,
        int(manifest["feature_selection"]["n_input_features"]),
        model_cfg=(manifest.get("training") or {}).get("model"),
    )
    model.load_weights(str(weights_path))
    mmX, mmY, vids, starts = open_training_artifacts(manifest, manifest_path)

    print(
        f"Screening {len(mmX)} training windows for {manifest['behavior']} "
        f"with a {rolling_frames}-frame rolling score..."
    )
    scores = score_fully_negative_windows(
        model,
        mmX,
        mmY,
        vids,
        starts,
        behavior_idx=int(manifest["behavior_idx"]),
        rolling_frames=rolling_frames,
        batch_size=batch_size,
    )
    pool = select_diverse_hard_negative_pool(
        scores,
        pool_fraction=float(args.pool_fraction),
        min_start_separation=min_start_separation,
    )

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "hard_negative_scores.csv"
    pool_path = out_dir / "hard_negative_pool.csv"
    scores = scores.copy()
    selected_indices = set(pool["source_window_index"].astype(int))
    scores["selected_for_pool"] = scores["source_window_index"].isin(selected_indices)
    scores.to_csv(scores_path, index=False)
    pool.to_csv(pool_path, index=False)

    summary = {
        "screen_format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_weights": str(weights_path.resolve()),
        "source_weights_sha256": file_sha256(weights_path),
        "behavior": manifest["behavior"],
        "behavior_idx": int(manifest["behavior_idx"]),
        "subset": "train",
        "window_size": window_size,
        "rolling_frames": rolling_frames,
        "hardness_method": "maximum_contiguous_rolling_mean_probability",
        "pool_fraction": float(args.pool_fraction),
        "min_start_separation": min_start_separation,
        "inference_batch_size": batch_size,
        "n_training_windows": int(len(mmX)),
        "n_fully_negative_candidates": int(len(scores)),
        "n_selected_hard_negatives": int(len(pool)),
        "candidate_hardness": {
            "min": float(scores["hardness"].min()),
            "median": float(scores["hardness"].median()),
            "max": float(scores["hardness"].max()),
        },
        "selected_hardness": {
            "min": float(pool["hardness"].min()),
            "median": float(pool["hardness"].median()),
            "max": float(pool["hardness"].max()),
        },
        "scores_csv": str(scores_path.resolve()),
        "scores_sha256": file_sha256(scores_path),
        "pool_csv": str(pool_path.resolve()),
        "pool_sha256": file_sha256(pool_path),
    }
    save_yaml(out_dir / "hard_negative_screen.yml", summary)

    print(f"Selected {len(pool)} of {len(scores)} fully negative windows.")
    print(f"\n📝 Wrote {scores_path}")
    print(f"📝 Wrote {pool_path}")
    print(f"📝 Wrote {out_dir / 'hard_negative_screen.yml'}")


if __name__ == "__main__":
    main()
