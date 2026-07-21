#!/usr/bin/env python3
"""
Train a single-behavior KineLearn classifier from precomputed frame features.

This script:
- loads the KineLearn config and a saved train/test split
- derives a validation subset from the training stems
- loads per-video feature and label Parquet files
- optionally excludes raw absolute x/y keypoint columns from model input
- windows train/val/test subsets into memmap-backed arrays
- trains a keypoints-only BiLSTM with focal loss
- checkpoints on val_loss or opt-in reconstructed episode validation performance
- evaluates the selected checkpoint on the test subset
- writes run artifacts and a train_manifest.yml under results/<behavior>/<timestamp>/

Training is single-behavior per run. Focal-loss alpha can be specified in the config
per behavior or overridden at the CLI for split-specific validation tuning.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from KineLearn.core.features import select_behavior_feature_columns
from KineLearn.core.generators import KeypointWindowGenerator
from KineLearn.core.losses import focal_loss
from KineLearn.core.memmap import make_windowed_memmaps
from KineLearn.core.models import build_sequence_model
from KineLearn.scripts.eval import (
    aggregate_member_predictions,
    compute_episode_outputs,
    frame_table_from_buffers,
    populate_true_labels,
    prepare_frame_buffers,
)

# (Optional for future training step)
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
except Exception:
    tf = K = None

# Path: src/KineLearn/scripts/train.py


# ----------------------------
# Helpers
# ----------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split_file(path: Path) -> dict:
    """
    Load either:
    - current YAML split files with keys like train/test
    - legacy plain-text files with sections like 'Train videos:' / 'Test videos:'
    """
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    sections: dict[str, list[str]] = {}
    current_key = None
    for line in lines:
        if line.endswith(":"):
            current_key = line[:-1].strip().lower()
            sections[current_key] = []
            continue
        if current_key is None:
            raise ValueError(f"Malformed split file {path}: found entries before a section header.")
        sections[current_key].append(line)
    return sections


def split_section(d: dict, *names: str) -> list[str]:
    """
    Return the first matching split section from a normalized split dict.
    """
    lowered = {str(k).strip().lower(): v for k, v in d.items()}
    for name in names:
        key = name.strip().lower()
        if key in lowered:
            return lowered[key]
    raise ValueError(f"Missing split section; tried {names}, available={list(lowered.keys())}")


def ensure_disjoint(*, name_a: str, stems_a: List[str], name_b: str, stems_b: List[str]) -> None:
    overlap = sorted(set(stems_a) & set(stems_b))
    if overlap:
        preview = overlap[:10]
        suffix = " ..." if len(overlap) > 10 else ""
        raise ValueError(
            f"{name_a} and {name_b} overlap on {len(overlap)} stems: {preview}{suffix}"
        )


def require_keys(d: dict, keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys {missing} in {where}")


def available_feature_stems(features_dir: Path) -> list[str]:
    prefix = "frame_features_"
    suffix = ".parquet"
    stems = []
    for path in sorted(features_dir.glob(f"{prefix}*{suffix}")):
        name = path.name
        stems.append(name[len(prefix) : -len(suffix)])
    return stems


def resolve_requested_stems(
    requested: List[str], available: List[str], *, where: str
) -> List[str]:
    """
    Resolve split-file identifiers against available feature stems.

    Supports either:
    - exact full stem matches
    - legacy short ids like 20250730_190518 that uniquely match by suffix
    """
    available_set = set(available)
    resolved: list[str] = []
    for stem in requested:
        if stem in available_set:
            resolved.append(stem)
            continue
        matches = [cand for cand in available if cand.endswith(stem)]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if not matches:
            raise ValueError(f"{where}: could not resolve stem '{stem}' against available feature files.")
        raise ValueError(
            f"{where}: stem '{stem}' matched multiple feature files: {matches}"
        )
    return resolved


def load_parquets_for_stems(
    stems: List[str],
    features_dir: Path,
    behaviors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate per-stem features/labels Parquet files.

    - Features: features/frame_features_<stem>.parquet
    - Labels:   features/frame_labels_<stem>.parquet
    - If a label file is missing, create a zero-filled frame with the given behaviors.
    - Ensures columns for labels exactly match 'behaviors' (order preserved).
    """
    X_parts, y_parts = [], []
    for stem in stems:
        feat_path = features_dir / f"frame_features_{stem}.parquet"
        lab_path = features_dir / f"frame_labels_{stem}.parquet"

        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        X = pd.read_parquet(feat_path)
        if lab_path.exists():
            Y = pd.read_parquet(lab_path)
            # If label file has extra columns, reduce; if missing, add zeros.
            # Final label columns exactly match 'behaviors', in that order.
            for b in behaviors:
                if b not in Y.columns:
                    Y[b] = 0
            extra_cols = [c for c in Y.columns if c not in behaviors]
            if extra_cols:
                Y = Y.drop(columns=extra_cols, errors="ignore")
            Y = Y[behaviors]
        else:
            Y = pd.DataFrame(0, index=range(len(X)), columns=behaviors)

        if len(X) != len(Y):
            raise ValueError(
                f"Row mismatch for stem '{stem}': features={len(X)} vs labels={len(Y)}"
            )

        # (Optional) keep a stem column for tracing/debug
        X = X.copy()
        Y = Y.copy()
        X["__stem__"] = stem
        Y["__stem__"] = stem
        frame_idx = np.arange(len(X), dtype=np.int32)
        X["__frame__"] = frame_idx
        Y["__frame__"] = frame_idx

        X_parts.append(X)
        y_parts.append(Y)

    X_all = pd.concat(X_parts, axis=0, ignore_index=True)
    Y_all = pd.concat(y_parts, axis=0, ignore_index=True)
    return X_all, Y_all


def summarize_dataset(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    behaviors: List[str],
) -> None:
    print("\n=== Dataset summary ===")
    print(f"Train: X={X_train.shape}, Y={y_train.shape}")
    print(f" Test: X={X_test.shape}, Y={y_test.shape}")

    # Quick label support summary (counts of positive frames) per behavior
    def label_counts(df: pd.DataFrame) -> pd.Series:
        # df contains behavior columns + __stem__
        cols = [c for c in df.columns if c in behaviors]
        return df[cols].sum().astype(int)

    train_pos = label_counts(y_train)
    test_pos = label_counts(y_test)

    print("\nPositive frame counts (train):")
    for b in behaviors:
        print(f"  {b}: {train_pos.get(b, 0)}")

    print("\nPositive frame counts (test):")
    for b in behaviors:
        print(f"  {b}: {test_pos.get(b, 0)}")

    # Show a peek of feature columns (excluding helper column)
    feature_cols = [c for c in X_train.columns if c not in ("__stem__", "__frame__")]
    print(f"\nTotal feature columns: {len(feature_cols)} (showing first 10)")
    print(feature_cols[:10])


def resolve_focal_params(training_cfg: Dict, behavior: str) -> tuple[float, float]:
    """
    Resolve focal loss alpha and gamma.
    training_cfg may contain:
      training["focal"] = {"alpha": <float or {behavior: float}, "gamma": <float>}
    Falls back to alpha=0.7, gamma=2.0 if unspecified.
    """
    focal = training_cfg.get("focal", {}) or {}
    alpha_cfg = focal.get("alpha", 0.7)
    gamma = float(focal.get("gamma", 2.0))
    if isinstance(alpha_cfg, dict):
        if behavior not in alpha_cfg:
            raise ValueError(
                f"No focal.alpha specified for behavior '{behavior}'. Available: {list(alpha_cfg.keys())}"
            )
        alpha = float(alpha_cfg[behavior])
    else:
        alpha = float(alpha_cfg)
    return alpha, gamma


def resolve_keypoint_noise_std(training_cfg: Dict, behavior: str) -> float:
    """
    Resolve training-time keypoint noise std.
    training["keypoint_noise_std"] may be either:
      - a single float applied to all behaviors
      - a mapping from behavior name to float
    Falls back to 0.0 if unspecified.
    """
    noise_cfg = training_cfg.get("keypoint_noise_std", 0.0)
    if isinstance(noise_cfg, dict):
        if behavior not in noise_cfg:
            raise ValueError(
                "No training.keypoint_noise_std specified for behavior "
                f"'{behavior}'. Available: {list(noise_cfg.keys())}"
            )
        return float(noise_cfg[behavior])
    return float(noise_cfg)


def _positive_integer_setting(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"training.{name} must be a positive integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"training.{name} must be a positive integer.")
    return resolved


def resolve_execution_settings(training_cfg: Dict) -> tuple[int, int, int]:
    """Resolve optimizer, compiled-loop, and inference batch settings."""
    batch_size = _positive_integer_setting(
        training_cfg.get("batch_size", 8), "batch_size"
    )
    steps_per_execution = _positive_integer_setting(
        training_cfg.get("steps_per_execution", 1), "steps_per_execution"
    )
    inference_batch_size = _positive_integer_setting(
        training_cfg.get("inference_batch_size", batch_size),
        "inference_batch_size",
    )
    return batch_size, steps_per_execution, inference_batch_size


def checkpoint_thresholds(selection_cfg: dict[str, Any]) -> list[float]:
    """Resolve an explicit, finite probability-threshold grid."""
    threshold_cfg = selection_cfg.get("thresholds")
    if threshold_cfg is None:
        threshold_cfg = {"start": 0.35, "stop": 0.75, "step": 0.01}

    if isinstance(threshold_cfg, (list, tuple)):
        thresholds = [float(value) for value in threshold_cfg]
    elif isinstance(threshold_cfg, dict):
        start = float(threshold_cfg.get("start", 0.35))
        stop = float(threshold_cfg.get("stop", 0.75))
        step = float(threshold_cfg.get("step", 0.01))
        if step <= 0:
            raise ValueError("training.checkpoint_selection.thresholds.step must be positive.")
        if stop < start:
            raise ValueError(
                "training.checkpoint_selection.thresholds.stop must be >= start."
            )
        count = int(np.floor((stop - start) / step + 1e-9)) + 1
        thresholds = [start + idx * step for idx in range(count)]
        if thresholds[-1] < stop - 1e-9:
            thresholds.append(stop)
    else:
        raise ValueError(
            "training.checkpoint_selection.thresholds must be a list or mapping."
        )

    thresholds = sorted({round(float(value), 10) for value in thresholds})
    if not thresholds:
        raise ValueError("training.checkpoint_selection.thresholds cannot be empty.")
    if any(
        not np.isfinite(value) or value <= 0.0 or value >= 1.0
        for value in thresholds
    ):
        raise ValueError(
            "Checkpoint-selection thresholds must be finite and between 0 and 1."
        )
    return thresholds


def resolve_checkpoint_selection_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize opt-in episode-aware checkpoint selection settings."""
    raw_cfg = training_cfg.get("checkpoint_selection") or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("training.checkpoint_selection must be a mapping.")

    cfg = dict(raw_cfg)
    cfg.setdefault("enabled", False)
    cfg["enabled"] = bool(cfg["enabled"])
    cfg.setdefault("metric", "episode_f1")
    if cfg["metric"] != "episode_f1":
        raise ValueError(
            "Only training.checkpoint_selection.metric: episode_f1 is currently supported."
        )
    cfg["thresholds"] = checkpoint_thresholds(cfg)
    cfg.setdefault("episode_min_frames", 16)
    cfg.setdefault("episode_max_gap", 3)
    cfg.setdefault("episode_overlap_threshold", 0.2)
    cfg["episode_min_frames"] = int(cfg["episode_min_frames"])
    cfg["episode_max_gap"] = int(cfg["episode_max_gap"])
    cfg["episode_overlap_threshold"] = float(cfg["episode_overlap_threshold"])
    if cfg["episode_min_frames"] <= 0:
        raise ValueError("checkpoint_selection.episode_min_frames must be positive.")
    if cfg["episode_max_gap"] < 0:
        raise ValueError("checkpoint_selection.episode_max_gap cannot be negative.")
    if not 0.0 < cfg["episode_overlap_threshold"] <= 1.0:
        raise ValueError(
            "checkpoint_selection.episode_overlap_threshold must be in (0, 1]."
        )
    return cfg


def checkpoint_candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Rank by episode F1, then balance, with deterministic threshold tie-breaks."""
    precision = float(candidate["precision"])
    recall = float(candidate["recall"])
    threshold = float(candidate["threshold"])
    return (
        float(candidate["f1"]),
        min(precision, recall),
        precision,
        recall,
        -abs(threshold - 0.5),
        -threshold,
    )


def select_checkpoint_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one checkpoint candidate is required.")
    return max(candidates, key=checkpoint_candidate_rank)


def align_columns(
    df: pd.DataFrame,
    expected: List[str],
    *,
    df_name: str,
    helper_columns: tuple[str, ...] = (),
    allow_extra: bool = False,
) -> pd.DataFrame:
    """
    Reorder a DataFrame to a known column order and fail loudly on mismatch.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing expected columns: {missing}")

    extra = [c for c in df.columns if c not in expected and c not in helper_columns]
    if extra and not allow_extra:
        raise ValueError(f"{df_name} has unexpected columns: {extra}")

    ordered = list(expected) + [c for c in helper_columns if c in df.columns]
    return df.loc[:, ordered]


def zero_fill_remaining_nans(
    df: pd.DataFrame, *, df_name: str, helper_columns: tuple[str, ...] = ()
) -> pd.DataFrame:
    """
    Replace any remaining NaNs with zeros, excluding helper columns from reporting.
    """
    value_columns = [c for c in df.columns if c not in helper_columns]
    nan_count = int(df[value_columns].isna().sum().sum())
    if nan_count > 0:
        print(f"⚠️  Final zero-fill parity step on {df_name}: replacing {nan_count} NaNs.")
        df = df.copy()
        df[value_columns] = df[value_columns].fillna(0)
    return df


def is_absolute_coordinate_column(col: str) -> bool:
    """
    Return True for raw absolute x/y keypoint columns such as `thorax_x`.
    Derived feature columns are excluded.
    """
    if not (col.endswith("_x") or col.endswith("_y")):
        return False

    derived_markers = ("_coord_", "_velocity_", "_acceleration_")
    if any(marker in col for marker in derived_markers):
        return False

    return True


class HistoryCapture(tf.keras.callbacks.Callback if tf is not None else object):
    """
    Lightweight callback that retains epoch-end logs even if training is interrupted.
    """

    def __init__(self):
        if tf is None:
            raise ImportError("TensorFlow is required for HistoryCapture.")
        super().__init__()
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))


class EpisodeCheckpointSelector(tf.keras.callbacks.Callback if tf is not None else object):
    """Select weights using reconstructed validation episode performance."""

    def __init__(
        self,
        *,
        mmX: np.memmap,
        mmY: np.memmap,
        vids: np.ndarray,
        starts: np.ndarray,
        behavior: str,
        behavior_idx: int,
        window_size: int,
        batch_size: int,
        selection_cfg: dict[str, Any],
        checkpoint_path: Path,
        output_dir: Path,
    ) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required for EpisodeCheckpointSelector.")
        super().__init__()
        self.mmX = mmX
        self.vids = vids
        self.starts = starts
        self.behavior = behavior
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.selection_cfg = dict(selection_cfg)
        self.thresholds = list(selection_cfg["thresholds"])
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.candidates_path = output_dir / "checkpoint_candidates.csv"
        self.summary_path = output_dir / "checkpoint_selection_summary.yml"
        self.predictions_path = (
            output_dir / "checkpoint_selection_val_predictions.parquet"
        )
        self._base_buffers = prepare_frame_buffers(vids, starts, self.window_size)
        populate_true_labels(
            self._base_buffers,
            mmY,
            vids,
            starts,
            behavior_idx=int(behavior_idx),
            window_size=self.window_size,
            batch_size=self.batch_size,
        )
        self.candidate_rows: list[dict[str, Any]] = []
        self.best_candidate: dict[str, Any] | None = None
        self.best_frame_df: pd.DataFrame | None = None

    def _fresh_buffers(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            stem: {
                "prob_sum": np.zeros_like(buf["prob_sum"]),
                "count": np.zeros_like(buf["count"]),
                "true": buf["true"].copy(),
            }
            for stem, buf in self._base_buffers.items()
        }

    def _write_artifacts(self) -> None:
        best_epoch = self.best_candidate["epoch"] if self.best_candidate else None
        best_threshold = self.best_candidate["threshold"] if self.best_candidate else None
        rows = []
        for row in self.candidate_rows:
            output_row = dict(row)
            output_row["selected_overall_so_far"] = bool(
                row["epoch"] == best_epoch and row["threshold"] == best_threshold
            )
            rows.append(output_row)
        pd.DataFrame(rows).to_csv(self.candidates_path, index=False)

        summary = {
            "enabled": True,
            "metric": self.selection_cfg["metric"],
            "ranking": [
                "episode_f1",
                "min_episode_precision_recall",
                "episode_precision",
                "episode_recall",
                "threshold_closest_to_0.5",
                "lower_threshold",
            ],
            "thresholds": self.thresholds,
            "episode_min_frames": self.selection_cfg["episode_min_frames"],
            "episode_max_gap": self.selection_cfg["episode_max_gap"],
            "episode_overlap_threshold": self.selection_cfg[
                "episode_overlap_threshold"
            ],
            "selected": dict(self.best_candidate) if self.best_candidate else None,
            "candidates_csv": str(self.candidates_path.resolve()),
            "validation_predictions": (
                str(self.predictions_path.resolve())
                if self.best_frame_df is not None
                else None
            ),
        }
        with open(self.summary_path, "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        if self.best_frame_df is not None:
            self.best_frame_df.to_parquet(self.predictions_path, index=False)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        buffers = self._fresh_buffers()
        aggregate_member_predictions(
            self.model,
            self.mmX,
            self.vids,
            self.starts,
            buffers,
            window_size=self.window_size,
            batch_size=self.batch_size,
        )
        frame_df = frame_table_from_buffers(
            buffers, behavior=self.behavior, threshold=self.thresholds[0]
        )
        probability_column = f"prob_{self.behavior}"
        prediction_column = f"pred_{self.behavior}"
        probabilities = frame_df[probability_column].to_numpy()
        epoch_candidates = []
        for threshold in self.thresholds:
            frame_df[prediction_column] = (probabilities >= threshold).astype(np.uint8)
            metrics, _error_rows = compute_episode_outputs(
                frame_df,
                behavior=self.behavior,
                min_pred_frames=self.selection_cfg["episode_min_frames"],
                max_gap=self.selection_cfg["episode_max_gap"],
                overlap_threshold=self.selection_cfg["episode_overlap_threshold"],
            )
            candidate = {
                "epoch": int(epoch) + 1,
                "threshold": float(threshold),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "min_precision_recall": float(
                    min(metrics["precision"], metrics["recall"])
                ),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "n_predicted_episodes": int(metrics["n_predicted_episodes"]),
                "n_true_episodes": int(metrics["n_true_episodes"]),
            }
            epoch_candidates.append(candidate)

        epoch_best = select_checkpoint_candidate(epoch_candidates)
        for candidate in epoch_candidates:
            candidate["selected_for_epoch"] = candidate is epoch_best
        self.candidate_rows.extend(epoch_candidates)

        if self.best_candidate is None or checkpoint_candidate_rank(
            epoch_best
        ) > checkpoint_candidate_rank(self.best_candidate):
            candidate_path = self.output_dir / ".best_model.candidate.weights.h5"
            self.model.save_weights(str(candidate_path))
            candidate_path.replace(self.checkpoint_path)
            self.best_candidate = dict(epoch_best)
            frame_df[prediction_column] = (
                probabilities >= float(epoch_best["threshold"])
            ).astype(np.uint8)
            self.best_frame_df = frame_df.copy()

        logs["val_selected_episode_f1"] = float(epoch_best["f1"])
        logs["val_selected_episode_precision"] = float(epoch_best["precision"])
        logs["val_selected_episode_recall"] = float(epoch_best["recall"])
        logs["val_selected_threshold"] = float(epoch_best["threshold"])
        self._write_artifacts()
        print(
            "\nEpisode checkpoint candidate: "
            f"epoch={int(epoch) + 1} threshold={epoch_best['threshold']:.3f} "
            f"F1={epoch_best['f1']:.4f} precision={epoch_best['precision']:.4f} "
            f"recall={epoch_best['recall']:.4f}"
        )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KineLearn training (Step 1): load hyperparameters and dataset."
    )
    parser.add_argument(
        "--kl-config",
        required=True,
        help="Path to KineLearn config YAML.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Path to train/test split file. Supports current YAML produced by "
            "kinelearn-split or legacy plain-text files with 'Train videos:' / 'Test videos:'."
        ),
    )
    parser.add_argument(
        "--val-split",
        default=None,
        help=(
            "Optional explicit train/val split file. Supports legacy plain-text files "
            "with 'Train videos:' / 'Val   videos:' sections. If omitted, validation is "
            "derived from the training stems using training.val_fraction."
        ),
    )
    parser.add_argument(
        "--behavior",
        required=True,
        help="Behavior name to train (must be present in the KineLearn config's `behaviors` list).",
    )
    parser.add_argument(
        "--features-dir",
        default="features",
        help="Directory containing frame_features_*.parquet and frame_labels_*.parquet (default: features).",
    )
    # Optional quick overrides (useful even before a full training section exists)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config (optional).",
    )
    parser.add_argument(
        "--steps-per-execution",
        type=int,
        default=None,
        help="Override training.steps_per_execution from config (optional).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help=(
            "Override training.inference_batch_size for validation, checkpoint "
            "selection, and final test evaluation (optional)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the training seed for this run. "
            "This affects the train/validation split and training-time shuffling."
        ),
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help=(
            "Override focal loss alpha for this training run. "
            "Useful for split-specific tuning in single-behavior training."
        ),
    )
    parser.add_argument(
        "--keypoint-noise-std",
        type=float,
        default=None,
        help=(
            "Override training-time Gaussian keypoint noise std for this run. "
            "Validation and test windows remain noise-free."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional run output directory. Defaults to results/<behavior>/<timestamp>/."
        ),
    )

    args = parser.parse_args()

    kl_config = load_yaml(Path(args.kl_config))
    require_keys(kl_config, ["behaviors"], "KineLearn config")
    behaviors: List[str] = kl_config["behaviors"] or []
    if not isinstance(behaviors, list) or not all(
        isinstance(b, str) for b in behaviors
    ):
        raise ValueError("`behaviors` in KineLearn config must be a list of strings.")

    behavior = args.behavior
    if behavior not in behaviors:
        raise ValueError(
            f"--behavior '{behavior}' not found in config behaviors: {behaviors}"
        )

    # Optional training hyperparameters nested under config["training"]
    training_cfg: Dict = kl_config.get("training") or {}
    # Allow CLI overrides
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.steps_per_execution is not None:
        training_cfg["steps_per_execution"] = args.steps_per_execution
    if args.inference_batch_size is not None:
        training_cfg["inference_batch_size"] = args.inference_batch_size
    if args.seed is not None:
        training_cfg["seed"] = int(args.seed)
    if args.focal_alpha is not None:
        training_cfg.setdefault("focal", {})
        training_cfg["focal"]["alpha"] = float(args.focal_alpha)
    if args.keypoint_noise_std is not None:
        training_cfg["keypoint_noise_std"] = float(args.keypoint_noise_std)

    # Provide some sane defaults (used later when we add training)
    training_cfg.setdefault("epochs", 10)
    training_cfg.setdefault("batch_size", 8)
    training_cfg.setdefault("steps_per_execution", 1)
    training_cfg.setdefault("inference_batch_size", training_cfg["batch_size"])
    training_cfg.setdefault("learning_rate", 1e-3)
    training_cfg.setdefault(
        "loss", "focal"
    )  # focal is the only supported loss initially
    training_cfg.setdefault("metrics", ["accuracy"])
    training_cfg.setdefault("val_fraction", 0.1)  # split from training later
    training_cfg.setdefault("seed", 42)
    training_cfg.setdefault("include_absolute_coordinates", False)
    training_cfg.setdefault("early_stopping", False)
    training_cfg.setdefault("early_stopping_patience", 3)
    training_cfg.setdefault("early_stopping_min_delta", 0.0)
    training_cfg.setdefault("keypoint_noise_std", 0.0)
    training_cfg.setdefault("model", {})
    training_cfg["model"].setdefault("variant", "bilstm")
    training_cfg.setdefault("final_zero_fill", False)
    checkpoint_selection_cfg = resolve_checkpoint_selection_config(training_cfg)
    training_cfg["checkpoint_selection"] = checkpoint_selection_cfg
    batch_size, steps_per_execution, inference_batch_size = (
        resolve_execution_settings(training_cfg)
    )
    training_cfg["batch_size"] = batch_size
    training_cfg["steps_per_execution"] = steps_per_execution
    training_cfg["inference_batch_size"] = inference_batch_size

    # Resolve focal params (alpha can be global or per-behavior)
    alpha, gamma = resolve_focal_params(training_cfg, behavior)
    noise_std = resolve_keypoint_noise_std(training_cfg, behavior)

    print("Loaded training hyperparameters:")
    for k in [
        "epochs",
        "batch_size",
        "steps_per_execution",
        "inference_batch_size",
        "learning_rate",
        "loss",
        "metrics",
        "val_fraction",
        "seed",
        "include_absolute_coordinates",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "keypoint_noise_std",
        "final_zero_fill",
        "checkpoint_selection",
    ]:
        print(f"  {k}: {training_cfg[k]}")
    if training_cfg.get("loss", "focal") == "focal":
        print(f"  focal.alpha({behavior}): {alpha}")
        print(f"  focal.gamma: {gamma}")
    print(f"  keypoint_noise_std({behavior}): {noise_std}")
    print(f"  model.variant: {training_cfg['model']['variant']}")

    features_dir = Path(args.features_dir)
    known_stems = available_feature_stems(features_dir)
    if not known_stems:
        raise FileNotFoundError(
            f"No frame_features_*.parquet files found in features directory: {features_dir}"
        )

    split_info = load_split_file(Path(args.split))
    split_train_stems = resolve_requested_stems(
        split_section(split_info, "train", "train videos"), known_stems, where="split.train"
    )
    test_stems = resolve_requested_stems(
        split_section(split_info, "test", "test videos"), known_stems, where="split.test"
    )
    ensure_disjoint(
        name_a="split.train",
        stems_a=split_train_stems,
        name_b="split.test",
        stems_b=test_stems,
    )

    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results") / behavior / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir

    X_test, y_test = load_parquets_for_stems(test_stems, features_dir, behaviors)

    wcfg = kl_config.get("window") or training_cfg.get("window") or {}
    wsize = int(wcfg.get("size", 60))
    stride = int(wcfg.get("stride", 10))

    if wsize <= 0 or stride <= 0:
        raise ValueError(
            f"window.size and window.stride must be positive; got size={wsize}, stride={stride}"
        )
    if stride > wsize:
        print(f"⚠️  stride ({stride}) > window_size ({wsize}); windows will be sparse.")

    # Avoid duplicating window inside training in the manifest
    training_cfg: dict = dict(training_cfg)
    training_cfg.pop("window", None)

    n_classes = len(behaviors)
    seed = training_cfg.get("seed", 42)

    if args.val_split:
        val_split_info = load_split_file(Path(args.val_split))
        train_stems = resolve_requested_stems(
            split_section(val_split_info, "train", "train videos"),
            known_stems,
            where="val_split.train",
        )
        val_stems = resolve_requested_stems(
            split_section(val_split_info, "val", "val videos", "val   videos"),
            known_stems,
            where="val_split.val",
        )
        ensure_disjoint(
            name_a="val_split.train",
            stems_a=train_stems,
            name_b="val_split.val",
            stems_b=val_stems,
        )
        ensure_disjoint(
            name_a="val_split.train",
            stems_a=train_stems,
            name_b="split.test",
            stems_b=test_stems,
        )
        ensure_disjoint(
            name_a="val_split.val",
            stems_a=val_stems,
            name_b="split.test",
            stems_b=test_stems,
        )
        explicit_train_val = sorted(set(train_stems) | set(val_stems))
        expected_train = sorted(set(split_train_stems))
        if explicit_train_val != expected_train:
            missing = sorted(set(expected_train) - set(explicit_train_val))
            extras = sorted(set(explicit_train_val) - set(expected_train))
            raise ValueError(
                "Explicit val split must partition the training stems from --split exactly. "
                f"Missing={missing[:10]}{' ...' if len(missing) > 10 else ''} "
                f"Extras={extras[:10]}{' ...' if len(extras) > 10 else ''}"
            )
        print(
            f"🧩 Using explicit validation split with {len(train_stems)} train and {len(val_stems)} validation videos."
        )
    else:
        # Split training stems into train/val sets
        train_stems, val_stems = train_test_split(
            split_train_stems, test_size=training_cfg["val_fraction"], random_state=seed
        )
        if not train_stems or not val_stems:
            raise ValueError(
                "Train/validation split produced an empty partition. "
                "Adjust training.val_fraction or provide more training videos."
            )
        print(
            f"🧩 Split {len(train_stems) + len(val_stems)} total training videos "
            f"into {len(train_stems)} train and {len(val_stems)} validation."
        )

    X_train, y_train = load_parquets_for_stems(train_stems, features_dir, behaviors)
    X_val, y_val = load_parquets_for_stems(val_stems, features_dir, behaviors)

    helper_columns = ("__stem__", "__frame__")
    if training_cfg.get("final_zero_fill", False):
        X_train = zero_fill_remaining_nans(
            X_train, df_name="X_train", helper_columns=helper_columns
        )
        X_val = zero_fill_remaining_nans(
            X_val, df_name="X_val", helper_columns=helper_columns
        )
        X_test = zero_fill_remaining_nans(
            X_test, df_name="X_test", helper_columns=helper_columns
        )
        y_train = zero_fill_remaining_nans(
            y_train, df_name="y_train", helper_columns=helper_columns
        )
        y_val = zero_fill_remaining_nans(
            y_val, df_name="y_val", helper_columns=helper_columns
        )
        y_test = zero_fill_remaining_nans(
            y_test, df_name="y_test", helper_columns=helper_columns
        )

    # Basic sanity check
    if any(dt == "object" for dt in X_train.dtypes):
        objs = [
            c
            for c in X_train.columns
            if X_train[c].dtype == "object" and c not in ("__stem__", "__frame__")
        ]
        if objs:
            raise TypeError(f"Found non-numeric feature columns: {objs}")

    summarize_dataset(X_train, y_train, X_test, y_test, behaviors)

    # Feature column order as written into memmaps (must be stable + recorded)
    all_feature_columns = [
        c for c in X_train.columns if c not in ("__stem__", "__frame__")
    ]
    behavior_feature_columns = select_behavior_feature_columns(
        all_feature_columns,
        kl_config.get("features") or {},
        behavior,
    )
    excluded_relational_columns = sorted(
        set(all_feature_columns) - set(behavior_feature_columns)
    )
    if excluded_relational_columns:
        print(
            f"Excluding {len(excluded_relational_columns)} relational feature columns "
            f"not assigned to behavior '{behavior}'."
        )
    all_feature_columns = behavior_feature_columns
    include_absolute_coordinates = bool(
        training_cfg["include_absolute_coordinates"]
    )
    if include_absolute_coordinates:
        feature_columns = list(all_feature_columns)
    else:
        feature_columns = [
            c for c in all_feature_columns if not is_absolute_coordinate_column(c)
        ]
        dropped_columns = [
            c for c in all_feature_columns if is_absolute_coordinate_column(c)
        ]
        print(
            f"Excluding {len(dropped_columns)} absolute coordinate columns from training input."
        )
    derived_dim = len(feature_columns)
    label_columns = list(behaviors)
    behavior_idx = label_columns.index(behavior)

    X_train = align_columns(
        X_train,
        feature_columns,
        df_name="X_train",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    X_val = align_columns(
        X_val,
        feature_columns,
        df_name="X_val",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    X_test = align_columns(
        X_test,
        feature_columns,
        df_name="X_test",
        helper_columns=helper_columns,
        allow_extra=True,
    )
    y_train = align_columns(
        y_train,
        label_columns,
        df_name="y_train",
        helper_columns=helper_columns,
    )
    y_val = align_columns(
        y_val,
        label_columns,
        df_name="y_val",
        helper_columns=helper_columns,
    )
    y_test = align_columns(
        y_test,
        label_columns,
        df_name="y_test",
        helper_columns=helper_columns,
    )

    split_positive_counts = {
        "train": int(y_train[behavior].sum()),
        "val": int(y_val[behavior].sum()),
        "test": int(y_test[behavior].sum()),
    }
    if split_positive_counts["train"] == 0:
        raise ValueError(
            f"Selected behavior '{behavior}' has zero positive frames in training data."
        )
    if split_positive_counts["val"] == 0:
        print(
            f"⚠️  Selected behavior '{behavior}' has zero positive frames in validation data."
        )
    if split_positive_counts["test"] == 0:
        print(
            f"⚠️  Selected behavior '{behavior}' has zero positive frames in test data."
        )

    train_count, mmX_tr, mmY_tr, tr_vids, tr_starts = make_windowed_memmaps(
        X_train,
        y_train,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "train"),
    )
    val_count, mmX_va, mmY_va, va_vids, va_starts = make_windowed_memmaps(
        X_val,
        y_val,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "val"),
    )
    test_count, mmX_te, mmY_te, te_vids, te_starts = make_windowed_memmaps(
        X_test,
        y_test,
        wsize,
        stride,
        derived_dim,
        n_classes,
        str(out / "test"),
    )
    for split_name, count in (
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ):
        if count == 0:
            raise ValueError(
                f"No {split_name} windows were created. "
                "Check window.size/window.stride and per-video frame counts."
            )

    # Persist index arrays (vids + starts) for traceability / later evaluation
    def _save_index(
        name: str, vids: np.ndarray, starts: np.ndarray
    ) -> tuple[Path, Path]:
        vids_path = out / f"{name}_vids.npy"
        starts_path = out / f"{name}_starts.npy"
        np.save(vids_path, vids)
        np.save(starts_path, starts)
        return vids_path, starts_path

    tr_vids_path, tr_starts_path = _save_index("train", tr_vids, tr_starts)
    va_vids_path, va_starts_path = _save_index("val", va_vids, va_starts)
    te_vids_path, te_starts_path = _save_index("test", te_vids, te_starts)

    # Record memmap + index paths explicitly in the manifest
    def _artifact_block(
        name: str, count: int, vids_path: Path, starts_path: Path
    ) -> dict:
        prefix = out / name
        X_path = (prefix.parent / f"{prefix.name}_features.fp32").resolve()
        Y_path = (prefix.parent / f"{prefix.name}_labels.u8").resolve()
        return {
            "count": int(count),
            "X_path": str(X_path),
            "Y_path": str(Y_path),
            "vids_path": str(vids_path.resolve()),
            "starts_path": str(starts_path.resolve()),
            "X_dtype": "float32",
            "Y_dtype": "uint8",
            "X_shape": [int(count), int(wsize), int(derived_dim)],
            "Y_shape": [int(count), int(wsize), int(n_classes)],
        }

    # Write training manifest
    manifest = {
        "kl_config": str(Path(args.kl_config).resolve()),
        "split": str(Path(args.split).resolve()),
        "val_split": str(Path(args.val_split).resolve()) if args.val_split else None,
        "features_dir": str(features_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "behaviors": behaviors,
        "label_columns": label_columns,
        "feature_columns": feature_columns,
        "training": training_cfg,
        "window": {"size": wsize, "stride": stride},
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "feature_selection": {
            "include_absolute_coordinates": include_absolute_coordinates,
            "n_input_features": len(feature_columns),
        },
        "positive_frames": split_positive_counts,
        "n_features": derived_dim,
        "n_classes": n_classes,
        "resolved_stems": {
            "train": list(train_stems),
            "val": list(val_stems),
            "test": list(test_stems),
        },
    }

    # Include resolved behavior + focal params in manifest for traceability
    manifest["behavior"] = behavior
    manifest["behavior_idx"] = int(behavior_idx)
    if training_cfg.get("loss", "focal") == "focal":
        manifest["focal"] = {"alpha": alpha, "gamma": gamma}
    manifest["keypoint_noise_std"] = noise_std

    manifest["artifacts"] = {
        "train": _artifact_block("train", train_count, tr_vids_path, tr_starts_path),
        "val": _artifact_block("val", val_count, va_vids_path, va_starts_path),
        "test": _artifact_block("test", test_count, te_vids_path, te_starts_path),
    }

    if tf is None:
        raise ImportError(
            "TensorFlow is required for training. "
            "Install tensorflow (or tensorflow-cpu) in this environment."
        )

    # ----------------------------
    # Generators (keypoints-only)
    # ----------------------------
    train_gen = KeypointWindowGenerator(
        mmX_tr,
        mmY_tr,
        behavior_idx=behavior_idx,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        noise_std=noise_std,
    )
    val_gen = KeypointWindowGenerator(
        mmX_va,
        mmY_va,
        behavior_idx=behavior_idx,
        batch_size=inference_batch_size,
        shuffle=False,
        seed=seed,
        noise_std=0.0,
    )
    test_gen = KeypointWindowGenerator(
        mmX_te,
        mmY_te,
        behavior_idx=behavior_idx,
        batch_size=inference_batch_size,
        shuffle=False,
        seed=seed,
        noise_std=0.0,
    )

    # ----------------------------
    # Model + compile
    # ----------------------------
    model = build_sequence_model(wsize, derived_dim, model_cfg=training_cfg.get("model"))

    lr = float(training_cfg["learning_rate"])
    if training_cfg.get("loss", "focal") != "focal":
        raise ValueError(
            f"Unsupported loss: {training_cfg.get('loss')} (only 'focal' supported)"
        )

    loss_fn = focal_loss(alpha=alpha, gamma=gamma)

    # Metrics: keep lightweight + stable
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=metrics,
        steps_per_execution=steps_per_execution,
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    ckpt_path = out / "best_model.weights.h5"
    interrupted_ckpt_path = out / "interrupted_model.weights.h5"
    history_capture = HistoryCapture()
    episode_selector = None
    if checkpoint_selection_cfg["enabled"]:
        episode_selector = EpisodeCheckpointSelector(
            mmX=mmX_va,
            mmY=mmY_va,
            vids=va_vids,
            starts=va_starts,
            behavior=behavior,
            behavior_idx=behavior_idx,
            window_size=wsize,
            batch_size=inference_batch_size,
            selection_cfg=checkpoint_selection_cfg,
            checkpoint_path=ckpt_path,
            output_dir=out,
        )
        callbacks = [episode_selector]
    else:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            )
        ]
    callbacks.extend(
        [
            tf.keras.callbacks.CSVLogger(str(out / "train_history.csv")),
            history_capture,
        ]
    )

    if training_cfg.get("reduce_lr", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )
        )
    if training_cfg.get("early_stopping", False):
        early_stopping_monitor = (
            "val_selected_episode_f1"
            if checkpoint_selection_cfg["enabled"]
            else "val_loss"
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                mode="max" if checkpoint_selection_cfg["enabled"] else "min",
                patience=int(training_cfg["early_stopping_patience"]),
                min_delta=float(training_cfg["early_stopping_min_delta"]),
                restore_best_weights=not checkpoint_selection_cfg["enabled"],
                verbose=1,
            )
        )

    # ----------------------------
    # Fit
    # ----------------------------
    epochs = int(training_cfg["epochs"])
    interrupted = False
    interruption_reason = None
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        history_data = {
            k: [float(x) for x in v] for k, v in (history.history or {}).items()
        }
    except KeyboardInterrupt:
        interrupted = True
        interruption_reason = "keyboard_interrupt"
        print("\n⚠️  Training interrupted; saving partial run artifacts.")
        model.save_weights(str(interrupted_ckpt_path))
        history_data = history_capture.history
        print(
            "⚠️  Leaving partial artifacts without train_manifest.yml so the run can be resumed cleanly."
        )
        raise SystemExit(130)

    val_loss_history = history_data.get("val_loss", [])
    best_epoch = (
        int(np.argmin(val_loss_history)) + 1 if val_loss_history else None
    )

    evaluation_ckpt_path = None
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))
        evaluation_ckpt_path = ckpt_path
    elif interrupted_ckpt_path.exists():
        model.load_weights(str(interrupted_ckpt_path))
        evaluation_ckpt_path = interrupted_ckpt_path
    else:
        print(
            "⚠️  No saved checkpoint weights were found; evaluating final in-memory model."
        )

    # ----------------------------
    # Evaluate
    # ----------------------------
    test_metrics = model.evaluate(test_gen, verbose=1)
    test_results = dict(zip(model.metrics_names, [float(x) for x in test_metrics]))
    print("\n=== Test metrics ===")
    for k, v in test_results.items():
        print(f"  {k}: {v}")

    manifest["training_run"] = {
        "interrupted": interrupted,
        "interruption_reason": interruption_reason,
        "checkpoint_best_model": str(ckpt_path.resolve()),
        "checkpoint_interrupted_model": (
            str(interrupted_ckpt_path.resolve())
            if interrupted_ckpt_path.exists()
            else None
        ),
        "evaluation_weights": (
            str(evaluation_ckpt_path.resolve()) if evaluation_ckpt_path else None
        ),
        "history_csv": str((out / "train_history.csv").resolve()),
        "epochs_completed": int(len(history_data.get("loss", []))),
        "best_epoch_by_val_loss": best_epoch,
        "checkpoint_selection": (
            {
                "enabled": True,
                "metric": checkpoint_selection_cfg["metric"],
                "selected": dict(episode_selector.best_candidate),
                "candidates_csv": str(episode_selector.candidates_path.resolve()),
                "summary_yml": str(episode_selector.summary_path.resolve()),
                "validation_predictions": str(
                    episode_selector.predictions_path.resolve()
                ),
            }
            if episode_selector is not None and episode_selector.best_candidate is not None
            else {"enabled": False}
        ),
        "final_metrics": {k: float(v[-1]) for k, v in history_data.items() if len(v) > 0},
        "test_metrics": test_results,
    }

    with open(out / "train_manifest.yml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"\n📝 Wrote {out / 'train_manifest.yml'}")


if __name__ == "__main__":
    main()
