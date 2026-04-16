#!/usr/bin/env python3
"""
Evaluate one or more single-behavior KineLearn models from training manifests.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from KineLearn.core.manifests import (
    load_prediction_source,
    load_train_manifest,
    recusal_stems,
    resolve_recorded_path,
    resolve_weights_path,
    validate_train_manifests,
)
from KineLearn.core.models import build_sequence_model

try:
    import tensorflow as tf
except Exception:
    tf = None

# Path: src/KineLearn/scripts/eval.py


def open_memmap(artifact: dict, key: str, manifest_path: Path) -> np.memmap:
    path = resolve_recorded_path(artifact[f"{key}_path"], manifest_path)
    dtype = artifact[f"{key}_dtype"]
    shape = tuple(int(x) for x in artifact[f"{key}_shape"])
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def load_subset_arrays(
    manifest: dict,
    manifest_path: Path,
    subset: str,
) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray]:
    artifact = manifest["artifacts"][subset]
    mmX = open_memmap(artifact, "X", manifest_path)
    mmY = open_memmap(artifact, "Y", manifest_path)
    vids = np.load(resolve_recorded_path(artifact["vids_path"], manifest_path), allow_pickle=True)
    starts = np.load(resolve_recorded_path(artifact["starts_path"], manifest_path), allow_pickle=True)
    if len(vids) != int(artifact["count"]) or len(starts) != int(artifact["count"]):
        raise ValueError(f"Index array length mismatch for subset '{subset}'.")
    return mmX, mmY, vids, starts


def build_loaded_model(manifest: dict, weights_path: Path) -> "tf.keras.Model":
    if tf is None:
        raise ImportError("TensorFlow is required for evaluation.")
    window_size = int(manifest["window"]["size"])
    input_dim = int(manifest["feature_selection"]["n_input_features"])
    model = build_sequence_model(
        window_size,
        input_dim,
        model_cfg=(manifest.get("training") or {}).get("model"),
    )
    model.load_weights(str(weights_path))
    return model


def validate_prediction_sources_against_eval_manifest(
    prediction_sources: list[dict[str, Any]],
    eval_manifest: dict[str, Any],
) -> None:
    if not prediction_sources:
        raise ValueError("At least one prediction source is required.")

    behaviors = [source["behavior"] for source in prediction_sources]
    dupes = sorted({b for b in behaviors if behaviors.count(b) > 1})
    if dupes:
        raise ValueError(f"Duplicate behaviors in evaluation set: {dupes}")

    eval_training = eval_manifest.get("training") or {}
    eval_signature = {
        "kl_config": eval_manifest.get("kl_config"),
        "label_columns": list(eval_manifest["label_columns"]),
        "feature_columns": list(eval_manifest["feature_columns"]),
        "window": dict(eval_manifest["window"]),
        "feature_selection": dict(eval_manifest["feature_selection"]),
        "final_zero_fill": bool(eval_training.get("final_zero_fill", False)),
    }
    for source in prediction_sources:
        source_signature = {
            "kl_config": source.get("kl_config"),
            "label_columns": list(source["label_columns"]),
            "feature_columns": list(source["feature_columns"]),
            "window": dict(source["window"]),
            "feature_selection": dict(source["feature_selection"]),
            "final_zero_fill": bool((source.get("training") or {}).get("final_zero_fill", False)),
        }
        if source_signature != eval_signature:
            raise ValueError(
                f"Prediction source for behavior '{source['behavior']}' is incompatible "
                "with the evaluation manifest."
            )


def prepare_frame_buffers(
    vids: np.ndarray, starts: np.ndarray, window_size: int
) -> dict[str, dict[str, np.ndarray]]:
    per_stem_max = defaultdict(int)
    for vid, start in zip(vids, starts):
        stem = str(vid)
        per_stem_max[stem] = max(per_stem_max[stem], int(start) + window_size)

    buffers: dict[str, dict[str, np.ndarray]] = {}
    for stem, n_frames in per_stem_max.items():
        buffers[stem] = {
            "prob_sum": np.zeros(n_frames, dtype=np.float64),
            "count": np.zeros(n_frames, dtype=np.int32),
            "true": np.zeros(n_frames, dtype=np.uint8),
        }
    return buffers


def populate_true_labels(
    buffers: dict[str, dict[str, np.ndarray]],
    mmY: np.memmap,
    vids: np.ndarray,
    starts: np.ndarray,
    *,
    behavior_idx: int,
    window_size: int,
    batch_size: int,
) -> None:
    n = int(mmY.shape[0])

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        yb = np.asarray(mmY[start_idx:end_idx, :, behavior_idx], dtype=np.uint8)
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            stem = str(vids[global_idx])
            frame_start = int(starts[global_idx])
            frame_end = frame_start + window_size
            buffers[stem]["true"][frame_start:frame_end] = np.maximum(
                buffers[stem]["true"][frame_start:frame_end], yb[local_idx]
            )


def aggregate_member_predictions(
    model: "tf.keras.Model",
    mmX: np.memmap,
    vids: np.ndarray,
    starts: np.ndarray,
    buffers: dict[str, dict[str, np.ndarray]],
    *,
    window_size: int,
    batch_size: int,
    excluded_stems: set[str] | None = None,
) -> None:
    excluded_stems = excluded_stems or set()
    n = int(mmX.shape[0])

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        Xb = np.asarray(mmX[start_idx:end_idx], dtype=np.float32)
        pred = np.asarray(model.predict_on_batch(Xb), dtype=np.float32)[..., 0]

        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            stem = str(vids[global_idx])
            if stem in excluded_stems:
                continue
            frame_start = int(starts[global_idx])
            frame_end = frame_start + window_size
            buf = buffers[stem]
            buf["prob_sum"][frame_start:frame_end] += pred[local_idx]
            buf["count"][frame_start:frame_end] += 1


def frame_table_from_buffers(
    buffers: dict[str, dict[str, np.ndarray]],
    *,
    behavior: str,
    threshold: float,
) -> pd.DataFrame:
    parts = []
    for stem in sorted(buffers):
        buf = buffers[stem]
        counts = buf["count"]
        valid = counts > 0
        if not np.any(valid):
            continue
        frame_idx = np.flatnonzero(valid)
        probs = buf["prob_sum"][valid] / counts[valid]
        truth = buf["true"][valid].astype(np.uint8)
        pred = (probs >= threshold).astype(np.uint8)
        parts.append(
            pd.DataFrame(
                {
                    "__stem__": stem,
                    "__frame__": frame_idx.astype(np.int32),
                    f"true_{behavior}": truth,
                    f"prob_{behavior}": probs.astype(np.float32),
                    f"pred_{behavior}": pred,
                }
            )
        )
    if not parts:
        raise ValueError(f"No frame predictions were reconstructed for behavior '{behavior}'.")
    return pd.concat(parts, ignore_index=True)


def coverage_summary_from_buffers(
    buffers: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    total_frames = int(sum(len(buf["true"]) for buf in buffers.values()))
    scored_frames = int(sum(np.count_nonzero(buf["count"] > 0) for buf in buffers.values()))
    total_videos = int(len(buffers))
    scored_videos = int(sum(bool(np.any(buf["count"] > 0)) for buf in buffers.values()))
    return {
        "n_total_frames": total_frames,
        "n_scored_frames": scored_frames,
        "frame_coverage": float(scored_frames / total_frames) if total_frames else 0.0,
        "n_total_videos": total_videos,
        "n_scored_videos": scored_videos,
        "n_fully_recused_videos": int(total_videos - scored_videos),
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def build_bouts_from_mask(
    mask: np.ndarray, *, min_length: int = 16, max_gap: int = 3
) -> list[tuple[int, int]]:
    bouts = []
    start = None
    gap = 0
    ones = 0

    for i, value in enumerate(mask.astype(np.uint8)):
        if value:
            if start is None:
                start, ones, gap = i, 1, 0
            else:
                ones += 1
                gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = i - gap
                if ones >= min_length:
                    bouts.append((start, end))
                start = None
                gap = 0
                ones = 0

    if start is not None:
        end = len(mask) - 1
        if ones >= min_length:
            bouts.append((start, end))

    return bouts


def compute_bout_level_metrics(
    pred_bouts: list[tuple[int, int]],
    gt_bouts: list[tuple[int, int]],
    *,
    overlap_threshold: float = 0.2,
) -> dict[str, Any]:
    tp = 0
    matched_gt = 0

    for pred_bout in pred_bouts:
        pred_len = pred_bout[1] - pred_bout[0] + 1
        matched = False
        for gt_bout in gt_bouts:
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched = True
        if matched:
            tp += 1

    fp = len(pred_bouts) - tp

    for gt_bout in gt_bouts:
        matched = False
        for pred_bout in pred_bouts:
            pred_len = pred_bout[1] - pred_bout[0] + 1
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched = True
                break
        if matched:
            matched_gt += 1

    fn = len(gt_bouts) - matched_gt
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def identify_bout_errors(
    pred_bouts: list[tuple[int, int]],
    gt_bouts: list[tuple[int, int]],
    *,
    overlap_threshold: float = 0.2,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    matched_pred = set()
    matched_gt = set()

    for i, pred_bout in enumerate(pred_bouts):
        pred_len = pred_bout[1] - pred_bout[0] + 1
        for j, gt_bout in enumerate(gt_bouts):
            overlap = max(
                0, min(pred_bout[1], gt_bout[1]) - max(pred_bout[0], gt_bout[0]) + 1
            )
            if overlap >= overlap_threshold * pred_len:
                matched_pred.add(i)
                matched_gt.add(j)

    fp_bouts = [pred_bouts[i] for i in range(len(pred_bouts)) if i not in matched_pred]
    fn_bouts = [gt_bouts[j] for j in range(len(gt_bouts)) if j not in matched_gt]
    return fp_bouts, fn_bouts


def compute_episode_outputs(
    frame_df: pd.DataFrame,
    *,
    behavior: str,
    min_pred_frames: int,
    max_gap: int,
    overlap_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows = []
    error_rows = []

    for stem, group in frame_df.groupby("__stem__", sort=True):
        pred_mask = group[f"pred_{behavior}"].to_numpy(dtype=np.uint8)
        true_mask = group[f"true_{behavior}"].to_numpy(dtype=np.uint8)

        pred_bouts = build_bouts_from_mask(
            pred_mask, min_length=min_pred_frames, max_gap=max_gap
        )
        gt_bouts = build_bouts_from_mask(true_mask, min_length=1, max_gap=0)

        metrics = compute_bout_level_metrics(
            pred_bouts, gt_bouts, overlap_threshold=overlap_threshold
        )
        metric_rows.append(metrics)

        fp_bouts, fn_bouts = identify_bout_errors(
            pred_bouts, gt_bouts, overlap_threshold=overlap_threshold
        )
        for start, end in fp_bouts:
            error_rows.append(
                {
                    "__stem__": stem,
                    "behavior": behavior,
                    "level": "episode",
                    "error_type": "false_positive",
                    "start_frame": int(group["__frame__"].iloc[start]),
                    "end_frame": int(group["__frame__"].iloc[end]),
                }
            )
        for start, end in fn_bouts:
            error_rows.append(
                {
                    "__stem__": stem,
                    "behavior": behavior,
                    "level": "episode",
                    "error_type": "false_negative",
                    "start_frame": int(group["__frame__"].iloc[start]),
                    "end_frame": int(group["__frame__"].iloc[end]),
                }
            )

    total_tp = int(sum(row["tp"] for row in metric_rows))
    total_fp = int(sum(row["fp"] for row in metric_rows))
    total_fn = int(sum(row["fn"] for row in metric_rows))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    episode_metrics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_predicted_episodes": int(total_tp + total_fp),
        "n_true_episodes": int(total_tp + total_fn),
    }
    return episode_metrics, error_rows


def evaluate_manifest(
    manifest: dict,
    manifest_path: Path,
    *,
    subset: str,
    threshold: float,
    batch_size: int | None,
    level: str,
    episode_min_frames: int,
    episode_max_gap: int,
    episode_overlap_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    behavior = manifest["behavior"]
    behavior_idx = int(manifest["behavior_idx"])
    window_size = int(manifest["window"]["size"])
    weights_path = resolve_weights_path(manifest, manifest_path)
    model = build_loaded_model(manifest, weights_path)
    mmX, mmY, vids, starts = load_subset_arrays(manifest, manifest_path, subset)

    eval_batch_size = batch_size or int(manifest.get("training", {}).get("batch_size", 8))
    buffers = prepare_frame_buffers(vids, starts, window_size)
    populate_true_labels(
        buffers,
        mmY,
        vids,
        starts,
        behavior_idx=behavior_idx,
        window_size=window_size,
        batch_size=eval_batch_size,
    )
    aggregate_member_predictions(
        model,
        mmX,
        vids,
        starts,
        buffers,
        window_size=window_size,
        batch_size=eval_batch_size,
    )
    frame_df = frame_table_from_buffers(buffers, behavior=behavior, threshold=threshold)
    coverage = coverage_summary_from_buffers(buffers)
    metric_rows = []
    error_rows: list[dict[str, Any]] = []

    if level in {"frame", "both"}:
        frame_metrics = compute_binary_metrics(
            frame_df[f"true_{behavior}"].to_numpy(),
            frame_df[f"pred_{behavior}"].to_numpy(),
        )
        frame_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "frame",
                "threshold": float(threshold),
                "n_frames": int(len(frame_df)),
                "n_positive_frames": int(frame_df[f"true_{behavior}"].sum()),
                "manifest_path": str(manifest_path.resolve()),
                "weights_path": str(weights_path.resolve()),
                "manifest_kind": "train",
                "recusal_policy": "none",
                "ensemble_n_members": 1,
                **coverage,
            }
        )
        metric_rows.append(frame_metrics)

    if level in {"episode", "both"}:
        episode_metrics, episode_errors = compute_episode_outputs(
            frame_df,
            behavior=behavior,
            min_pred_frames=episode_min_frames,
            max_gap=episode_max_gap,
            overlap_threshold=episode_overlap_threshold,
        )
        episode_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "episode",
                "threshold": float(threshold),
                "episode_min_frames": int(episode_min_frames),
                "episode_max_gap": int(episode_max_gap),
                "episode_overlap_threshold": float(episode_overlap_threshold),
                "manifest_path": str(manifest_path.resolve()),
                "weights_path": str(weights_path.resolve()),
                "manifest_kind": "train",
                "recusal_policy": "none",
                "ensemble_n_members": 1,
                **coverage,
            }
        )
        metric_rows.append(episode_metrics)
        error_rows.extend(episode_errors)

    return frame_df, metric_rows, error_rows


def evaluate_prediction_source(
    source: dict[str, Any],
    eval_manifest: dict,
    eval_manifest_path: Path,
    *,
    subset: str,
    threshold: float,
    batch_size: int | None,
    level: str,
    episode_min_frames: int,
    episode_max_gap: int,
    episode_overlap_threshold: float,
    ensemble_recusal_policy: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    behavior = source["behavior"]
    behavior_idx = int(source["behavior_idx"])
    window_size = int(source["window"]["size"])
    mmX, mmY, vids, starts = load_subset_arrays(eval_manifest, eval_manifest_path, subset)

    eval_batch_size = batch_size or int(eval_manifest.get("training", {}).get("batch_size", 8))
    buffers = prepare_frame_buffers(vids, starts, window_size)
    populate_true_labels(
        buffers,
        mmY,
        vids,
        starts,
        behavior_idx=behavior_idx,
        window_size=window_size,
        batch_size=eval_batch_size,
    )

    for member in source["members"]:
        excluded_stems: set[str] = set()
        if source["manifest_kind"] == "ensemble" and ensemble_recusal_policy != "none":
            excluded_stems = recusal_stems(
                member["manifest"],
                policy=ensemble_recusal_policy,
            )
        model = build_loaded_model(member["manifest"], member["weights_path"])
        aggregate_member_predictions(
            model,
            mmX,
            vids,
            starts,
            buffers,
            window_size=window_size,
            batch_size=eval_batch_size,
            excluded_stems=excluded_stems,
        )

    frame_df = frame_table_from_buffers(buffers, behavior=behavior, threshold=threshold)
    coverage = coverage_summary_from_buffers(buffers)
    metric_rows = []
    error_rows: list[dict[str, Any]] = []

    if level in {"frame", "both"}:
        frame_metrics = compute_binary_metrics(
            frame_df[f"true_{behavior}"].to_numpy(),
            frame_df[f"pred_{behavior}"].to_numpy(),
        )
        frame_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "frame",
                "threshold": float(threshold),
                "n_frames": int(len(frame_df)),
                "n_positive_frames": int(frame_df[f"true_{behavior}"].sum()),
                "manifest_path": str(source["manifest_path"]),
                "weights_path": None,
                "manifest_kind": str(source["manifest_kind"]),
                "evaluation_manifest_path": str(eval_manifest_path.resolve()),
                "recusal_policy": (
                    ensemble_recusal_policy if source["manifest_kind"] == "ensemble" else "none"
                ),
                "ensemble_n_members": int(source["aggregation"]["n_members"]),
                **coverage,
            }
        )
        metric_rows.append(frame_metrics)

    if level in {"episode", "both"}:
        episode_metrics, episode_errors = compute_episode_outputs(
            frame_df,
            behavior=behavior,
            min_pred_frames=episode_min_frames,
            max_gap=episode_max_gap,
            overlap_threshold=episode_overlap_threshold,
        )
        episode_metrics.update(
            {
                "behavior": behavior,
                "subset": subset,
                "level": "episode",
                "threshold": float(threshold),
                "episode_min_frames": int(episode_min_frames),
                "episode_max_gap": int(episode_max_gap),
                "episode_overlap_threshold": float(episode_overlap_threshold),
                "manifest_path": str(source["manifest_path"]),
                "weights_path": None,
                "manifest_kind": str(source["manifest_kind"]),
                "evaluation_manifest_path": str(eval_manifest_path.resolve()),
                "recusal_policy": (
                    ensemble_recusal_policy if source["manifest_kind"] == "ensemble" else "none"
                ),
                "ensemble_n_members": int(source["aggregation"]["n_members"]),
                **coverage,
            }
        )
        metric_rows.append(episode_metrics)
        error_rows.extend(episode_errors)

    return frame_df, metric_rows, error_rows


def merge_behavior_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame_df in frames[1:]:
        merged = merged.merge(frame_df, on=["__stem__", "__frame__"], how="outer")
    merged = merged.sort_values(["__stem__", "__frame__"]).reset_index(drop=True)
    return merged


def build_summary(
    *,
    source_paths: list[Path],
    metrics_rows: list[dict[str, Any]],
    eval_manifest: dict | None,
    eval_manifest_path: Path | None,
    subset: str,
    level: str,
    threshold: float,
    episode_min_frames: int,
    episode_max_gap: int,
    episode_overlap_threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    frame_rows = [row for row in metrics_rows if row["level"] == "frame"]
    episode_rows = [row for row in metrics_rows if row["level"] == "episode"]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subset": subset,
        "level": level,
        "threshold": float(threshold),
        "episode_settings": {
            "min_pred_frames": int(episode_min_frames),
            "max_gap": int(episode_max_gap),
            "overlap_threshold": float(episode_overlap_threshold),
        },
        "out_dir": str(out_dir.resolve()),
        "kl_config": eval_manifest.get("kl_config") if eval_manifest is not None else None,
        "split": eval_manifest.get("split") if eval_manifest is not None else None,
        "manifests": [str(path.resolve()) for path in source_paths],
        "evaluation_manifest": (
            str(eval_manifest_path.resolve()) if eval_manifest_path is not None else None
        ),
        "behaviors": sorted({row["behavior"] for row in metrics_rows}),
        "frame_level_metrics": (
            {
                "macro_precision": float(np.mean([row["precision"] for row in frame_rows])),
                "macro_recall": float(np.mean([row["recall"] for row in frame_rows])),
                "macro_f1": float(np.mean([row["f1"] for row in frame_rows])),
                "macro_accuracy": float(np.mean([row["accuracy"] for row in frame_rows])),
            }
            if frame_rows
            else None
        ),
        "episode_level_metrics": (
            {
                "macro_precision": float(np.mean([row["precision"] for row in episode_rows])),
                "macro_recall": float(np.mean([row["recall"] for row in episode_rows])),
                "macro_f1": float(np.mean([row["f1"] for row in episode_rows])),
            }
            if episode_rows
            else None
        ),
        "per_behavior": metrics_rows,
    }


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "evaluations" / timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more KineLearn single-behavior models."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help=(
            "Path to a train_manifest.yml or ensemble_manifest.yml file. "
            "Provide once per behavior source."
        ),
    )
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help=(
            "Optional train_manifest.yml whose subset artifacts define the evaluation "
            "dataset and labels."
        ),
    )
    parser.add_argument(
        "--ensemble-recusal-policy",
        choices=["none", "train", "train_val"],
        default="train_val",
        help=(
            "Which member-owned stems should cause abstention during ensemble evaluation "
            "(default: train_val). Applies only to ensemble sources."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["train", "val", "test"],
        default="test",
        help="Which dataset subset to evaluate (default: test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for frame-level binary predictions (default: 0.5).",
    )
    parser.add_argument(
        "--level",
        choices=["frame", "episode", "both"],
        default="frame",
        help="Evaluation level to report (default: frame).",
    )
    parser.add_argument(
        "--episode-min-frames",
        type=int,
        default=16,
        help="Minimum positive frames required to keep a predicted episode (default: 16).",
    )
    parser.add_argument(
        "--episode-max-gap",
        type=int,
        default=3,
        help="Maximum internal gap of negative frames allowed within a predicted episode (default: 3).",
    )
    parser.add_argument(
        "--episode-overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap fraction of a predicted episode required to match ground truth (default: 0.2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override. Defaults to each manifest's training batch size.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for evaluation artifacts. Defaults to results/evaluations/<timestamp>/",
    )
    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if not (0.0 < args.episode_overlap_threshold <= 1.0):
        raise ValueError("--episode-overlap-threshold must be in (0, 1].")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.episode_min_frames <= 0:
        raise ValueError("--episode-min-frames must be positive.")
    if args.episode_max_gap < 0:
        raise ValueError("--episode-max-gap must be non-negative.")

    source_paths = [Path(p) for p in args.manifest]
    eval_manifest_path = Path(args.eval_manifest) if args.eval_manifest else None

    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_tables = []
    metrics_rows = []
    error_rows = []
    if eval_manifest_path is None:
        manifest_paths = source_paths
        manifests = [load_train_manifest(path) for path in manifest_paths]
        validate_train_manifests(manifests, args.subset)

        for manifest, manifest_path in zip(manifests, manifest_paths):
            frame_df, manifest_metrics, manifest_errors = evaluate_manifest(
                manifest,
                manifest_path,
                subset=args.subset,
                threshold=float(args.threshold),
                batch_size=args.batch_size,
                level=args.level,
                episode_min_frames=args.episode_min_frames,
                episode_max_gap=args.episode_max_gap,
                episode_overlap_threshold=float(args.episode_overlap_threshold),
            )
            frame_tables.append(frame_df)
            metrics_rows.extend(manifest_metrics)
            error_rows.extend(manifest_errors)
            for metrics in manifest_metrics:
                print(
                    f"[{metrics['behavior']}:{metrics['level']}] "
                    f"precision={metrics['precision']:.4f} "
                    f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
                )
        summary_eval_manifest = manifests[0]
        summary_eval_manifest_path = None
    else:
        eval_manifest = load_train_manifest(eval_manifest_path)
        prediction_sources = [load_prediction_source(path) for path in source_paths]
        validate_prediction_sources_against_eval_manifest(prediction_sources, eval_manifest)

        for source in prediction_sources:
            frame_df, source_metrics, source_errors = evaluate_prediction_source(
                source,
                eval_manifest,
                eval_manifest_path,
                subset=args.subset,
                threshold=float(args.threshold),
                batch_size=args.batch_size,
                level=args.level,
                episode_min_frames=args.episode_min_frames,
                episode_max_gap=args.episode_max_gap,
                episode_overlap_threshold=float(args.episode_overlap_threshold),
                ensemble_recusal_policy=args.ensemble_recusal_policy,
            )
            frame_tables.append(frame_df)
            metrics_rows.extend(source_metrics)
            error_rows.extend(source_errors)
            for metrics in source_metrics:
                print(
                    f"[{metrics['behavior']}:{metrics['level']}] "
                    f"precision={metrics['precision']:.4f} "
                    f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
                    f"(coverage={metrics['frame_coverage']:.4f})"
                )
        summary_eval_manifest = eval_manifest
        summary_eval_manifest_path = eval_manifest_path

    merged_frames = merge_behavior_frames(frame_tables)
    metrics_df = pd.DataFrame(metrics_rows)
    errors_df = pd.DataFrame(error_rows)
    summary = build_summary(
        source_paths=source_paths,
        metrics_rows=metrics_rows,
        eval_manifest=summary_eval_manifest,
        eval_manifest_path=summary_eval_manifest_path,
        subset=args.subset,
        level=args.level,
        threshold=float(args.threshold),
        episode_min_frames=args.episode_min_frames,
        episode_max_gap=args.episode_max_gap,
        episode_overlap_threshold=float(args.episode_overlap_threshold),
        out_dir=out_dir,
    )

    summary_path = out_dir / "eval_summary.yml"
    metrics_path = out_dir / "per_behavior_metrics.csv"
    frames_path = out_dir / "frame_predictions.parquet"
    errors_path = out_dir / "episode_errors.csv"

    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    metrics_df.to_csv(metrics_path, index=False)
    merged_frames.to_parquet(frames_path, index=False)
    if args.level in {"episode", "both"}:
        errors_df.to_csv(errors_path, index=False)

    print(f"\n📝 Wrote {summary_path}")
    print(f"📝 Wrote {metrics_path}")
    print(f"📝 Wrote {frames_path}")
    if args.level in {"episode", "both"}:
        print(f"📝 Wrote {errors_path}")


if __name__ == "__main__":
    main()
