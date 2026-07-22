from __future__ import annotations

from bisect import bisect_left, insort
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def maximum_rolling_mean(probabilities: np.ndarray, width: int) -> np.ndarray:
    """Return each window's largest contiguous rolling probability mean."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (windows, frames)")
    width = int(width)
    if width <= 0 or width > probabilities.shape[1]:
        raise ValueError(
            f"rolling width must be in [1, {probabilities.shape[1]}], got {width}"
        )

    cumulative = np.pad(
        np.cumsum(probabilities, axis=1),
        ((0, 0), (1, 0)),
        mode="constant",
    )
    rolling = (cumulative[:, width:] - cumulative[:, :-width]) / float(width)
    return rolling.max(axis=1)


def score_fully_negative_windows(
    model: Any,
    mmX: np.ndarray,
    mmY: np.ndarray,
    vids: np.ndarray,
    starts: np.ndarray,
    *,
    behavior_idx: int,
    rolling_frames: int,
    batch_size: int,
) -> pd.DataFrame:
    """Score label-negative training windows with sustained model confidence."""
    n_windows = int(mmX.shape[0])
    if int(mmY.shape[0]) != n_windows:
        raise ValueError("mmX/mmY window counts do not match")
    if len(vids) != n_windows or len(starts) != n_windows:
        raise ValueError("Window index arrays do not match the memmap count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= int(behavior_idx) < int(mmY.shape[-1]):
        raise ValueError(f"behavior_idx out of range: {behavior_idx}")

    labels = np.asarray(mmY[:, :, int(behavior_idx)], dtype=np.uint8)
    negative_indices = np.flatnonzero(labels.sum(axis=1) == 0).astype(np.int64)
    if len(negative_indices) == 0:
        raise ValueError("No fully negative training windows are available to screen")

    chunks: list[pd.DataFrame] = []
    for offset in range(0, len(negative_indices), int(batch_size)):
        indices = negative_indices[offset : offset + int(batch_size)]
        predictions = np.asarray(
            model.predict_on_batch(np.asarray(mmX[indices], dtype=np.float32))
        )
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions[:, :, 0]
        if predictions.shape != (len(indices), int(mmX.shape[1])):
            raise ValueError(
                "Model predictions must have shape "
                f"({len(indices)}, {mmX.shape[1]}, 1); got {predictions.shape}"
            )

        chunks.append(
            pd.DataFrame(
                {
                    "source_window_index": indices,
                    "stem": [str(vids[index]) for index in indices],
                    "start": [int(starts[index]) for index in indices],
                    "hardness": maximum_rolling_mean(
                        predictions, int(rolling_frames)
                    ),
                    "max_probability": predictions.max(axis=1),
                    "mean_probability": predictions.mean(axis=1),
                    "positive_frames": np.zeros(len(indices), dtype=np.int64),
                }
            )
        )

    scores = pd.concat(chunks, ignore_index=True)
    scores = scores.sort_values(
        ["hardness", "max_probability", "stem", "start"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scores.insert(0, "hardness_rank", np.arange(1, len(scores) + 1))
    return scores


def select_diverse_hard_negative_pool(
    scores: pd.DataFrame,
    *,
    pool_fraction: float,
    min_start_separation: int,
) -> pd.DataFrame:
    """Greedily retain high-scoring windows without near-duplicate starts."""
    required = {
        "source_window_index",
        "stem",
        "start",
        "hardness",
        "max_probability",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Hard-negative scores are missing columns: {missing}")
    pool_fraction = float(pool_fraction)
    if not 0.0 < pool_fraction <= 1.0:
        raise ValueError("pool_fraction must be in (0, 1]")
    min_start_separation = int(min_start_separation)
    if min_start_separation < 0:
        raise ValueError("min_start_separation cannot be negative")
    if scores.empty:
        raise ValueError("Hard-negative scores cannot be empty")

    ordered = scores.sort_values(
        ["hardness", "max_probability", "stem", "start"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    target = int(ceil(len(ordered) * pool_fraction))
    starts_by_stem: dict[str, list[int]] = {}
    selected_rows: list[int] = []

    for row_index, row in ordered.iterrows():
        stem = str(row["stem"])
        start = int(row["start"])
        selected_starts = starts_by_stem.setdefault(stem, [])
        position = bisect_left(selected_starts, start)
        neighbors = []
        if position > 0:
            neighbors.append(selected_starts[position - 1])
        if position < len(selected_starts):
            neighbors.append(selected_starts[position])
        if any(abs(start - other) < min_start_separation for other in neighbors):
            continue

        selected_rows.append(row_index)
        insort(selected_starts, start)
        if len(selected_rows) >= target:
            break

    pool = ordered.loc[selected_rows].copy().reset_index(drop=True)
    pool.insert(0, "pool_rank", np.arange(1, len(pool) + 1))
    return pool


def match_hard_negative_pool(
    pool_path: Path,
    vids: np.ndarray,
    starts: np.ndarray,
    mmY: np.ndarray,
    *,
    behavior_idx: int,
) -> np.ndarray:
    """Match an audited pool to current training windows by stem and start."""
    pool = pd.read_csv(pool_path)
    required = {"stem", "start"}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"Hard-negative pool is missing columns: {missing}")
    if pool.empty:
        raise ValueError(f"Hard-negative pool is empty: {pool_path}")

    current: dict[tuple[str, int], int] = {}
    for index, (stem, start) in enumerate(zip(vids, starts)):
        key = (str(stem), int(start))
        if key in current:
            raise ValueError(f"Duplicate training window identity: {key}")
        current[key] = index

    requested = [(str(row.stem), int(row.start)) for row in pool.itertuples()]
    if len(set(requested)) != len(requested):
        raise ValueError("Hard-negative pool contains duplicate stem/start rows")
    unmatched = [key for key in requested if key not in current]
    if unmatched:
        raise ValueError(
            f"Hard-negative pool is incompatible with this training split; "
            f"{len(unmatched)} windows were not found. Examples: {unmatched[:5]}"
        )

    indices = np.asarray([current[key] for key in requested], dtype=np.int64)
    labels = np.asarray(mmY[indices, :, int(behavior_idx)], dtype=np.uint8)
    invalid = np.flatnonzero(labels.sum(axis=1) != 0)
    if len(invalid):
        examples = [requested[int(index)] for index in invalid[:5]]
        raise ValueError(
            "Hard-negative pool contains windows with positive labels in the current "
            f"training data. Examples: {examples}"
        )
    return indices
