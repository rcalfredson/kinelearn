#!/usr/bin/env python3
"""Compare compatible split-sweep evaluation batches with paired run deltas."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


RUN_KEYS = ["outer_id", "inner_seed", "behavior", "subset", "level"]
METRICS = ["f1", "precision", "recall"]


def parse_assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} must use LABEL=VALUE syntax: {value!r}")
    label, assigned = value.split("=", 1)
    if not label or not assigned:
        raise ValueError(f"{option} must use non-empty LABEL=VALUE syntax.")
    return label, assigned


def resolve_eval_source(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / "batch_eval_summary.csv", path / "batch_eval_config.yml"
    return path, path.parent / "batch_eval_config.yml"


def normalized_split_hash(path: Path) -> str:
    with path.open() as f:
        payload = yaml.safe_load(f)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_recorded_split(recorded: str, config: dict[str, Any]) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    marker = "splits/"
    text = str(path)
    if marker in text:
        relative = text.split(marker, 1)[1]
        candidate = Path(config["sweep_dir"]) / "splits" / relative
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve recorded split file: {recorded}")


def threshold_policy(config: dict[str, Any]) -> str:
    mode = config.get("threshold_mode")
    if mode == "fixed" or mode is None:
        return f"fixed:{float(config.get('threshold', 0.5)):g}"
    if mode == "selected_checkpoint":
        return "validation_selected_from_checkpoint_training"
    if mode == "external_map":
        return "validation_selected_posthoc_map"
    return str(mode)


def load_batch(
    label: str,
    source: Path,
    *,
    checkpoint_policy: str,
    behavior: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path, config_path = resolve_eval_source(source)
    if not summary_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Batch {label!r} requires {summary_path} and {config_path}."
        )
    frame = pd.read_csv(summary_path)
    config = yaml.safe_load(config_path.read_text())
    if behavior is not None:
        frame = frame[frame["behavior"] == behavior]
    frame = frame[
        (pd.to_numeric(frame["eval_returncode"], errors="coerce") == 0)
        & frame[METRICS].notna().all(axis=1)
    ].copy()
    if frame.empty:
        raise ValueError(f"Batch {label!r} has no successful metric rows.")
    duplicates = frame.duplicated(RUN_KEYS, keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, RUN_KEYS].to_dict(orient="records")
        raise ValueError(f"Batch {label!r} has duplicate metric keys: {keys[:3]}")

    frame["outer_id"] = frame["outer_id"].astype(str)
    frame["inner_seed"] = frame["inner_seed"].astype(str)
    frame["batch"] = label
    frame["checkpoint_policy"] = checkpoint_policy
    frame["threshold_policy"] = threshold_policy(config)
    for split_column in ("split_path", "val_split_path"):
        signature_column = split_column.replace("_path", "_signature")
        frame[signature_column] = frame[split_column].map(
            lambda value: normalized_split_hash(
                resolve_recorded_split(str(value), config)
            )
        )
    metadata = {
        "label": label,
        "source": str(source.resolve()),
        "summary": str(summary_path.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint_policy": checkpoint_policy,
        "threshold_policy": threshold_policy(config),
        "subset": sorted(frame["subset"].unique().tolist()),
        "levels": sorted(frame["level"].unique().tolist()),
        "episode_matching_method": config.get("episode_matching_method"),
        "episode_overlap_denominator": config.get(
            "episode_overlap_denominator"
        ),
        "threshold_selection_episode_matching_method": config.get(
            "threshold_selection_episode_matching_method"
        ),
        "threshold_selection_episode_overlap_denominator": config.get(
            "threshold_selection_episode_overlap_denominator"
        ),
    }
    return frame, metadata


def validate_compatibility(batches: list[pd.DataFrame], metadata: list[dict]) -> None:
    subsets = {tuple(item["subset"]) for item in metadata}
    if len(subsets) != 1:
        raise ValueError(f"Evaluation subsets differ across batches: {subsets}")
    if any("episode" in set(frame["level"]) for frame in batches):
        methods = {item["episode_matching_method"] for item in metadata}
        denominators = {item["episode_overlap_denominator"] for item in metadata}
        if None in methods or len(methods) != 1 or None in denominators or len(denominators) != 1:
            raise ValueError(
                "Episode scorer provenance is missing or differs across batches."
            )

    signature_columns = ["split_signature", "val_split_signature"]
    reference = batches[0][RUN_KEYS + signature_columns]
    for candidate in batches[1:]:
        merged = reference.merge(
            candidate[RUN_KEYS + signature_columns],
            on=RUN_KEYS,
            suffixes=("_reference", "_candidate"),
            how="outer",
            indicator=True,
        )
        if not (merged["_merge"] == "both").all():
            raise ValueError("Batches do not contain the same run/behavior/level keys.")
        for column in signature_columns:
            if not (
                merged[f"{column}_reference"] == merged[f"{column}_candidate"]
            ).all():
                raise ValueError(f"Batches use different {column.replace('_', ' ')}s.")


def summarize_batches(combined: pd.DataFrame, *, by_outer: bool) -> pd.DataFrame:
    groups = [
        "batch",
        "checkpoint_policy",
        "threshold_policy",
        "behavior",
        "subset",
        "level",
    ]
    if by_outer:
        groups.insert(1, "outer_id")
    rows = []
    for keys, group in combined.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, keys))
        row["n_runs"] = int(len(group))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        meets = (group[METRICS] >= 0.8).all(axis=1)
        row["n_runs_all_metrics_ge_0_80"] = int(meets.sum())
        row["all_metric_means_ge_0_80"] = bool(
            all(row[f"mean_{metric}"] >= 0.8 for metric in METRICS)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_deltas(
    batches: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_parts = []
    for (baseline_label, baseline), (candidate_label, candidate) in itertools.combinations(
        batches, 2
    ):
        merged = baseline[RUN_KEYS + METRICS].merge(
            candidate[RUN_KEYS + METRICS],
            on=RUN_KEYS,
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        merged.insert(0, "baseline_batch", baseline_label)
        merged.insert(1, "candidate_batch", candidate_label)
        for metric in METRICS:
            merged[f"delta_{metric}"] = (
                merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
            )
        delta_parts.append(merged)
    deltas = pd.concat(delta_parts, ignore_index=True)

    rows = []
    groups = ["baseline_batch", "candidate_batch", "behavior", "subset", "level"]
    for keys, group in deltas.groupby(groups, sort=True):
        row = dict(zip(groups, keys))
        row["n_paired_runs"] = int(len(group))
        for metric in METRICS:
            values = group[f"delta_{metric}"]
            row[f"mean_delta_{metric}"] = float(values.mean())
            row[f"wins_{metric}"] = int((values > 0).sum())
            row[f"ties_{metric}"] = int((values == 0).sum())
            row[f"losses_{metric}"] = int((values < 0).sum())
        rows.append(row)
    return deltas, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compatible split-sweep evaluation batches."
    )
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        help="Evaluation batch in LABEL=PATH form. Provide at least twice.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        action="append",
        required=True,
        help="Checkpoint policy in LABEL=POLICY form, one for each batch.",
    )
    parser.add_argument("--behavior", default=None, help="Optional behavior filter.")
    parser.add_argument("--out-dir", required=True, help="Comparison output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_specs = [parse_assignment(value, option="--batch") for value in args.batch]
    if len(batch_specs) < 2 or len({label for label, _ in batch_specs}) != len(batch_specs):
        raise ValueError("Provide at least two uniquely labeled --batch values.")
    policy_specs = dict(
        parse_assignment(value, option="--checkpoint-policy")
        for value in args.checkpoint_policy
    )
    labels = {label for label, _ in batch_specs}
    if set(policy_specs) != labels:
        raise ValueError("Provide exactly one --checkpoint-policy for every batch label.")

    loaded = [
        load_batch(
            label,
            Path(path),
            checkpoint_policy=policy_specs[label],
            behavior=args.behavior,
        )
        for label, path in batch_specs
    ]
    frames = [frame for frame, _metadata in loaded]
    metadata = [item for _frame, item in loaded]
    validate_compatibility(frames, metadata)
    combined = pd.concat(frames, ignore_index=True)
    deltas, pair_summary = pairwise_deltas(
        [(label, frame) for (label, _path), frame in zip(batch_specs, frames)]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_metrics.csv", index=False)
    summarize_batches(combined, by_outer=False).to_csv(
        out_dir / "batch_summary.csv", index=False
    )
    summarize_batches(combined, by_outer=True).to_csv(
        out_dir / "outer_summary.csv", index=False
    )
    deltas.to_csv(out_dir / "paired_run_deltas.csv", index=False)
    pair_summary.to_csv(out_dir / "pairwise_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "behavior_filter": args.behavior,
        "batches": metadata,
        "pair_orientation": "later --batch values are candidates against earlier values",
        "outputs": {
            name: str((out_dir / name).resolve())
            for name in (
                "combined_metrics.csv",
                "batch_summary.csv",
                "outer_summary.csv",
                "paired_run_deltas.csv",
                "pairwise_summary.csv",
            )
        },
    }
    with (out_dir / "comparison_manifest.yml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"📝 Wrote comparison reports to {out_dir}")


if __name__ == "__main__":
    main()
