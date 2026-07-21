#!/usr/bin/env python3
"""
Run kinelearn-eval over manifests produced by a split-variability sweep.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml

from KineLearn.core.evaluation import (
    EPISODE_MATCHING_METHOD,
    EPISODE_OVERLAP_DENOMINATOR,
)


def load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "split_variability_evals" / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run kinelearn-eval across manifests from a split-variability sweep "
            "and aggregate the resulting metrics."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Sweep output directory, results_summary.csv, or experiment_plan.csv "
            "from kinelearn-split-variability."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "Optional evaluation source manifest(s). When omitted, each run evaluates its own "
            "train_manifest.yml. When provided, those source manifests are evaluated against "
            "each run via --eval-manifest."
        ),
    )
    parser.add_argument(
        "--eval-command",
        default="kinelearn-eval",
        help="Evaluation executable to invoke (default: kinelearn-eval).",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "val", "test"],
        default="val",
        help="Subset to evaluate (default: val).",
    )
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Fixed probability threshold to pass to every evaluation "
            "(default: 0.5 unless --use-selected-threshold is set)."
        ),
    )
    threshold_group.add_argument(
        "--use-selected-threshold",
        action="store_true",
        help=(
            "Use each run manifest's validation-selected checkpoint threshold. "
            "This is supported only when each run evaluates its own manifest."
        ),
    )
    threshold_group.add_argument(
        "--threshold-map",
        default=None,
        help=(
            "CSV containing outer_id, inner_seed, and threshold columns. Use this "
            "for thresholds selected independently from the training manifest."
        ),
    )
    parser.add_argument(
        "--level",
        choices=["frame", "episode", "both"],
        default="frame",
        help="Evaluation level to request (default: frame).",
    )
    parser.add_argument(
        "--episode-min-frames",
        type=int,
        default=16,
        help="Minimum positive frames for predicted episodes (default: 16).",
    )
    parser.add_argument(
        "--episode-max-gap",
        type=int,
        default=3,
        help="Maximum internal gap for predicted episodes (default: 3).",
    )
    parser.add_argument(
        "--episode-overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap fraction for episode matching (default: 0.2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override.",
    )
    parser.add_argument(
        "--ensemble-recusal-policy",
        choices=["none", "train", "train_val"],
        default="train_val",
        help="Recusal policy to pass through for ensemble evaluation (default: train_val).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/split_variability_evals/<timestamp>/",
    )
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        help=(
            "Additional directory to search for relocated train manifests. "
            "May be provided more than once."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible evaluation in --out-dir, reusing only complete "
            "per-run outputs and rerunning missing or incomplete runs."
        ),
    )
    return parser.parse_args()


def resolve_source(source: Path) -> tuple[Path, Path]:
    if source.is_dir():
        summary_path = source / "results_summary.csv"
        plan_path = source / "experiment_plan.csv"
        if summary_path.exists():
            return source, summary_path
        if plan_path.exists():
            return source, plan_path
        raise FileNotFoundError(
            f"No results_summary.csv or experiment_plan.csv found in {source}"
        )

    if source.name in {"results_summary.csv", "experiment_plan.csv"}:
        return source.parent, source

    raise ValueError(
        f"Unsupported source {source}; pass a sweep directory, results_summary.csv, or experiment_plan.csv."
    )


def load_run_rows(table_path: Path) -> list[dict[str, Any]]:
    with open(table_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {table_path}")
    return rows


def discover_runs(table_path: Path) -> list[dict[str, Any]]:
    rows = load_run_rows(table_path)
    if "manifest_path" in rows[0]:
        runs = [row for row in rows if row.get("manifest_path")]
        if not runs:
            raise ValueError(
                f"{table_path} does not contain any manifest_path values yet; "
                "run kinelearn-split-variability with --execute first."
            )
        return runs

    if {"outer_id", "inner_seed", "split_path", "val_split_path"}.issubset(rows[0].keys()):
        return rows

    raise ValueError(f"Could not interpret sweep table schema in {table_path}")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def eval_output_dir(base_out_dir: Path, *, outer_id: str, inner_seed: str) -> Path:
    return base_out_dir / "runs" / str(outer_id) / f"inner_seed{inner_seed}"


def infer_manifest_path(
    run: dict[str, Any],
    sweep_dir: Path,
    manifest_roots: list[Path] | None = None,
) -> Path | None:
    manifest_path = run.get("manifest_path")
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            return path.resolve()

    split_path = run.get("split_path")
    val_split_path = run.get("val_split_path")
    if not split_path or not val_split_path:
        return None

    roots = [sweep_dir, *(manifest_roots or [])]
    recorded_run_name = Path(manifest_path).parent.name if manifest_path else None
    if recorded_run_name:
        for root in roots:
            direct = root / recorded_run_name / "train_manifest.yml"
            if direct.exists():
                return direct.resolve()

    candidates = [
        candidate
        for root in roots
        for candidate in root.rglob("train_manifest.yml")
    ]
    for candidate in candidates:
        try:
            manifest = load_yaml(candidate)
        except Exception:
            continue
        if (
            manifest.get("split") == str(Path(split_path).resolve())
            and manifest.get("val_split") == str(Path(val_split_path).resolve())
        ):
            return candidate.resolve()
    return None


def selected_checkpoint_threshold(manifest_path: Path) -> float:
    manifest = load_yaml(manifest_path)
    selection = (manifest.get("training_run") or {}).get("checkpoint_selection") or {}
    if selection.get("enabled") is not True:
        raise ValueError(
            f"Manifest {manifest_path} does not have checkpoint selection enabled."
        )
    selected = selection.get("selected") or {}
    value = selected.get("threshold")
    if value is None:
        raise ValueError(
            f"Manifest {manifest_path} has no validation-selected checkpoint threshold."
        )
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Manifest {manifest_path} records an invalid selected threshold: {value!r}."
        )
    return threshold


def threshold_for_run(args: argparse.Namespace, manifest_path: Path) -> float:
    if args.use_selected_threshold:
        return selected_checkpoint_threshold(manifest_path)
    return 0.5 if args.threshold is None else float(args.threshold)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_map_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(".yml")
    if not metadata_path.exists():
        return None
    metadata = load_yaml(metadata_path)
    recorded_map = metadata.get("threshold_map")
    if recorded_map and Path(recorded_map).resolve() != path.resolve():
        raise ValueError(
            f"Threshold-map metadata {metadata_path} refers to a different CSV."
        )
    return metadata


def load_threshold_map(path: Path) -> dict[tuple[str, str], float]:
    rows = load_run_rows(path)
    required = {"outer_id", "inner_seed", "threshold"}
    if not required.issubset(rows[0]):
        raise ValueError(
            f"Threshold map {path} must contain columns {sorted(required)}."
        )
    thresholds: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["outer_id"]), str(row["inner_seed"]))
        if key in thresholds:
            raise ValueError(f"Threshold map {path} contains duplicate run key {key}.")
        threshold = float(row["threshold"])
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold map {path} has invalid threshold for {key}.")
        thresholds[key] = threshold
    return thresholds


def build_eval_command(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    out_dir: Path,
    threshold: float,
) -> list[str]:
    command = [args.eval_command]
    if args.manifest:
        for source_manifest in args.manifest:
            command.extend(["--manifest", str(Path(source_manifest).resolve())])
        command.extend(["--eval-manifest", str(manifest_path.resolve())])
        command.extend(["--ensemble-recusal-policy", args.ensemble_recusal_policy])
    else:
        command.extend(["--manifest", str(manifest_path.resolve())])
    command.extend(
        [
            "--subset",
            args.subset,
            "--threshold",
            str(threshold),
            "--level",
            args.level,
            "--episode-min-frames",
            str(args.episode_min_frames),
            "--episode-max-gap",
            str(args.episode_max_gap),
            "--episode-overlap-threshold",
            str(args.episode_overlap_threshold),
            "--out",
            str(out_dir),
        ]
    )
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    return command


def load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected evaluation metrics file not found: {metrics_path}")
    return pd.read_csv(metrics_path).to_dict(orient="records")


def batch_config_payload(
    args: argparse.Namespace,
    *,
    source: Path,
    sweep_dir: Path,
    table_path: Path,
) -> dict[str, Any]:
    map_path = Path(args.threshold_map) if args.threshold_map else None
    map_metadata = threshold_map_metadata(map_path) if map_path else None
    return {
        "source": str(source.resolve()),
        "sweep_dir": str(sweep_dir.resolve()),
        "table_path": str(table_path.resolve()),
        "manifests": [str(Path(p).resolve()) for p in args.manifest],
        "manifest_roots": [str(Path(p).resolve()) for p in args.manifest_root],
        "subset": args.subset,
        "threshold_mode": (
            "selected_checkpoint"
            if args.use_selected_threshold
            else "external_map"
            if args.threshold_map
            else "fixed"
        ),
        "threshold": (
            None
            if args.use_selected_threshold or args.threshold_map
            else float(0.5 if args.threshold is None else args.threshold)
        ),
        "threshold_map": (
            str(Path(args.threshold_map).resolve()) if args.threshold_map else None
        ),
        "threshold_map_sha256": (
            file_sha256(map_path) if map_path else None
        ),
        "threshold_map_metadata": (
            str(map_path.with_suffix(".yml").resolve()) if map_metadata else None
        ),
        "threshold_map_metadata_sha256": (
            file_sha256(map_path.with_suffix(".yml")) if map_metadata else None
        ),
        "threshold_selection_episode_matching_method": (
            map_metadata.get("episode_matching_method") if map_metadata else None
        ),
        "threshold_selection_episode_overlap_denominator": (
            map_metadata.get("episode_overlap_denominator") if map_metadata else None
        ),
        "level": args.level,
        "ensemble_recusal_policy": args.ensemble_recusal_policy,
        "episode_min_frames": int(args.episode_min_frames),
        "episode_max_gap": int(args.episode_max_gap),
        "episode_overlap_threshold": float(args.episode_overlap_threshold),
        "episode_matching_method": EPISODE_MATCHING_METHOD,
        "episode_overlap_denominator": EPISODE_OVERLAP_DENOMINATOR,
        "batch_size": args.batch_size,
        "eval_command": args.eval_command,
    }


def validate_resume_config(config_path: Path, expected: dict[str, Any]) -> None:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: evaluation config not found at {config_path}."
        )
    recorded = load_yaml(config_path)
    if recorded != expected:
        raise ValueError(
            "Cannot resume because the requested evaluation settings differ from "
            f"those recorded in {config_path}."
        )


def completed_eval_is_reusable(
    run_out_dir: Path,
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    threshold: float,
) -> bool:
    required = [
        run_out_dir / "eval_summary.yml",
        run_out_dir / "per_behavior_metrics.csv",
        run_out_dir / "frame_predictions.parquet",
    ]
    if args.level in {"episode", "both"}:
        required.append(run_out_dir / "episode_errors.csv")
    if not all(path.exists() for path in required):
        return False

    try:
        summary = load_yaml(run_out_dir / "eval_summary.yml")
        expected_sources = (
            [str(Path(path).resolve()) for path in args.manifest]
            if args.manifest
            else [str(manifest_path.resolve())]
        )
        expected_eval_manifest = (
            str(manifest_path.resolve()) if args.manifest else None
        )
        episode_settings = summary.get("episode_settings") or {}
        return bool(
            summary.get("subset") == args.subset
            and summary.get("level") == args.level
            and math.isclose(
                float(summary.get("threshold")), threshold, abs_tol=1e-12
            )
            and summary.get("manifests") == expected_sources
            and summary.get("evaluation_manifest") == expected_eval_manifest
            and episode_settings.get("min_pred_frames") == args.episode_min_frames
            and episode_settings.get("max_gap") == args.episode_max_gap
            and math.isclose(
                float(episode_settings.get("overlap_threshold")),
                args.episode_overlap_threshold,
                abs_tol=1e-12,
            )
            and episode_settings.get("matching_method")
            == EPISODE_MATCHING_METHOD
            and episode_settings.get("overlap_denominator")
            == EPISODE_OVERLAP_DENOMINATOR
        )
    except (AttributeError, TypeError, ValueError):
        return False


def metric_aggregate_rows(
    summary_rows: list[dict[str, Any]], *, by_outer: bool
) -> list[dict[str, Any]]:
    successful = [
        row
        for row in summary_rows
        if row.get("eval_returncode") == 0
        and not row.get("error")
        and all(metric in row for metric in ("f1", "precision", "recall"))
    ]
    if not successful:
        return []

    frame = pd.DataFrame(successful)
    group_columns = ["behavior", "subset", "level"]
    if by_outer:
        group_columns = ["outer_id", "outer_seed", *group_columns]

    output_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_runs"] = int(len(group))
        for metric in ("f1", "precision", "recall", "threshold"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        meets = (
            (pd.to_numeric(group["f1"], errors="coerce") >= 0.8)
            & (pd.to_numeric(group["precision"], errors="coerce") >= 0.8)
            & (pd.to_numeric(group["recall"], errors="coerce") >= 0.8)
        )
        row["n_runs_all_metrics_ge_0_80"] = int(meets.sum())
        row["fraction_runs_all_metrics_ge_0_80"] = float(meets.mean())
        row["all_metric_means_ge_0_80"] = bool(
            all(
                row.get(f"mean_{metric}", 0.0) >= 0.8
                for metric in ("f1", "precision", "recall")
            )
        )
        output_rows.append(row)
    return output_rows


def write_batch_reports(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    aggregate_csv(out_dir / "batch_eval_summary.csv", summary_rows)
    aggregate_csv(
        out_dir / "batch_eval_aggregate.csv",
        metric_aggregate_rows(summary_rows, by_outer=False),
    )
    aggregate_csv(
        out_dir / "batch_eval_outer_summary.csv",
        metric_aggregate_rows(summary_rows, by_outer=True),
    )


def main() -> None:
    args = parse_args()
    fixed_threshold = 0.5 if args.threshold is None else float(args.threshold)
    if not args.use_selected_threshold and not (0.0 < fixed_threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if args.use_selected_threshold and args.manifest:
        raise ValueError(
            "--use-selected-threshold cannot be combined with --manifest because "
            "external prediction sources do not have a per-target-run threshold."
        )
    if not (0.0 < args.episode_overlap_threshold <= 1.0):
        raise ValueError("--episode-overlap-threshold must be in (0, 1].")
    if args.episode_min_frames <= 0:
        raise ValueError("--episode-min-frames must be positive.")
    if args.episode_max_gap < 0:
        raise ValueError("--episode-max-gap must be non-negative.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.resume and not args.out_dir:
        raise ValueError("--resume requires an explicit --out-dir.")

    source = Path(args.source)
    sweep_dir, table_path = resolve_source(source)
    run_rows = discover_runs(table_path)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    if args.resume and not out_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume: output directory not found: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = batch_config_payload(
        args, source=source, sweep_dir=sweep_dir, table_path=table_path
    )
    config_path = out_dir / "batch_eval_config.yml"
    if args.resume:
        validate_resume_config(config_path, config)
    else:
        write_yaml(config_path, config)

    manifest_roots = [Path(path) for path in args.manifest_root]
    threshold_map = (
        load_threshold_map(Path(args.threshold_map)) if args.threshold_map else None
    )
    if threshold_map is not None:
        expected_keys = {
            (str(run.get("outer_id")), str(run.get("inner_seed")))
            for run in run_rows
        }
        missing = sorted(expected_keys - set(threshold_map))
        extra = sorted(set(threshold_map) - expected_keys)
        if missing or extra:
            raise ValueError(
                "Threshold-map run coverage does not match the sweep: "
                f"missing={missing}, extra={extra}."
            )

    if args.use_selected_threshold:
        for run in run_rows:
            manifest_path = infer_manifest_path(run, sweep_dir, manifest_roots)
            if manifest_path is not None:
                selected_checkpoint_threshold(manifest_path)

    summary_rows: list[dict[str, Any]] = []
    for idx, run in enumerate(run_rows, start=1):
        outer_id = str(run.get("outer_id", "unknown"))
        inner_seed = str(run.get("inner_seed", "unknown"))
        manifest_path = infer_manifest_path(run, sweep_dir, manifest_roots)

        print(
            f"\n=== Eval {idx}/{len(run_rows)} "
            f"(outer={outer_id}, inner_seed={inner_seed}) ==="
        )

        if manifest_path is None:
            row = {
                "outer_id": run.get("outer_id"),
                "outer_seed": run.get("outer_seed"),
                "inner_seed": run.get("inner_seed"),
                "split_path": run.get("split_path"),
                "val_split_path": run.get("val_split_path"),
                "manifest_path": None,
                "eval_returncode": None,
                "error": "Could not resolve manifest path for run.",
            }
            summary_rows.append(row)
            write_batch_reports(out_dir, summary_rows)
            print("⚠️  Could not resolve manifest path; skipping.")
            continue

        run_out_dir = eval_output_dir(out_dir, outer_id=outer_id, inner_seed=inner_seed)
        run_key = (outer_id, inner_seed)
        threshold = (
            threshold_map[run_key]
            if threshold_map is not None
            else threshold_for_run(args, manifest_path)
        )
        reuse = args.resume and completed_eval_is_reusable(
            run_out_dir,
            args=args,
            manifest_path=manifest_path,
            threshold=threshold,
        )
        if reuse:
            print("↪️  Reusing completed evaluation artifacts.")
            returncode = 0
        else:
            command = build_eval_command(
                args,
                manifest_path=manifest_path,
                out_dir=run_out_dir,
                threshold=threshold,
            )
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            returncode = int(completed.returncode)

        base_row = {
            "outer_id": run.get("outer_id"),
            "outer_seed": run.get("outer_seed"),
            "inner_seed": run.get("inner_seed"),
            "split_path": run.get("split_path"),
            "val_split_path": run.get("val_split_path"),
            "manifest_path": str(manifest_path.resolve()),
            "eval_returncode": returncode,
            "evaluation_status": "reused" if reuse else "executed",
            "eval_out_dir": str(run_out_dir.resolve()),
            "subset": args.subset,
            "threshold": float(threshold),
            "threshold_source": (
                "checkpoint_selection"
                if args.use_selected_threshold
                else "external_map"
                if threshold_map is not None
                else "fixed"
            ),
            "level_requested": args.level,
        }

        if returncode != 0:
            row = dict(base_row)
            row["error"] = "kinelearn-eval returned non-zero exit status."
            summary_rows.append(row)
            write_batch_reports(out_dir, summary_rows)
            continue

        metrics_path = run_out_dir / "per_behavior_metrics.csv"
        try:
            metric_rows = load_metrics_rows(metrics_path)
        except Exception as exc:
            row = dict(base_row)
            row["error"] = str(exc)
            summary_rows.append(row)
            write_batch_reports(out_dir, summary_rows)
            continue

        for metric_row in metric_rows:
            row = dict(base_row)
            row.update(metric_row)
            summary_rows.append(row)

        write_batch_reports(out_dir, summary_rows)

    print(f"\n📝 Wrote {out_dir / 'batch_eval_summary.csv'}")
    print(f"📝 Wrote {out_dir / 'batch_eval_aggregate.csv'}")
    print(f"📝 Wrote {out_dir / 'batch_eval_outer_summary.csv'}")


if __name__ == "__main__":
    main()
