#!/usr/bin/env python3
"""
Run kinelearn-eval over manifests produced by a split-variability sweep.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to pass to kinelearn-eval (default: 0.5).",
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


def infer_manifest_path(run: dict[str, Any], sweep_dir: Path) -> Path | None:
    manifest_path = run.get("manifest_path")
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            return path.resolve()

    split_path = run.get("split_path")
    val_split_path = run.get("val_split_path")
    if not split_path or not val_split_path:
        return None

    candidates = list(sweep_dir.rglob("train_manifest.yml"))
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


def build_eval_command(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    out_dir: Path,
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
            str(args.threshold),
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


def main() -> None:
    args = parse_args()
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if not (0.0 < args.episode_overlap_threshold <= 1.0):
        raise ValueError("--episode-overlap-threshold must be in (0, 1].")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    source = Path(args.source)
    sweep_dir, table_path = resolve_source(source)
    run_rows = discover_runs(table_path)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(
        out_dir / "batch_eval_config.yml",
        {
            "source": str(source.resolve()),
            "sweep_dir": str(sweep_dir.resolve()),
            "table_path": str(table_path.resolve()),
            "manifests": [str(Path(p).resolve()) for p in args.manifest],
            "subset": args.subset,
            "threshold": float(args.threshold),
            "level": args.level,
            "ensemble_recusal_policy": args.ensemble_recusal_policy,
            "episode_min_frames": int(args.episode_min_frames),
            "episode_max_gap": int(args.episode_max_gap),
            "episode_overlap_threshold": float(args.episode_overlap_threshold),
            "batch_size": args.batch_size,
            "eval_command": args.eval_command,
        },
    )

    summary_rows: list[dict[str, Any]] = []
    for idx, run in enumerate(run_rows, start=1):
        outer_id = str(run.get("outer_id", "unknown"))
        inner_seed = str(run.get("inner_seed", "unknown"))
        manifest_path = infer_manifest_path(run, sweep_dir)

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
            aggregate_csv(out_dir / "batch_eval_summary.csv", summary_rows)
            print("⚠️  Could not resolve manifest path; skipping.")
            continue

        run_out_dir = eval_output_dir(out_dir, outer_id=outer_id, inner_seed=inner_seed)
        command = build_eval_command(args, manifest_path=manifest_path, out_dir=run_out_dir)
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

        base_row = {
            "outer_id": run.get("outer_id"),
            "outer_seed": run.get("outer_seed"),
            "inner_seed": run.get("inner_seed"),
            "split_path": run.get("split_path"),
            "val_split_path": run.get("val_split_path"),
            "manifest_path": str(manifest_path.resolve()),
            "eval_returncode": int(completed.returncode),
            "eval_out_dir": str(run_out_dir.resolve()),
            "subset": args.subset,
            "threshold": float(args.threshold),
            "level_requested": args.level,
        }

        if completed.returncode != 0:
            row = dict(base_row)
            row["error"] = "kinelearn-eval returned non-zero exit status."
            summary_rows.append(row)
            aggregate_csv(out_dir / "batch_eval_summary.csv", summary_rows)
            continue

        metrics_path = run_out_dir / "per_behavior_metrics.csv"
        try:
            metric_rows = load_metrics_rows(metrics_path)
        except Exception as exc:
            row = dict(base_row)
            row["error"] = str(exc)
            summary_rows.append(row)
            aggregate_csv(out_dir / "batch_eval_summary.csv", summary_rows)
            continue

        for metric_row in metric_rows:
            row = dict(base_row)
            row.update(metric_row)
            summary_rows.append(row)

        aggregate_csv(out_dir / "batch_eval_summary.csv", summary_rows)

    print(f"\n📝 Wrote {out_dir / 'batch_eval_summary.csv'}")


if __name__ == "__main__":
    main()
