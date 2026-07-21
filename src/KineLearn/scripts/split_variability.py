#!/usr/bin/env python3
"""
Generate and optionally execute split-variability experiments for KineLearn.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

import yaml
from sklearn.model_selection import train_test_split


def load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Split file {path} must be a mapping.")
        return payload

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


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure model sensitivity to train/test and train/val split choice by "
            "generating reproducible split files and optionally running training. "
            "Can also resume an existing split-variability sweep."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video-list",
        help="YAML list of video paths used to generate outer train/test splits.",
    )
    source.add_argument(
        "--base-split",
        help="Existing train/test split file to hold test set fixed while varying train/val splits.",
    )
    source.add_argument(
        "--resume",
        help="Existing split-variability output directory to inspect and resume.",
    )
    parser.add_argument("--kl-config", default=None, help="KineLearn config YAML.")
    parser.add_argument("--behavior", default=None, help="Behavior to train.")
    parser.add_argument(
        "--features-dir",
        default="features",
        help="Directory containing frame/extracted feature files.",
    )
    parser.add_argument(
        "--outer-seeds",
        nargs="*",
        type=int,
        default=[],
        help="Seeds for outer train/test splits. Required with --video-list.",
    )
    parser.add_argument(
        "--inner-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for explicit train/val splits within each outer split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of videos reserved for test when generating outer splits.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of training videos reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed passed through to kinelearn-train.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional focal alpha override passed through to kinelearn-train.",
    )
    parser.add_argument(
        "--keypoint-noise-std",
        type=float,
        default=None,
        help="Optional training-time keypoint noise std override passed through to kinelearn-train.",
    )
    parser.add_argument(
        "--train-command",
        default="kinelearn-train",
        help="Training executable to invoke when --execute is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run trainings immediately. Otherwise, only write the experiment plan.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/split_variability/<timestamp>/",
    )
    return parser.parse_args()


def default_out_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "split_variability" / timestamp


def load_video_stems(video_list_path: Path) -> list[str]:
    video_paths = load_yaml(video_list_path)
    if not isinstance(video_paths, list) or not all(isinstance(v, str) for v in video_paths):
        raise ValueError(f"{video_list_path} must be a YAML list of video paths.")
    return [Path(v).stem for v in video_paths]


def normalize_split_sections(split_info: dict[str, Any], path: Path) -> tuple[list[str], list[str]]:
    lowered = {str(k).strip().lower(): v for k, v in split_info.items()}
    train = lowered.get("train", lowered.get("train videos"))
    test = lowered.get("test", lowered.get("test videos"))
    if train is None or test is None:
        raise ValueError(f"Split file {path} must contain train/test sections.")
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError(f"Split file {path} must contain list-valued train/test sections.")
    return list(train), list(test)


def build_outer_splits(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.base_split:
        split_info = load_split_file(Path(args.base_split))
        train_stems, test_stems = normalize_split_sections(split_info, Path(args.base_split))
        return [
            {
                "outer_id": "fixed",
                "outer_seed": None,
                "train_stems": train_stems,
                "test_stems": test_stems,
                "source_split": str(Path(args.base_split).resolve()),
            }
        ]

    if not args.outer_seeds:
        raise ValueError("--outer-seeds is required when using --video-list.")

    stems = load_video_stems(Path(args.video_list))
    outer_splits = []
    for outer_seed in args.outer_seeds:
        train_stems, test_stems = train_test_split(
            stems, test_size=args.test_fraction, random_state=outer_seed
        )
        outer_splits.append(
            {
                "outer_id": f"outer_seed{outer_seed}",
                "outer_seed": int(outer_seed),
                "train_stems": list(train_stems),
                "test_stems": list(test_stems),
                "source_split": None,
            }
        )
    return outer_splits


def manifest_from_stdout(stdout: str) -> str | None:
    matches = re.findall(r"Wrote\s+(.+train_manifest\.yml)", stdout)
    return matches[-1].strip() if matches else None


def run_output_dir(base_out_dir: Path, *, outer_id: str, inner_seed: int | str) -> Path:
    return base_out_dir / "runs" / str(outer_id) / f"inner_seed{inner_seed}"


def build_plan(
    args: argparse.Namespace, out_dir: Path, val_fraction: float
) -> list[dict[str, Any]]:
    runs = []
    split_root = out_dir / "splits"
    outer_splits = build_outer_splits(args)

    for outer in outer_splits:
        outer_dir = split_root / outer["outer_id"]
        if outer["source_split"] is None:
            split_path = outer_dir / "train_test_split.yaml"
            save_yaml(
                split_path,
                {
                    "seed": outer["outer_seed"],
                    "test_fraction": args.test_fraction,
                    "train": outer["train_stems"],
                    "test": outer["test_stems"],
                },
            )
        else:
            split_path = Path(outer["source_split"])

        for inner_seed in args.inner_seeds:
            inner_train, inner_val = train_test_split(
                outer["train_stems"], test_size=val_fraction, random_state=inner_seed
            )
            val_split_path = outer_dir / f"train_val_split_seed{inner_seed}.yaml"
            save_yaml(
                val_split_path,
                {
                    "seed": int(inner_seed),
                    "val_fraction": float(val_fraction),
                    "train": list(inner_train),
                    "val": list(inner_val),
                },
            )

            command = [
                args.train_command,
                "--kl-config",
                args.kl_config,
                "--split",
                str(split_path),
                "--val-split",
                str(val_split_path),
                "--behavior",
                args.behavior,
                "--features-dir",
                args.features_dir,
                "--seed",
                str(args.seed),
                "--out-dir",
                str(run_output_dir(out_dir, outer_id=outer["outer_id"], inner_seed=inner_seed)),
            ]
            if args.focal_alpha is not None:
                command.extend(["--focal-alpha", str(args.focal_alpha)])
            if args.keypoint_noise_std is not None:
                command.extend(["--keypoint-noise-std", str(args.keypoint_noise_std)])

            runs.append(
                {
                    "outer_id": outer["outer_id"],
                    "outer_seed": outer["outer_seed"],
                    "inner_seed": int(inner_seed),
                    "split_path": str(Path(split_path).resolve()),
                    "val_split_path": str(val_split_path.resolve()),
                    "train_count": len(inner_train),
                    "val_count": len(inner_val),
                    "test_count": len(outer["test_stems"]),
                    "run_output_dir": str(
                        run_output_dir(
                            out_dir,
                            outer_id=outer["outer_id"],
                            inner_seed=inner_seed,
                        ).resolve()
                    ),
                    "command": command,
                }
            )
    return runs


def write_plan_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "outer_id",
        "outer_seed",
        "inner_seed",
        "split_path",
        "val_split_path",
        "train_count",
        "val_count",
        "test_count",
        "run_output_dir",
        "command",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["command"] = shlex.join(row["command"])
            writer.writerow(row)


def load_plan_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"outer_id", "inner_seed", "split_path", "val_split_path", "command"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required plan columns: {sorted(missing)}")
    return rows


def with_run_output_dir(command: list[str], out_dir: Path) -> list[str]:
    if "--out-dir" in command:
        idx = command.index("--out-dir")
        if idx == len(command) - 1:
            raise ValueError("Malformed command: --out-dir is missing its value.")
        updated = list(command)
        updated[idx + 1] = str(out_dir)
        return updated
    return [*command, "--out-dir", str(out_dir)]


def parse_run_command(row: dict[str, Any], sweep_dir: Path) -> list[str]:
    command_field = row.get("command")
    if not command_field:
        raise ValueError("Plan row is missing the command field.")
    command = shlex.split(command_field)
    managed_out_dir = managed_run_output_dir(row, sweep_dir)
    return with_run_output_dir(command, managed_out_dir)


def load_summary_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["outer_id"], row["inner_seed"]): row for row in rows}


def infer_manifest_path_from_run_dir(run_dir: Path) -> Path | None:
    manifest_path = run_dir / "train_manifest.yml"
    return manifest_path.resolve() if manifest_path.exists() else None


def infer_manifest_path_from_summary(row: dict[str, Any]) -> Path | None:
    manifest_path = row.get("manifest_path")
    if not manifest_path:
        return None
    candidate = Path(manifest_path)
    return candidate.resolve() if candidate.exists() else None


def infer_manifest_path_by_split_match(
    sweep_dir: Path,
    *,
    split_path: str,
    val_split_path: str,
) -> Path | None:
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


def managed_run_output_dir(row: dict[str, Any], sweep_dir: Path) -> Path:
    configured = row.get("run_output_dir")
    if configured:
        return Path(configured)
    return run_output_dir(
        sweep_dir,
        outer_id=row["outer_id"],
        inner_seed=row["inner_seed"],
    )


def inspect_resume_runs(
    sweep_dir: Path,
    plan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows = load_summary_rows(sweep_dir / "results_summary.csv")
    inspected: list[dict[str, Any]] = []

    for row in plan_rows:
        run_key = (row["outer_id"], row["inner_seed"])
        run_dir = managed_run_output_dir(row, sweep_dir)
        manifest_path = infer_manifest_path_from_run_dir(run_dir)
        if manifest_path is None:
            manifest_path = infer_manifest_path_from_summary(summary_rows.get(run_key, {}))
        if manifest_path is None:
            manifest_path = infer_manifest_path_by_split_match(
                sweep_dir,
                split_path=row["split_path"],
                val_split_path=row["val_split_path"],
            )

        state = "complete" if manifest_path is not None else "pending"
        inspected.append(
            {
                **row,
                "run_output_dir": str(run_dir.resolve()),
                "command_list": parse_run_command(row, sweep_dir),
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "state": state,
                "has_partial_run_dir": bool(run_dir.exists() and manifest_path is None),
            }
        )

    return inspected


def cleanup_incomplete_run_dir(run: dict[str, Any]) -> bool:
    run_dir = Path(run["run_output_dir"])
    if run["state"] == "complete" or not run_dir.exists():
        return False
    shutil.rmtree(run_dir)
    return True


def summarize_resume_runs(runs: list[dict[str, Any]]) -> dict[str, int]:
    total = len(runs)
    complete = sum(1 for run in runs if run["state"] == "complete")
    pending = total - complete
    incomplete = sum(1 for run in runs if run["has_partial_run_dir"])
    missing = pending - incomplete
    return {
        "total": total,
        "complete": complete,
        "pending": pending,
        "incomplete": incomplete,
        "missing": missing,
    }


def print_resume_report(runs: list[dict[str, Any]], *, execute: bool) -> None:
    summary = summarize_resume_runs(runs)
    print(
        "Resume summary: "
        f"planned={summary['total']}, "
        f"complete={summary['complete']}, "
        f"pending={summary['pending']}, "
        f"incomplete={summary['incomplete']}, "
        f"missing={summary['missing']}"
    )

    completed_runs = [run for run in runs if run["state"] == "complete"]
    if completed_runs:
        print("Completed runs:")
        for run in completed_runs:
            print(
                f"  - outer={run['outer_id']}, inner_seed={run['inner_seed']} "
                f"-> {run['manifest_path']}"
            )

    incomplete_runs = [run for run in runs if run["has_partial_run_dir"]]
    if incomplete_runs:
        print("Incomplete runs found:")
        for run in incomplete_runs:
            print(
                f"  - outer={run['outer_id']}, inner_seed={run['inner_seed']} "
                f"-> {run['run_output_dir']}"
            )

    pending_runs = [run for run in runs if run["state"] != "complete"]
    if pending_runs:
        verb = "Will rerun" if execute else "Would rerun"
        print(f"{verb}:")
        for run in pending_runs:
            print(
                f"  - outer={run['outer_id']}, inner_seed={run['inner_seed']} "
                f"-> {run['run_output_dir']}"
            )
    else:
        print("No pending runs detected.")


def validate_new_plan_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.kl_config:
        missing.append("--kl-config")
    if not args.behavior:
        missing.append("--behavior")
    if not args.inner_seeds:
        missing.append("--inner-seeds")
    if missing:
        raise ValueError(
            "The following arguments are required unless --resume is used: "
            + ", ".join(missing)
        )


def enrich_summary_row_from_manifest(
    row: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """Add recorded training and checkpoint metrics to a sweep summary row."""
    manifest = load_yaml(manifest_path)
    training_run = manifest.get("training_run", {})
    test_metrics = training_run.get("test_metrics", {})
    checkpoint_selection = training_run.get("checkpoint_selection") or {}
    selected_checkpoint = checkpoint_selection.get("selected") or {}

    row["best_epoch_by_val_loss"] = training_run.get("best_epoch_by_val_loss")
    row["epochs_completed"] = training_run.get("epochs_completed")
    row["best_epoch_by_checkpoint_selection"] = selected_checkpoint.get("epoch")
    row["selected_threshold"] = selected_checkpoint.get("threshold")
    row["selected_val_episode_f1"] = selected_checkpoint.get("f1")
    row["selected_val_episode_precision"] = selected_checkpoint.get("precision")
    row["selected_val_episode_recall"] = selected_checkpoint.get("recall")
    for key, value in test_metrics.items():
        row[f"test_{key}"] = value
    return row


def execute_runs(
    runs: list[dict[str, Any]],
    *,
    out_dir: Path,
    clean_incomplete: bool,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []

    for idx, run in enumerate(runs, start=1):
        if run["state"] == "complete":
            manifest_path = Path(run["manifest_path"])
            row = {
                "outer_id": run["outer_id"],
                "outer_seed": run.get("outer_seed"),
                "inner_seed": run["inner_seed"],
                "split_path": run["split_path"],
                "val_split_path": run["val_split_path"],
                "run_output_dir": run["run_output_dir"],
                "status": "complete_existing",
                "returncode": 0,
                "manifest_path": run["manifest_path"],
            }
            enrich_summary_row_from_manifest(row, manifest_path)
            summary_rows.append(row)
            aggregate_results(out_dir / "results_summary.csv", summary_rows)
            continue

        print(
            f"\n=== Run {idx}/{len(runs)} "
            f"(outer={run['outer_id']}, inner_seed={run['inner_seed']}) ==="
        )

        cleaned = False
        if clean_incomplete and cleanup_incomplete_run_dir(run):
            cleaned = True
            print(f"🧹 Removed incomplete run directory {run['run_output_dir']}")

        Path(run["run_output_dir"]).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            run["command_list"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")

        manifest_path = infer_manifest_path_from_run_dir(Path(run["run_output_dir"]))
        if manifest_path is None:
            manifest_path_str = manifest_from_stdout(completed.stdout)
            if manifest_path_str:
                candidate = Path(manifest_path_str)
                if candidate.exists():
                    manifest_path = candidate.resolve()

        row = {
            "outer_id": run["outer_id"],
            "outer_seed": run.get("outer_seed"),
            "inner_seed": run["inner_seed"],
            "split_path": run["split_path"],
            "val_split_path": run["val_split_path"],
            "run_output_dir": run["run_output_dir"],
            "status": "rerun" if cleaned else "executed",
            "returncode": int(completed.returncode),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
        }

        if manifest_path is not None and manifest_path.exists():
            enrich_summary_row_from_manifest(row, manifest_path)

        summary_rows.append(row)
        aggregate_results(out_dir / "results_summary.csv", summary_rows)

    return summary_rows


def aggregate_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.resume:
        out_dir = Path(args.resume)
        plan_path = out_dir / "experiment_plan.csv"
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        if not plan_path.exists():
            raise FileNotFoundError(f"Expected experiment plan not found: {plan_path}")

        runs = inspect_resume_runs(out_dir, load_plan_csv(plan_path))
        print_resume_report(runs, execute=args.execute)
        if not args.execute:
            print("Dry run only; no trainings launched.")
            return

        execute_runs(runs, out_dir=out_dir, clean_incomplete=True)
        print(f"\n📝 Wrote {out_dir / 'results_summary.csv'}")
        return

    validate_new_plan_args(args)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    kl_config = load_yaml(Path(args.kl_config))
    training_cfg = kl_config.get("training", {})
    val_fraction = (
        float(args.val_fraction)
        if args.val_fraction is not None
        else float(training_cfg.get("val_fraction", 0.2))
    )

    runs = build_plan(args, out_dir, val_fraction)
    write_plan_csv(out_dir / "experiment_plan.csv", runs)
    save_yaml(
        out_dir / "experiment_config.yml",
        {
            "kl_config": str(Path(args.kl_config).resolve()),
            "behavior": args.behavior,
            "features_dir": str(Path(args.features_dir).resolve()),
            "training_seed": int(args.seed),
            "keypoint_noise_std": (
                float(args.keypoint_noise_std)
                if args.keypoint_noise_std is not None
                else None
            ),
            "val_fraction": float(val_fraction),
            "test_fraction": float(args.test_fraction),
            "source": {
                "video_list": str(Path(args.video_list).resolve()) if args.video_list else None,
                "base_split": str(Path(args.base_split).resolve()) if args.base_split else None,
            },
            "outer_seeds": list(args.outer_seeds),
            "inner_seeds": list(args.inner_seeds),
            "execute": bool(args.execute),
        },
    )
    print(f"📝 Wrote {out_dir / 'experiment_plan.csv'}")

    prepared_runs = [
        {
            **run,
            "state": "pending",
            "command_list": list(run["command"]),
            "manifest_path": None,
            "has_partial_run_dir": False,
        }
        for run in runs
    ]

    if not args.execute:
        print("Dry run only; no trainings launched.")
        return

    execute_runs(prepared_runs, out_dir=out_dir, clean_incomplete=False)
    print(f"\n📝 Wrote {out_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
