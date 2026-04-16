#!/usr/bin/env python3
"""
Select ensemble members from compatible KineLearn runs using validation metrics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from KineLearn.core.manifests import (
    build_ensemble_manifest_payload,
    load_train_manifest,
    save_yaml,
    validate_selection_candidate_manifests,
)
from KineLearn.scripts.batch_eval_splits import (
    discover_runs,
    infer_manifest_path,
    resolve_source,
)
from KineLearn.scripts.eval import evaluate_manifest


FRAME_METRICS = {"frame_f1": "f1", "frame_precision": "precision", "frame_recall": "recall"}
EPISODE_METRICS = {
    "episode_f1": "f1",
    "episode_precision": "precision",
    "episode_recall": "recall",
}


def default_out_dir(behavior: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "ensembles" / behavior / f"selection_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deployable KineLearn ensemble from compatible train manifests "
            "using validation metrics."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Split-variability sweep directory, results_summary.csv, or "
            "experiment_plan.csv to scan for train manifests."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Direct path to a train_manifest.yml file to include as a candidate.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional human-readable ensemble name recorded in the output manifest.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted([*FRAME_METRICS.keys(), *EPISODE_METRICS.keys()]),
        default="frame_f1",
        help="Validation metric used for ranking and band selection (default: frame_f1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used during validation scoring (default: 0.5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum validation score candidates must meet before selection.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["band_diverse", "top_n"],
        default="band_diverse",
        help=(
            "How to choose members after validation scoring: "
            "'band_diverse' keeps an in-band set with a light outer-split diversity preference, "
            "while 'top_n' uses strict score ranking (default: band_diverse)."
        ),
    )
    parser.add_argument(
        "--band-tolerance",
        type=float,
        default=0.03,
        help=(
            "Keep candidates within this absolute metric distance of the best "
            "validation score before diversity/cap filtering (default: 0.03). "
            "Used only with --selection-mode band_diverse."
        ),
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=None,
        help=(
            "Optional maximum number of selected members to write into the ensemble. "
            "Leave unset to keep all candidates that survive the active selection filters."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override for validation scoring.",
    )
    parser.add_argument(
        "--episode-min-frames",
        type=int,
        default=16,
        help="Minimum predicted episode length when using episode metrics (default: 16).",
    )
    parser.add_argument(
        "--episode-max-gap",
        type=int,
        default=3,
        help="Maximum internal gap when using episode metrics (default: 3).",
    )
    parser.add_argument(
        "--episode-overlap-threshold",
        type=float,
        default=0.2,
        help="Episode overlap threshold when using episode metrics (default: 0.2).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for selection artifacts. Defaults to "
            "results/ensembles/<behavior>/selection_<timestamp>/."
        ),
    )
    return parser.parse_args()


def metric_level(metric: str) -> str:
    return "episode" if metric in EPISODE_METRICS else "frame"


def metric_column(metric: str) -> str:
    if metric in FRAME_METRICS:
        return FRAME_METRICS[metric]
    return EPISODE_METRICS[metric]


def discover_source_candidates(source: Path) -> list[dict[str, Any]]:
    sweep_dir, table_path = resolve_source(source)
    rows = discover_runs(table_path)
    candidates = []
    for row in rows:
        manifest_path = infer_manifest_path(row, sweep_dir)
        if manifest_path is None:
            continue
        candidates.append(
            {
                "manifest_path": manifest_path.resolve(),
                "source_kind": "sweep",
                "source_path": str(source.resolve()),
                "sweep_dir": str(sweep_dir.resolve()),
                "table_path": str(table_path.resolve()),
                "outer_id": row.get("outer_id"),
                "outer_seed": row.get("outer_seed"),
                "inner_seed": row.get("inner_seed"),
                "split_path": row.get("split_path"),
                "val_split_path": row.get("val_split_path"),
            }
        )
    return candidates


def collect_candidates(
    *, source_paths: list[Path], manifest_paths: list[Path]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_path in source_paths:
        candidates.extend(discover_source_candidates(source_path))
    for manifest_path in manifest_paths:
        candidates.append(
            {
                "manifest_path": manifest_path.resolve(),
                "source_kind": "manifest",
                "source_path": str(manifest_path.resolve()),
                "sweep_dir": None,
                "table_path": None,
                "outer_id": None,
                "outer_seed": None,
                "inner_seed": None,
                "split_path": None,
                "val_split_path": None,
            }
        )

    unique: dict[Path, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(candidate["manifest_path"], candidate)

    return list(unique.values())


def load_candidate_manifests(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("No candidate manifests were discovered.")
    loaded = []
    for candidate in candidates:
        manifest = load_train_manifest(candidate["manifest_path"])
        loaded.append({**candidate, "manifest": manifest})
    return loaded


def evaluate_candidate_record(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    level = metric_level(args.metric)
    metric_name = metric_column(args.metric)
    _frame_df, metric_rows, _error_rows = evaluate_manifest(
        candidate["manifest"],
        candidate["manifest_path"],
        subset="val",
        threshold=float(args.threshold),
        batch_size=args.batch_size,
        level=level,
        episode_min_frames=int(args.episode_min_frames),
        episode_max_gap=int(args.episode_max_gap),
        episode_overlap_threshold=float(args.episode_overlap_threshold),
    )
    metric_row = next(row for row in metric_rows if row["level"] == level)
    return {
        **candidate,
        "score": float(metric_row[metric_name]),
        "score_level": level,
        "frame_metrics": {k: metric_row.get(k) for k in ("precision", "recall", "f1") if level == "frame"},
        "episode_metrics": {k: metric_row.get(k) for k in ("precision", "recall", "f1") if level == "episode"},
        "metric_row": metric_row,
    }


def score_candidates(
    candidates: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    scored = [evaluate_candidate_record(candidate, args) for candidate in candidates]
    return sorted(scored, key=lambda row: (-row["score"], str(row["manifest_path"])))


def diversity_group(candidate: dict[str, Any]) -> str:
    outer_id = candidate.get("outer_id")
    if outer_id:
        return f"outer:{outer_id}"
    return f"manifest:{candidate['manifest_path']}"


def select_candidate_rows(
    scored_rows: list[dict[str, Any]],
    *,
    selection_mode: str,
    min_score: float | None,
    band_tolerance: float,
    max_members: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not scored_rows:
        raise ValueError("No scored candidates are available for selection.")
    if max_members is not None and max_members < 2:
        raise ValueError("--max-members must be at least 2.")
    if band_tolerance < 0:
        raise ValueError("--band-tolerance must be non-negative.")

    prefiltered = [row for row in scored_rows if min_score is None or row["score"] >= min_score]
    if len(prefiltered) < 2:
        raise ValueError(
            "Fewer than two candidates remain after applying the minimum-score filter."
        )

    best_score = prefiltered[0]["score"]
    band_floor = None
    n_in_band = None

    if selection_mode == "top_n":
        selected = (
            list(prefiltered)
            if max_members is None
            else list(prefiltered[:max_members])
        )
    elif selection_mode == "band_diverse":
        band_floor = best_score - band_tolerance
        band_rows = [row for row in prefiltered if row["score"] >= band_floor]
        n_in_band = int(len(band_rows))
        if len(band_rows) < 2:
            raise ValueError(
                "Fewer than two candidates remain inside the stable selection band."
            )

        if max_members is None:
            selected = list(band_rows)
        else:
            selected = []
            used_groups: set[str] = set()
            for row in band_rows:
                group = diversity_group(row)
                if group in used_groups:
                    continue
                selected.append(row)
                used_groups.add(group)
                if len(selected) >= max_members:
                    break

            if len(selected) < min(max_members, len(band_rows)):
                selected_paths = {row["manifest_path"] for row in selected}
                for row in band_rows:
                    if row["manifest_path"] in selected_paths:
                        continue
                    selected.append(row)
                    selected_paths.add(row["manifest_path"])
                    if len(selected) >= max_members:
                        break
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    if len(selected) < 2:
        raise ValueError("Selection must produce at least two ensemble members.")

    excluded = [row for row in scored_rows if row["manifest_path"] not in {r["manifest_path"] for r in selected}]
    summary = {
        "selection_mode": selection_mode,
        "best_score": float(best_score),
        "band_floor": float(band_floor) if band_floor is not None else None,
        "n_candidates": int(len(scored_rows)),
        "n_after_min_score": int(len(prefiltered)),
        "n_in_band": n_in_band,
        "n_selected": int(len(selected)),
    }
    return selected, excluded, summary


def candidate_csv_rows(
    scored_rows: list[dict[str, Any]],
    selected_paths: set[Path],
    *,
    metric: str,
    selection_mode: str,
    best_score: float,
    band_floor: float | None,
    min_score: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for row in scored_rows:
        metric_row = row["metric_row"]
        rows.append(
            {
                "manifest_path": str(row["manifest_path"]),
                "behavior": row["manifest"]["behavior"],
                "score_metric": metric,
                "score": float(row["score"]),
                "selection_mode": selection_mode,
                "selected": row["manifest_path"] in selected_paths,
                "source_kind": row["source_kind"],
                "source_path": row["source_path"],
                "outer_id": row.get("outer_id"),
                "outer_seed": row.get("outer_seed"),
                "inner_seed": row.get("inner_seed"),
                "split_path": row["manifest"].get("split"),
                "val_split_path": row["manifest"].get("val_split"),
                "frame_precision": metric_row.get("precision") if row["score_level"] == "frame" else None,
                "frame_recall": metric_row.get("recall") if row["score_level"] == "frame" else None,
                "frame_f1": metric_row.get("f1") if row["score_level"] == "frame" else None,
                "episode_precision": metric_row.get("precision") if row["score_level"] == "episode" else None,
                "episode_recall": metric_row.get("recall") if row["score_level"] == "episode" else None,
                "episode_f1": metric_row.get("f1") if row["score_level"] == "episode" else None,
                "best_score": float(best_score),
                "band_floor": float(band_floor) if band_floor is not None else None,
                "min_score": min_score,
            }
        )
    return rows


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.source and not args.manifest:
        raise ValueError("Provide at least one --source or --manifest.")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if args.min_score is not None and not (0.0 <= args.min_score <= 1.0):
        raise ValueError("--min-score must be between 0 and 1 when provided.")
    if args.selection_mode == "top_n" and args.max_members is None and args.min_score is None:
        raise ValueError(
            "top_n without --max-members or --min-score would select every candidate. "
            "Set at least one of those filters explicitly."
        )
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.episode_min_frames <= 0:
        raise ValueError("--episode-min-frames must be positive.")
    if args.episode_max_gap < 0:
        raise ValueError("--episode-max-gap must be non-negative.")

    candidates = collect_candidates(
        source_paths=[Path(p) for p in args.source],
        manifest_paths=[Path(p) for p in args.manifest],
    )
    loaded_candidates = load_candidate_manifests(candidates)
    candidate_paths = [candidate["manifest_path"] for candidate in loaded_candidates]
    candidate_manifests = [candidate["manifest"] for candidate in loaded_candidates]
    shared_signature = validate_selection_candidate_manifests(
        candidate_manifests,
        candidate_paths,
    )

    scored_rows = score_candidates(loaded_candidates, args)
    selected_rows, excluded_rows, selection_stats = select_candidate_rows(
        scored_rows,
        selection_mode=args.selection_mode,
        min_score=args.min_score,
        band_tolerance=float(args.band_tolerance),
        max_members=args.max_members,
    )

    selected_paths = [row["manifest_path"] for row in selected_rows]
    selected_manifests = [row["manifest"] for row in selected_rows]
    behavior = selected_manifests[0]["behavior"]
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(behavior)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_payload = build_ensemble_manifest_payload(
        selected_paths,
        selected_manifests,
        name=args.name,
    )
    ensemble_path = out_dir / "ensemble_manifest.yml"
    save_yaml(ensemble_path, ensemble_payload)

    selected_set = set(selected_paths)
    candidate_rows = candidate_csv_rows(
        scored_rows,
        selected_set,
        metric=args.metric,
        selection_mode=args.selection_mode,
        best_score=selection_stats["best_score"],
        band_floor=selection_stats["band_floor"],
        min_score=args.min_score,
    )
    write_candidate_csv(out_dir / "candidate_scores.csv", candidate_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir.resolve()),
        "ensemble_name": args.name,
        "ensemble_manifest": str(ensemble_path.resolve()),
        "metric": args.metric,
        "selection_mode": args.selection_mode,
        "subset": "val",
        "threshold": float(args.threshold),
        "min_score": args.min_score,
        "band_tolerance": float(args.band_tolerance),
        "max_members": int(args.max_members) if args.max_members is not None else None,
        "episode_settings": {
            "min_pred_frames": int(args.episode_min_frames),
            "max_gap": int(args.episode_max_gap),
            "overlap_threshold": float(args.episode_overlap_threshold),
        },
        "selection_stats": selection_stats,
        "compatibility_signature": shared_signature,
        "sources": {
            "sweeps": [str(Path(p).resolve()) for p in args.source],
            "manifests": [str(Path(p).resolve()) for p in args.manifest],
        },
        "selected_members": [
            {
                "manifest_path": str(row["manifest_path"]),
                "score": float(row["score"]),
                "source_kind": row["source_kind"],
                "outer_id": row.get("outer_id"),
                "inner_seed": row.get("inner_seed"),
            }
            for row in selected_rows
        ],
        "excluded_candidates": [
            {
                "manifest_path": str(row["manifest_path"]),
                "score": float(row["score"]),
                "source_kind": row["source_kind"],
                "outer_id": row.get("outer_id"),
                "inner_seed": row.get("inner_seed"),
            }
            for row in excluded_rows
        ],
        "artifacts": {
            "candidate_scores_csv": str((out_dir / "candidate_scores.csv").resolve()),
            "selection_summary_yml": str((out_dir / "selection_summary.yml").resolve()),
        },
    }
    save_yaml(out_dir / "selection_summary.yml", summary)

    print(
        f"Selected {len(selected_rows)} ensemble members for behavior '{behavior}' "
        f"using validation metric {args.metric}."
    )
    print(f"📝 Wrote {ensemble_path}")
    print(f"📝 Wrote {out_dir / 'candidate_scores.csv'}")
    print(f"📝 Wrote {out_dir / 'selection_summary.yml'}")


if __name__ == "__main__":
    main()
