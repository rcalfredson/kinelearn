from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core.evaluation import (
    EPISODE_MATCHING_METHOD,
    EPISODE_OVERLAP_DENOMINATOR,
)
from KineLearn.scripts.batch_eval_splits import (
    completed_eval_is_reusable,
    main as batch_eval_main,
    metric_aggregate_rows,
    selected_checkpoint_threshold,
    validate_resume_config,
)


def evaluation_args(**overrides) -> argparse.Namespace:
    values = {
        "manifest": [],
        "subset": "test",
        "level": "both",
        "episode_min_frames": 16,
        "episode_max_gap": 3,
        "episode_overlap_threshold": 0.2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SelectedThresholdTests(unittest.TestCase):
    def test_reads_selected_threshold_from_training_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "train_manifest.yml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "training_run": {
                            "checkpoint_selection": {
                                "enabled": True,
                                "selected": {"threshold": 0.63}
                            }
                        }
                    }
                )
            )

            self.assertEqual(selected_checkpoint_threshold(manifest_path), 0.63)

    def test_missing_selected_threshold_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "train_manifest.yml"
            manifest_path.write_text(
                "training_run:\n  checkpoint_selection:\n    enabled: true\n"
            )

            with self.assertRaisesRegex(ValueError, "no validation-selected"):
                selected_checkpoint_threshold(manifest_path)

    def test_disabled_checkpoint_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "train_manifest.yml"
            manifest_path.write_text("training_run: {}\n")

            with self.assertRaisesRegex(
                ValueError, "does not have checkpoint selection enabled"
            ):
                selected_checkpoint_threshold(manifest_path)


class ResumeEvaluationTests(unittest.TestCase):
    def test_complete_matching_evaluation_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_out = root / "evaluation"
            run_out.mkdir()
            manifest_path = root / "train_manifest.yml"
            manifest_path.write_text("training_run: {}\n")
            for filename in (
                "per_behavior_metrics.csv",
                "frame_predictions.parquet",
                "episode_errors.csv",
            ):
                (run_out / filename).write_text("complete\n")
            summary = {
                "subset": "test",
                "level": "both",
                "threshold": 0.63,
                "manifests": [str(manifest_path.resolve())],
                "evaluation_manifest": None,
                "episode_settings": {
                    "min_pred_frames": 16,
                    "max_gap": 3,
                    "overlap_threshold": 0.2,
                    "matching_method": EPISODE_MATCHING_METHOD,
                    "overlap_denominator": EPISODE_OVERLAP_DENOMINATOR,
                },
            }
            (run_out / "eval_summary.yml").write_text(
                yaml.safe_dump(summary, sort_keys=False)
            )

            self.assertTrue(
                completed_eval_is_reusable(
                    run_out,
                    args=evaluation_args(),
                    manifest_path=manifest_path,
                    threshold=0.63,
                )
            )
            self.assertFalse(
                completed_eval_is_reusable(
                    run_out,
                    args=evaluation_args(),
                    manifest_path=manifest_path,
                    threshold=0.62,
                )
            )

    def test_resume_rejects_changed_batch_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "batch_eval_config.yml"
            config_path.write_text("subset: test\nthreshold: 0.5\n")

            with self.assertRaisesRegex(ValueError, "settings differ"):
                validate_resume_config(
                    config_path, {"subset": "test", "threshold": 0.6}
                )


class AggregateReportTests(unittest.TestCase):
    def test_reports_overall_and_per_outer_metrics(self) -> None:
        rows = [
            {
                "outer_id": "outer_seed0",
                "outer_seed": 0,
                "behavior": "back_leg_together",
                "subset": "test",
                "level": "episode",
                "eval_returncode": 0,
                "threshold": 0.4,
                "f1": 0.8,
                "precision": 0.6,
                "recall": 0.9,
            },
            {
                "outer_id": "outer_seed0",
                "outer_seed": 0,
                "behavior": "back_leg_together",
                "subset": "test",
                "level": "episode",
                "eval_returncode": 0,
                "threshold": 0.6,
                "f1": 0.9,
                "precision": 0.9,
                "recall": 0.9,
            },
            {
                "outer_id": "outer_seed1",
                "outer_seed": 1,
                "behavior": "back_leg_together",
                "subset": "test",
                "level": "episode",
                "eval_returncode": 0,
                "threshold": 0.5,
                "f1": 0.7,
                "precision": 0.8,
                "recall": 0.6,
            },
        ]

        overall = metric_aggregate_rows(rows, by_outer=False)
        by_outer = metric_aggregate_rows(rows, by_outer=True)

        self.assertEqual(len(overall), 1)
        self.assertAlmostEqual(overall[0]["mean_f1"], 0.8)
        self.assertAlmostEqual(overall[0]["mean_threshold"], 0.5)
        self.assertEqual(overall[0]["n_runs_all_metrics_ge_0_80"], 1)
        self.assertFalse(overall[0]["all_metric_means_ge_0_80"])
        self.assertEqual(len(by_outer), 2)
        outer_zero = next(row for row in by_outer if row["outer_seed"] == 0)
        self.assertAlmostEqual(outer_zero["mean_f1"], 0.85)


class BatchEvaluationIntegrationTests(unittest.TestCase):
    def test_selected_threshold_batch_can_resume_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "sweep"
            out_dir = root / "evaluation"
            sweep_dir.mkdir()
            summary_rows = []
            for outer_seed, threshold in ((0, 0.4), (1, 0.6)):
                run_dir = sweep_dir / "runs" / f"outer_seed{outer_seed}" / "inner_seed0"
                run_dir.mkdir(parents=True)
                manifest_path = run_dir / "train_manifest.yml"
                manifest_path.write_text(
                    yaml.safe_dump(
                        {
                            "training_run": {
                                "checkpoint_selection": {
                                    "enabled": True,
                                    "selected": {"threshold": threshold}
                                }
                            }
                        }
                    )
                )
                summary_rows.append(
                    {
                        "outer_id": f"outer_seed{outer_seed}",
                        "outer_seed": outer_seed,
                        "inner_seed": 0,
                        "split_path": f"split_{outer_seed}.yml",
                        "val_split_path": f"val_{outer_seed}.yml",
                        "manifest_path": str(manifest_path.resolve()),
                    }
                )
            with (sweep_dir / "results_summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)

            def fake_eval(command, **_kwargs):
                def value_after(option):
                    return command[command.index(option) + 1]

                run_out = Path(value_after("--out"))
                run_out.mkdir(parents=True, exist_ok=True)
                threshold = float(value_after("--threshold"))
                manifest_path = Path(value_after("--manifest")).resolve()
                eval_summary = {
                    "subset": "test",
                    "level": "episode",
                    "threshold": threshold,
                    "manifests": [str(manifest_path)],
                    "evaluation_manifest": None,
                    "episode_settings": {
                        "min_pred_frames": 16,
                        "max_gap": 3,
                        "overlap_threshold": 0.2,
                        "matching_method": EPISODE_MATCHING_METHOD,
                        "overlap_denominator": EPISODE_OVERLAP_DENOMINATOR,
                    },
                }
                (run_out / "eval_summary.yml").write_text(
                    yaml.safe_dump(eval_summary, sort_keys=False)
                )
                pd.DataFrame(
                    [
                        {
                            "behavior": "back_leg_together",
                            "subset": "test",
                            "level": "episode",
                            "threshold": threshold,
                            "f1": 0.8,
                            "precision": 0.75,
                            "recall": 0.86,
                        }
                    ]
                ).to_csv(run_out / "per_behavior_metrics.csv", index=False)
                (run_out / "frame_predictions.parquet").write_text("complete\n")
                (run_out / "episode_errors.csv").write_text("complete\n")
                return subprocess.CompletedProcess(command, 0, "", "")

            base_argv = [
                "kinelearn-batch-eval-splits",
                str(sweep_dir),
                "--subset",
                "test",
                "--use-selected-threshold",
                "--level",
                "episode",
                "--out-dir",
                str(out_dir),
            ]
            with patch(
                "KineLearn.scripts.batch_eval_splits.subprocess.run",
                side_effect=fake_eval,
            ) as run_mock:
                old_argv = sys.argv
                try:
                    sys.argv = base_argv
                    with contextlib.redirect_stdout(io.StringIO()):
                        batch_eval_main()
                    sys.argv = [*base_argv, "--resume"]
                    with contextlib.redirect_stdout(io.StringIO()):
                        batch_eval_main()
                finally:
                    sys.argv = old_argv

            self.assertEqual(run_mock.call_count, 2)
            raw = pd.read_csv(out_dir / "batch_eval_summary.csv")
            aggregate = pd.read_csv(out_dir / "batch_eval_aggregate.csv")
            outer = pd.read_csv(out_dir / "batch_eval_outer_summary.csv")
            self.assertEqual(sorted(raw["threshold"].tolist()), [0.4, 0.6])
            self.assertEqual(set(raw["evaluation_status"]), {"reused"})
            self.assertAlmostEqual(aggregate.iloc[0]["mean_threshold"], 0.5)
            self.assertEqual(len(outer), 2)


if __name__ == "__main__":
    unittest.main()
