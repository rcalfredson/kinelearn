from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path
import types
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sklearn = types.ModuleType("sklearn")
model_selection = types.ModuleType("sklearn.model_selection")
model_selection.train_test_split = lambda *args, **kwargs: (_ for _ in ()).throw(
    NotImplementedError("train_test_split stub should not be called in these tests")
)
sklearn.model_selection = model_selection
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.model_selection", model_selection)

from KineLearn.scripts.split_variability import (
    build_plan,
    enrich_summary_row_from_manifest,
    inspect_resume_runs,
    main,
)
from KineLearn.scripts import split_variability as split_variability_script


def write_plan_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class SplitVariabilityResumeTests(unittest.TestCase):
    def test_build_plan_passes_execution_overrides_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_path = root / "split.yaml"
            split_path.write_text(
                "train:\n  - a\n  - b\n  - c\n  - d\ntest:\n  - e\n"
            )
            args = types.SimpleNamespace(
                base_split=str(split_path),
                video_list=None,
                outer_seeds=[],
                test_fraction=0.2,
                inner_seeds=[7],
                train_command="kinelearn-train",
                kl_config="config.yaml",
                behavior="back_leg_together",
                features_dir="features",
                seed=0,
                focal_alpha=None,
                keypoint_noise_std=None,
                steps_per_execution=32,
                inference_batch_size=256,
            )
            original_split = split_variability_script.train_test_split
            split_variability_script.train_test_split = (
                lambda values, **_kwargs: (list(values[:-1]), [values[-1]])
            )
            try:
                runs = build_plan(args, root / "sweep", val_fraction=0.25)
            finally:
                split_variability_script.train_test_split = original_split

            command = runs[0]["command"]
            self.assertEqual(
                command[command.index("--steps-per-execution") + 1], "32"
            )
            self.assertEqual(
                command[command.index("--inference-batch-size") + 1], "256"
            )

    def test_summary_metrics_are_rehydrated_from_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "train_manifest.yml"
            manifest_path.write_text(
                "training_run:\n"
                "  best_epoch_by_val_loss: 1\n"
                "  epochs_completed: 7\n"
                "  checkpoint_selection:\n"
                "    enabled: true\n"
                "    selected:\n"
                "      epoch: 2\n"
                "      threshold: 0.63\n"
                "      f1: 0.72\n"
                "      precision: 0.64\n"
                "      recall: 0.82\n"
                "  test_metrics:\n"
                "    loss: 0.01\n"
            )

            row = enrich_summary_row_from_manifest({}, manifest_path)

            self.assertEqual(row["best_epoch_by_val_loss"], 1)
            self.assertEqual(row["epochs_completed"], 7)
            self.assertEqual(row["best_epoch_by_checkpoint_selection"], 2)
            self.assertEqual(row["selected_threshold"], 0.63)
            self.assertEqual(row["selected_val_episode_f1"], 0.72)
            self.assertEqual(row["selected_val_episode_precision"], 0.64)
            self.assertEqual(row["selected_val_episode_recall"], 0.82)
            self.assertEqual(row["test_loss"], 0.01)

    def test_inspect_resume_runs_classifies_complete_and_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "results" / "split_variability" / "resume_case"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            run_a = sweep_dir / "runs" / "outer_seed1" / "inner_seed10"
            run_b = sweep_dir / "runs" / "outer_seed1" / "inner_seed11"
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)
            (run_a / "train_manifest.yml").write_text("training_run: {}\n")
            (run_b / "partial.tmp").write_text("partial\n")

            plan_rows = [
                {
                    "outer_id": "outer_seed1",
                    "outer_seed": "1",
                    "inner_seed": "10",
                    "split_path": str((sweep_dir / "splits" / "a.yaml").resolve()),
                    "val_split_path": str((sweep_dir / "splits" / "a_val.yaml").resolve()),
                    "train_count": "1",
                    "val_count": "1",
                    "test_count": "1",
                    "run_output_dir": str(run_a.resolve()),
                    "command": "kinelearn-train --split split_a --val-split val_a",
                },
                {
                    "outer_id": "outer_seed1",
                    "outer_seed": "1",
                    "inner_seed": "11",
                    "split_path": str((sweep_dir / "splits" / "b.yaml").resolve()),
                    "val_split_path": str((sweep_dir / "splits" / "b_val.yaml").resolve()),
                    "train_count": "1",
                    "val_count": "1",
                    "test_count": "1",
                    "run_output_dir": str(run_b.resolve()),
                    "command": "kinelearn-train --split split_b --val-split val_b",
                },
            ]

            inspected = inspect_resume_runs(sweep_dir, plan_rows)

            self.assertEqual(inspected[0]["state"], "complete")
            self.assertEqual(
                inspected[0]["manifest_path"],
                str((run_a / "train_manifest.yml").resolve()),
            )
            self.assertEqual(inspected[1]["state"], "pending")
            self.assertTrue(inspected[1]["has_partial_run_dir"])

    def test_main_resume_dry_run_reports_without_deleting_partial_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "results" / "split_variability" / "resume_dry_run"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            complete_dir = sweep_dir / "runs" / "outer_seed1" / "inner_seed10"
            partial_dir = sweep_dir / "runs" / "outer_seed1" / "inner_seed11"
            complete_dir.mkdir(parents=True, exist_ok=True)
            partial_dir.mkdir(parents=True, exist_ok=True)
            (complete_dir / "train_manifest.yml").write_text("training_run: {}\n")
            (partial_dir / "partial.tmp").write_text("partial\n")

            write_plan_csv(
                sweep_dir / "experiment_plan.csv",
                [
                    {
                        "outer_id": "outer_seed1",
                        "outer_seed": "1",
                        "inner_seed": "10",
                        "split_path": str((sweep_dir / "splits" / "a.yaml").resolve()),
                        "val_split_path": str((sweep_dir / "splits" / "a_val.yaml").resolve()),
                        "train_count": "1",
                        "val_count": "1",
                        "test_count": "1",
                        "run_output_dir": str(complete_dir.resolve()),
                        "command": "kinelearn-train --split split_a --val-split val_a",
                    },
                    {
                        "outer_id": "outer_seed1",
                        "outer_seed": "1",
                        "inner_seed": "11",
                        "split_path": str((sweep_dir / "splits" / "b.yaml").resolve()),
                        "val_split_path": str((sweep_dir / "splits" / "b_val.yaml").resolve()),
                        "train_count": "1",
                        "val_count": "1",
                        "test_count": "1",
                        "run_output_dir": str(partial_dir.resolve()),
                        "command": "kinelearn-train --split split_b --val-split val_b",
                    },
                ],
            )

            stdout = io.StringIO()
            old_argv = sys.argv
            sys.argv = [
                "kinelearn-split-variability",
                "--resume",
                str(sweep_dir),
            ]
            try:
                with contextlib.redirect_stdout(stdout):
                    main()
            finally:
                sys.argv = old_argv

            output = stdout.getvalue()
            self.assertIn("Resume summary: planned=2, complete=1, pending=1, incomplete=1", output)
            self.assertIn("Would rerun:", output)
            self.assertTrue((partial_dir / "partial.tmp").exists())

    def test_main_resume_execute_reruns_only_pending_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "results" / "split_variability" / "resume_execute"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            complete_dir = sweep_dir / "runs" / "outer_seed1" / "inner_seed10"
            partial_dir = sweep_dir / "runs" / "outer_seed1" / "inner_seed11"
            log_path = root / "fake_train_invocations.txt"
            complete_dir.mkdir(parents=True, exist_ok=True)
            partial_dir.mkdir(parents=True, exist_ok=True)
            (complete_dir / "train_manifest.yml").write_text("training_run: {}\n")
            (partial_dir / "partial.tmp").write_text("partial\n")

            trainer = root / "fake_train.py"
            trainer.write_text(
                "#!/usr/bin/env python3\n"
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--out-dir', required=True)\n"
                "parser.add_argument('--split')\n"
                "parser.add_argument('--val-split')\n"
                "args, _ = parser.parse_known_args()\n"
                f"log_path = Path({str(log_path)!r})\n"
                "log_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "with open(log_path, 'a') as f:\n"
                "    f.write(args.out_dir + '\\n')\n"
                "out_dir = Path(args.out_dir)\n"
                "out_dir.mkdir(parents=True, exist_ok=True)\n"
                "manifest_path = out_dir / 'train_manifest.yml'\n"
                "manifest_path.write_text('training_run:\\n  test_metrics: {}\\n')\n"
                "print(f'📝 Wrote {manifest_path}')\n"
            )
            os.chmod(trainer, 0o755)

            write_plan_csv(
                sweep_dir / "experiment_plan.csv",
                [
                    {
                        "outer_id": "outer_seed1",
                        "outer_seed": "1",
                        "inner_seed": "10",
                        "split_path": str((sweep_dir / "splits" / "a.yaml").resolve()),
                        "val_split_path": str((sweep_dir / "splits" / "a_val.yaml").resolve()),
                        "train_count": "1",
                        "val_count": "1",
                        "test_count": "1",
                        "run_output_dir": str(complete_dir.resolve()),
                        "command": f"{trainer} --split split_a --val-split val_a --out-dir {complete_dir}",
                    },
                    {
                        "outer_id": "outer_seed1",
                        "outer_seed": "1",
                        "inner_seed": "11",
                        "split_path": str((sweep_dir / "splits" / "b.yaml").resolve()),
                        "val_split_path": str((sweep_dir / "splits" / "b_val.yaml").resolve()),
                        "train_count": "1",
                        "val_count": "1",
                        "test_count": "1",
                        "run_output_dir": str(partial_dir.resolve()),
                        "command": f"{trainer} --split split_b --val-split val_b --out-dir {partial_dir}",
                    },
                ],
            )

            old_argv = sys.argv
            sys.argv = [
                "kinelearn-split-variability",
                "--resume",
                str(sweep_dir),
                "--execute",
            ]
            try:
                main()
            finally:
                sys.argv = old_argv

            invocations = log_path.read_text().strip().splitlines()
            self.assertEqual(invocations, [str(partial_dir)])
            self.assertFalse((partial_dir / "partial.tmp").exists())
            self.assertTrue((partial_dir / "train_manifest.yml").exists())

            with open(sweep_dir / "results_summary.csv", newline="") as f:
                summary_rows = list(csv.DictReader(f))
            status_by_seed = {row["inner_seed"]: row["status"] for row in summary_rows}
            self.assertEqual(status_by_seed["10"], "complete_existing")
            self.assertEqual(status_by_seed["11"], "rerun")


if __name__ == "__main__":
    unittest.main()
