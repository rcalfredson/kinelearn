from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts.archive_results import (
    build_archive_plan,
    execute_archive,
    looks_like_incomplete_training_run,
    main,
    should_omit,
)


class ArchiveResultsTests(unittest.TestCase):
    def test_should_omit_only_memmap_payloads(self) -> None:
        self.assertTrue(should_omit(Path("train_features.fp32")))
        self.assertTrue(should_omit(Path("test_labels.u8")))
        self.assertFalse(should_omit(Path("train_vids.npy")))
        self.assertFalse(should_omit(Path("train_starts.npy")))
        self.assertFalse(should_omit(Path("train_manifest.yml")))

    def test_build_archive_plan_keeps_metadata_and_prunes_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "genitalia_extension" / "20260402_120000"
            destination = root / "archive" / "genitalia_extension" / "20260402_120000"
            source.mkdir(parents=True, exist_ok=True)

            kept_files = [
                source / "train_manifest.yml",
                source / "best_model.weights.h5",
                source / "train_history.csv",
                source / "train_vids.npy",
                source / "train_starts.npy",
                source / "eval" / "frame_predictions.parquet",
            ]
            omitted_files = [
                source / "train_features.fp32",
                source / "train_labels.u8",
                source / "nested" / "val_features.fp32",
            ]

            for path in kept_files + omitted_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"abc")

            plan = build_archive_plan(source, destination)

            moved_sources = {src for src, _, _ in plan.moved_files}
            omitted_sources = {src for src, _ in plan.omitted_files}
            self.assertEqual(moved_sources, set(kept_files))
            self.assertEqual(omitted_sources, set(omitted_files))
            self.assertEqual(
                {dst for _, dst, _ in plan.moved_files},
                {
                    destination / "train_manifest.yml",
                    destination / "best_model.weights.h5",
                    destination / "train_history.csv",
                    destination / "train_vids.npy",
                    destination / "train_starts.npy",
                    destination / "eval" / "frame_predictions.parquet",
                },
            )
            self.assertEqual(plan.skipped_files, [])
            self.assertEqual(plan.skipped_directories, [])

    def test_looks_like_incomplete_training_run_requires_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "behavior" / "run_partial"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "train_history.csv").write_text("epoch,loss\n1,0.5\n")
            self.assertTrue(looks_like_incomplete_training_run(run_dir))

            (run_dir / "train_manifest.yml").write_text("training_run: {}\n")
            self.assertFalse(looks_like_incomplete_training_run(run_dir))

    def test_build_archive_plan_skips_incomplete_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "behavior" / "run_partial"
            destination = root / "archive" / "behavior" / "run_partial"
            source.mkdir(parents=True, exist_ok=True)

            kept_elsewhere = root / "results" / "behavior" / "run_complete"
            kept_elsewhere.mkdir(parents=True, exist_ok=True)
            (kept_elsewhere / "train_manifest.yml").write_text("training_run: {}\n")

            partial_files = [
                source / "train_history.csv",
                source / "interrupted_model.weights.h5",
                source / "train_features.fp32",
            ]
            for path in partial_files:
                path.write_bytes(b"abc")

            plan = build_archive_plan(root / "results", root / "archive" / "results")

            self.assertEqual(
                dict(plan.skipped_directories),
                {
                    source.resolve(): "incomplete training run (missing train_manifest.yml)",
                },
            )
            self.assertEqual({src for src, _, _ in plan.moved_files}, {kept_elsewhere / "train_manifest.yml"})
            self.assertEqual({src for src, _, _ in plan.skipped_files}, set(partial_files))
            self.assertEqual(plan.omitted_files, [])

    def test_main_dry_run_reports_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "behavior" / "run_a"
            destination = root / "archive" / "behavior" / "run_a"
            source.mkdir(parents=True, exist_ok=True)
            kept = source / "train_manifest.yml"
            omitted = source / "train_features.fp32"
            kept.write_bytes(b"manifest")
            omitted.write_bytes(b"0123456789")

            stdout = io.StringIO()
            old_argv = sys.argv
            sys.argv = [
                "kinelearn-archive-results",
                str(source),
                str(destination),
                "--dry-run",
            ]
            try:
                with contextlib.redirect_stdout(stdout):
                    main()
            finally:
                sys.argv = old_argv

            output = stdout.getvalue()
            self.assertIn("Would move files: 1", output)
            self.assertIn("Would omit memmaps: 1", output)
            self.assertIn("Skipped files: 0", output)
            self.assertTrue(kept.exists())
            self.assertTrue(omitted.exists())
            self.assertFalse(destination.exists())

    def test_main_dry_run_reports_skipped_incomplete_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results"
            destination = root / "archive" / "results"
            incomplete_run = source / "behavior" / "run_partial"
            incomplete_run.mkdir(parents=True, exist_ok=True)
            (incomplete_run / "train_history.csv").write_text("epoch,loss\n")
            (incomplete_run / "train_features.fp32").write_bytes(b"012345")

            stdout = io.StringIO()
            old_argv = sys.argv
            sys.argv = [
                "kinelearn-archive-results",
                str(source),
                str(destination),
                "--dry-run",
                "--verbose",
            ]
            try:
                with contextlib.redirect_stdout(stdout):
                    main()
            finally:
                sys.argv = old_argv

            output = stdout.getvalue()
            self.assertIn("Skipped files: 2", output)
            self.assertIn("Skipped directories: 1", output)
            self.assertIn("SKIPDIR", output)
            self.assertIn("incomplete training run", output)
            self.assertFalse(destination.exists())

    def test_build_archive_plan_skips_unfinished_split_variability_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results"
            destination = root / "archive" / "results"
            sweep_dir = source / "split_variability" / "sweep_a"
            run_dir = sweep_dir / "runs" / "outer_seed1" / "inner_seed2"
            run_dir.mkdir(parents=True, exist_ok=True)

            (source / "behavior" / "run_complete").mkdir(parents=True, exist_ok=True)
            (source / "behavior" / "run_complete" / "train_manifest.yml").write_text(
                "training_run: {}\n"
            )
            (run_dir / "train_manifest.yml").write_text("training_run: {}\n")
            (sweep_dir / "experiment_plan.csv").write_text(
                "outer_id,inner_seed,run_output_dir,command,split_path,val_split_path\n"
                f"outer_seed1,1,{sweep_dir / 'runs' / 'outer_seed1' / 'inner_seed1'},x,a,b\n"
                f"outer_seed1,2,{run_dir},x,a,b\n"
            )
            (sweep_dir / "results_summary.csv").write_text(
                "outer_id,inner_seed,manifest_path\n"
                f"outer_seed1,2,{run_dir / 'train_manifest.yml'}\n"
            )

            plan = build_archive_plan(source, destination)

            self.assertIn(
                (
                    sweep_dir.resolve(),
                    "unfinished split-variability sweep (1/2 completed in results_summary.csv)",
                ),
                plan.skipped_directories,
            )
            moved_sources = {src for src, _, _ in plan.moved_files}
            skipped_sources = {src for src, _, _ in plan.skipped_files}
            self.assertIn(source / "behavior" / "run_complete" / "train_manifest.yml", moved_sources)
            self.assertIn(sweep_dir / "experiment_plan.csv", skipped_sources)
            self.assertIn(run_dir / "train_manifest.yml", skipped_sources)

    def test_unfinished_sweep_skips_external_completed_runs_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results"
            destination = root / "archive" / "results"
            sweep_dir = source / "split_variability" / "legacy_sweep"
            external_run = source / "back_leg_together" / "20260403_083956"
            unrelated_run = source / "back_leg_together" / "20260403_200000"

            sweep_dir.mkdir(parents=True, exist_ok=True)
            external_run.mkdir(parents=True, exist_ok=True)
            unrelated_run.mkdir(parents=True, exist_ok=True)

            (external_run / "train_manifest.yml").write_text("training_run: {}\n")
            (external_run / "best_model.weights.h5").write_bytes(b"abc")
            (unrelated_run / "train_manifest.yml").write_text("training_run: {}\n")

            (sweep_dir / "experiment_plan.csv").write_text(
                "outer_id,inner_seed,split_path,val_split_path,command\n"
                "outer_seed0,0,a,b,x\n"
                "outer_seed0,1,a,c,x\n"
            )
            (sweep_dir / "results_summary.csv").write_text(
                "outer_id,inner_seed,manifest_path,run_output_dir,status\n"
                f"outer_seed0,0,{external_run / 'train_manifest.yml'},{sweep_dir / 'runs' / 'outer_seed0' / 'inner_seed0'},complete_existing\n"
            )

            plan = build_archive_plan(source, destination)

            self.assertIn(
                (
                    external_run.resolve(),
                    f"run referenced by unfinished split-variability sweep {sweep_dir.resolve()}",
                ),
                plan.skipped_directories,
            )
            skipped_sources = {src for src, _, _ in plan.skipped_files}
            moved_sources = {src for src, _, _ in plan.moved_files}
            self.assertIn(external_run / "train_manifest.yml", skipped_sources)
            self.assertIn(external_run / "best_model.weights.h5", skipped_sources)
            self.assertIn(unrelated_run / "train_manifest.yml", moved_sources)

    def test_execute_archive_moves_files_and_removes_omitted_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results" / "behavior" / "run_b"
            destination = root / "archive" / "behavior" / "run_b"
            source.mkdir(parents=True, exist_ok=True)

            kept = source / "nested" / "train_manifest.yml"
            omitted = source / "nested" / "train_features.fp32"
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text("manifest: true\n")
            omitted.write_bytes(b"123456")

            plan = build_archive_plan(source, destination)
            execute_archive(plan, verbose=False)

            self.assertFalse(source.exists())
            self.assertTrue((destination / "nested" / "train_manifest.yml").exists())
            self.assertFalse((destination / "nested" / "train_features.fp32").exists())

    def test_build_archive_plan_allows_existing_gitkeep_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "results"
            destination = root / "archive"
            source.mkdir(parents=True, exist_ok=True)
            destination.mkdir(parents=True, exist_ok=True)

            (source / ".gitkeep").write_text("")
            (source / "behavior" / "run_a").mkdir(parents=True, exist_ok=True)
            (source / "behavior" / "run_a" / "train_manifest.yml").write_text(
                "training_run: {}\n"
            )
            (destination / ".gitkeep").write_text("")

            plan = build_archive_plan(source, destination)

            moved_destinations = {dst for _, dst, _ in plan.moved_files}
            self.assertIn(destination / ".gitkeep", moved_destinations)
            self.assertIn(
                destination / "behavior" / "run_a" / "train_manifest.yml",
                moved_destinations,
            )


if __name__ == "__main__":
    unittest.main()
