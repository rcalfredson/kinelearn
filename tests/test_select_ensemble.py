from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core.manifests import validate_selection_candidate_manifests
from KineLearn.scripts.select_ensemble import main as select_ensemble_main
from KineLearn.scripts.select_ensemble import select_candidate_rows


def write_train_manifest(
    path: Path,
    *,
    behavior: str = "genitalia_extension",
    split: str | None = None,
    val_split: str | None = None,
    seed: int = 0,
    learning_rate: float = 1e-3,
) -> None:
    weights_path = path.parent / "best_model.weights.h5"
    weights_path.write_text("weights\n")
    payload = {
        "kl_config": "/tmp/config.yaml",
        "behavior": behavior,
        "behavior_idx": 0,
        "label_columns": [behavior],
        "feature_columns": ["feat_1", "feat_2"],
        "window": {"size": 3, "stride": 1},
        "artifacts": {},
        "feature_selection": {
            "include_absolute_coordinates": False,
            "n_input_features": 2,
        },
        "training": {
            "final_zero_fill": False,
            "seed": int(seed),
            "val_fraction": 0.1,
            "learning_rate": float(learning_rate),
            "loss": "focal",
        },
        "focal": {"alpha": 0.7, "gamma": 2.0},
        "training_run": {
            "evaluation_weights": str(weights_path.resolve()),
        },
        "run_dir": str(path.parent.resolve()),
        "split": split,
        "val_split": val_split,
        "resolved_stems": {
            "train": ["train_a"],
            "val": ["val_a"],
            "test": ["test_a"],
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class SelectEnsembleTests(unittest.TestCase):
    def test_selection_compatibility_allows_split_and_seed_differences(self) -> None:
        first = {
            "kl_config": "/tmp/config.yaml",
            "behavior": "genitalia_extension",
            "behavior_idx": 0,
            "label_columns": ["genitalia_extension"],
            "feature_columns": ["feat_1", "feat_2"],
            "window": {"size": 3, "stride": 1},
            "feature_selection": {
                "include_absolute_coordinates": False,
                "n_input_features": 2,
            },
            "training": {
                "final_zero_fill": False,
                "seed": 0,
                "val_fraction": 0.1,
                "learning_rate": 1e-3,
                "loss": "focal",
            },
            "focal": {"alpha": 0.7, "gamma": 2.0},
        }
        second = {
            **first,
            "training": {
                **first["training"],
                "seed": 99,
                "val_fraction": 0.2,
            },
            "split": "/tmp/split_b.yaml",
            "val_split": "/tmp/val_b.yaml",
        }

        signature = validate_selection_candidate_manifests(
            [first, second],
            [Path("/tmp/a/train_manifest.yml"), Path("/tmp/b/train_manifest.yml")],
        )

        self.assertEqual(signature["behavior"], "genitalia_extension")
        self.assertEqual(signature["training_recipe"]["learning_rate"], 1e-3)

    def test_select_candidate_rows_prefers_outer_split_diversity(self) -> None:
        rows = [
            {"manifest_path": Path("/tmp/a.yml"), "score": 0.90, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/b.yml"), "score": 0.89, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/c.yml"), "score": 0.88, "outer_id": "outer_b"},
        ]

        selected, excluded, summary = select_candidate_rows(
            rows,
            selection_mode="band_diverse",
            min_score=None,
            band_tolerance=0.03,
            max_members=2,
        )

        self.assertEqual([row["manifest_path"] for row in selected], [Path("/tmp/a.yml"), Path("/tmp/c.yml")])
        self.assertEqual([row["manifest_path"] for row in excluded], [Path("/tmp/b.yml")])
        self.assertEqual(summary["n_in_band"], 3)
        self.assertEqual(summary["selection_mode"], "band_diverse")

    def test_select_candidate_rows_top_n_uses_strict_score_order(self) -> None:
        rows = [
            {"manifest_path": Path("/tmp/a.yml"), "score": 0.90, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/b.yml"), "score": 0.89, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/c.yml"), "score": 0.88, "outer_id": "outer_b"},
        ]

        selected, excluded, summary = select_candidate_rows(
            rows,
            selection_mode="top_n",
            min_score=None,
            band_tolerance=0.03,
            max_members=2,
        )

        self.assertEqual([row["manifest_path"] for row in selected], [Path("/tmp/a.yml"), Path("/tmp/b.yml")])
        self.assertEqual([row["manifest_path"] for row in excluded], [Path("/tmp/c.yml")])
        self.assertIsNone(summary["band_floor"])
        self.assertIsNone(summary["n_in_band"])
        self.assertEqual(summary["selection_mode"], "top_n")

    def test_select_candidate_rows_without_cap_keeps_all_band_members(self) -> None:
        rows = [
            {"manifest_path": Path("/tmp/a.yml"), "score": 0.90, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/b.yml"), "score": 0.89, "outer_id": "outer_a"},
            {"manifest_path": Path("/tmp/c.yml"), "score": 0.88, "outer_id": "outer_b"},
        ]

        selected, excluded, summary = select_candidate_rows(
            rows,
            selection_mode="band_diverse",
            min_score=None,
            band_tolerance=0.03,
            max_members=None,
        )

        self.assertEqual(
            [row["manifest_path"] for row in selected],
            [Path("/tmp/a.yml"), Path("/tmp/b.yml"), Path("/tmp/c.yml")],
        )
        self.assertEqual(excluded, [])
        self.assertEqual(summary["n_selected"], 3)

    def test_select_ensemble_cli_writes_manifest_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "results" / "split_variability" / "ge_fixed"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            run_a = sweep_dir / "runs" / "outer_seed1" / "inner_seed10"
            run_b = sweep_dir / "runs" / "outer_seed2" / "inner_seed11"
            run_c = root / "extra_run"
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)
            run_c.mkdir(parents=True, exist_ok=True)

            manifest_a = run_a / "train_manifest.yml"
            manifest_b = run_b / "train_manifest.yml"
            manifest_c = run_c / "train_manifest.yml"
            write_train_manifest(manifest_a, split="/tmp/split_a.yaml", val_split="/tmp/val_a.yaml", seed=0)
            write_train_manifest(manifest_b, split="/tmp/split_b.yaml", val_split="/tmp/val_b.yaml", seed=1)
            write_train_manifest(manifest_c, split="/tmp/split_c.yaml", val_split="/tmp/val_c.yaml", seed=2)

            with open(sweep_dir / "results_summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["outer_id", "outer_seed", "inner_seed", "manifest_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "outer_id": "outer_seed1",
                        "outer_seed": "1",
                        "inner_seed": "10",
                        "manifest_path": str(manifest_a.resolve()),
                    }
                )
                writer.writerow(
                    {
                        "outer_id": "outer_seed2",
                        "outer_seed": "2",
                        "inner_seed": "11",
                        "manifest_path": str(manifest_b.resolve()),
                    }
                )

            scores = {
                str(manifest_a.resolve()): 0.90,
                str(manifest_b.resolve()): 0.88,
                str(manifest_c.resolve()): 0.81,
            }

            def fake_evaluate(candidate, args):
                score = scores[str(candidate["manifest_path"])]
                return {
                    **candidate,
                    "score": score,
                    "score_level": "frame",
                    "metric_row": {"level": "frame", "precision": score, "recall": score, "f1": score},
                }

            out_dir = root / "selected_ensemble"
            old_argv = sys.argv
            sys.argv = [
                "kinelearn-select-ensemble",
                "--source",
                str(sweep_dir),
                "--manifest",
                str(manifest_c),
                "--name",
                "ge_selected",
                "--selection-mode",
                "band_diverse",
                "--max-members",
                "2",
                "--band-tolerance",
                "0.03",
                "--out-dir",
                str(out_dir),
            ]
            try:
                with patch("KineLearn.scripts.select_ensemble.evaluate_candidate_record", side_effect=fake_evaluate):
                    with contextlib.redirect_stdout(io.StringIO()):
                        select_ensemble_main()
            finally:
                sys.argv = old_argv

            ensemble_path = out_dir / "ensemble_manifest.yml"
            summary_path = out_dir / "selection_summary.yml"
            csv_path = out_dir / "candidate_scores.csv"
            self.assertTrue(ensemble_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(csv_path.exists())

            ensemble = yaml.safe_load(ensemble_path.read_text())
            self.assertEqual(ensemble["aggregation"]["n_members"], 2)
            member_paths = [row["manifest_path"] for row in ensemble["members"]]
            self.assertEqual(member_paths, [str(manifest_a.resolve()), str(manifest_b.resolve())])

            summary = yaml.safe_load(summary_path.read_text())
            self.assertEqual(summary["metric"], "frame_f1")
            self.assertEqual(summary["selection_mode"], "band_diverse")
            self.assertEqual(len(summary["selected_members"]), 2)


if __name__ == "__main__":
    unittest.main()
